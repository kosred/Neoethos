# Canonical Native Discovery Entry Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one explicit Native-CUDA-only, contract-referenced Generation-0 research entry shared by CLI, TUI, headless app, and Desktop UI, with a content-addressed dual-receipt ResearchOnly result.

**Architecture:** `neoethos-search` owns contract loading, private checked-request resolution, exact generation pinning, staged V5 execution, result sealing, and publication. App/CLI/UI layers only adapt explicit operator inputs into the shared from-reference boundary; all legacy Discovery routes remain separate and unchanged.

**Tech Stack:** Rust 2024, serde/serde_json, SHA-256, existing canonical dataset receipts, staged V5 resident CUDA Data/population APIs, Tokio/Axum, Clap, Ratatui, React/TypeScript, RTX3090 CUDA validation.

---

## Execution rules

- Follow `@superpowers:test-driven-development` for every behavior change.
- Use a fresh subagent for each task and run correctness review followed by
  quality review before advancing.
- Do not edit `crates/neoethos-search/src/discovery.rs` or
  `crates/neoethos-search/src/genetic/search_engine.rs` unless a RED proves a
  required public seam is absent and review approves that expansion.
- Never run two Cargo jobs concurrently in `/workspace/forex-ai`.
- Save complete command output under
  `target/audit-logs/canonical-native-discovery-entry/` and review the whole
  log, including INFO and WARNING lines.
- The VPS repository has no `.git` directory. Do not run or claim `git add`,
  commit, branch, merge, or diff checkpoints. Before and after every chunk,
  record SHA-256 manifests for all touched files. A hash manifest is evidence
  of exact bytes, not evidence of version-control integration.
- Keep every implementation/review chunk below 1,000 changed lines. If a chunk
  grows beyond that, split it before implementation.
- Before starting a task, record its estimated changed-line budget in the
  chunk log. Tasks 1–11 are separate implementation units; no worker may land
  two tasks as one patch when their combined estimate reaches 1,000 lines.
- Do not start Chunk 4A until Chunks 1A–3 have an RTX GREEN through the shared
  Search executor.

## File responsibility map

**Search domain**

- Create `crates/neoethos-search/src/canonical_native_root_io_v1.rs` for
  handle-rooted no-follow reads and generalized crash-safe create-new
  publication.
- Create `crates/neoethos-search/src/canonical_research_contract_export_v1.rs`
  for one-handle, 512 MiB-bounded streaming extraction and safe publication;
  CLI does not own this authority.
- Create `crates/neoethos-search/src/canonical_native_discovery_request_v1.rs`
  for the artifact reference, handle-rooted bounded loader, startup-settings
  authority, exact-series derivation, allowed overrides, checked P/K/result
  caps, and checked request.
- Create `crates/neoethos-search/src/canonical_native_runtime_authority_v1.rs`
  for a private-construction runtime-install receipt, same-Settings installed-
  override capture/revalidation, and the mutable migration check.
- Create `crates/neoethos-search/src/canonical_native_generation_zero_result_v1.rs`
  for the dual-receipt artifact, validation, identity, and content-addressed
  publication.
- Create `crates/neoethos-search/src/canonical_native_discovery_run_v1.rs`
  for the sole production staged-V5 executor.
- Modify `crates/neoethos-search/src/lib.rs` only for versioned exports.

**Application adapter**

- Create `crates/neoethos-app/src/app_services/canonical_native_discovery.rs`
  and its sibling test module.
- Put the single production `InProcessSearchRuntimeLeaseManagerV1` in
  `app_services/jobs.rs`; retrofit the direct Discovery, Training, validation,
  app-main, engine-control, and federation call sites so a move-only token is a
  required service argument rather than optional `AppApiState` policy.
- Modify app event/state/status/router/headless files only to add a separate
  `CanonicalNativeResearch` lane and retain the exact startup Settings,
  runtime-install receipt, and production manager Arc.

**CLI/TUI adapter**

- Create `crates/neoethos-cli/src/canonical_native_discovery.rs`.
- Create `crates/neoethos-cli/src/canonical_research_contract_export.rs`.
- Modify `crates/neoethos-cli/src/main.rs` for one new subcommand and help.
- Modify the existing TUI form/discover page to add an explicit route selector;
  legacy batch remains the default.

**Desktop adapter**

- Modify `desktop/src/apiContracts.ts`, `desktop/src/api.ts`, and
  `desktop/src/screens/Discovery.tsx` for a separate Native Gen0 action.
- Modify `desktop/src-tauri/src/lib.rs` only to pass the already-loaded startup
  Settings snapshot into app API state; do not reload it after runtime install.
- Modify only focused Desktop tests; do not rewrite the existing Discovery
  queue.

## Chunk 1A: Handle-rooted saved-contract reference

### Task 1: RED-lock root containment and exact-byte loading

**Files:**

- Create: `crates/neoethos-search/tests/canonical_native_discovery_contract_ref_v1.rs`
- Create: `crates/neoethos-search/tests/canonical_native_root_io_v1_contract.rs`
- Create: `crates/neoethos-search/src/canonical_native_root_io_v1.rs`
- Create: `crates/neoethos-search/src/canonical_native_discovery_request_v1.rs`
- Modify: `crates/neoethos-search/Cargo.toml`
- Modify only if dependency resolution changes exact bytes: `Cargo.lock`
- Modify: `crates/neoethos-search/src/lib.rs`

**Estimated implementation delta:** at most 750 changed lines. V1 implements
the production root boundary on Linux; every non-Linux public entry is a typed
fail-closed stub. Windows handle-relative reparse support is a later chunk and
must not be claimed from cfg-elided source.

- [ ] **Step 1: Record pre-edit hashes**

Run:

```bash
cd /workspace/forex-ai
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-1a
sha256sum crates/neoethos-search/src/lib.rs \
  > target/audit-logs/canonical-native-discovery-entry/chunk-1a/pre.sha256
```

Expected: one lowercase SHA-256 line; new files do not exist yet.

- [ ] **Step 2: Write the failing loader tests**

Cover all of these independently:

- valid root-relative regular file and exact lowercase SHA;
- absolute path, prefix, `.`, `..`, empty/control component, and root escape;
- symlink component and symlink final file;
- component/final-target swap between validation and open, with the opened
  handle rejected when its final identity/path is outside the retained root;
- directory/FIFO/non-regular final target;
- `8 MiB + 1` bytes;
- wrong expected hash and noncanonical expected hash;
- unknown JSON field/schema version and malformed contract;
- valid file mutated after loading does not change the loaded contract; and
- `contract.validate_against_receipt(contract.input_receipt())` rejects a
  symbol/anchor mismatch that plain `validate()` would miss.

Use this public shape in the RED:

```rust
pub struct CanonicalResearchContractArtifactRefV1 { /* private fields */ }

impl CanonicalResearchContractArtifactRefV1 {
    pub fn checked_new(relative_path: impl Into<String>, expected_sha256: impl Into<String>)
        -> Result<Self, CanonicalNativeDiscoveryRequestErrorV1>;
}

pub fn load_canonical_research_contract_artifact_v1(
    canonical_root: &SealedCanonicalRootV1,
    reference: CanonicalResearchContractArtifactRefV1,
) -> Result<LoadedCanonicalResearchContractV1, CanonicalNativeDiscoveryRequestErrorV1>;
```

- [ ] **Step 3: Run RED and retain the complete log**

Run:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_discovery_contract_ref_v1 \
  --test canonical_native_root_io_v1_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-1a/red.log
test ${PIPESTATUS[0]} -ne 0
```

Expected: FAIL only because the versioned types/functions are absent.

- [ ] **Step 4: Implement the minimal bounded loader**

Implement private fields, deny-unknown-fields wire decoding, canonical path
component checks, and a retained physical-root handle/identity. Traverse and
open with platform no-follow semantics, then verify the final opened handle is
a stable regular file contained by that same root before the single bounded
read. Path-only canonicalize-then-open is not acceptable. Compare the exact
SHA, deserialize `CanonicalTrendbarResearchExecutionContractV3`, then call
both `validate()` and `validate_against_receipt(input_receipt())`. Return a
move-only loaded value. Do not install an ambient contract and do not expose
raw construction.

Implement Linux traversal with root-relative directory/file descriptors and
no-follow/close-on-exec flags (add the same pinned `libc` version already used
by `neoethos-data` as a direct target dependency). Under
`cfg(not(target_os = "linux"))`, return typed `UnsupportedPlatform` before root
resolution/open. Do not add or claim untested Windows kernel32 traversal. Do
not reuse
`canonical_full_run::read_regular_file_with_limit`; it metadata-checks then
performs a second path open.

- [ ] **Step 5: Run GREEN and warning-denied library checks**

Run:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_discovery_contract_ref_v1 \
  --test canonical_native_root_io_v1_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-1a/green.log
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-search --features gpu-cuda --all-targets \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-1a/cuda-check.log
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-search --features gpu-b-adapter --all-targets \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-1a/adapter-check.log
```

Expected: all Linux tests pass; both CUDA and non-CUDA adapter graphs are
warning clean. A source/runtime RED proves the non-Linux stub returns
`UnsupportedPlatform` before any root/path function. Record Windows
handle-relative reparse support as a separate future version, not V1 evidence.

- [ ] **Step 6: Write `chunk-1a/post.sha256` and obtain review approval**

Include the root-I/O module/tests, loader/request module, `Cargo.toml`, changed
`Cargo.lock` if any, and `lib.rs`.

## Chunk 1B: Exact source pin, startup authority, and checked request

### Task 2: Derive the exact source pin and resolve the request centrally

**Files:**

- Modify: `crates/neoethos-search/src/canonical_native_discovery_request_v1.rs`
- Create: `crates/neoethos-search/src/canonical_native_runtime_authority_v1.rs`
- Modify: `crates/neoethos-search/src/lib.rs`
- Modify: `crates/neoethos-search/tests/canonical_native_discovery_contract_ref_v1.rs`
- Create: `crates/neoethos-search/tests/canonical_native_discovery_request_v1.rs`
- Create: `crates/neoethos-search/tests/canonical_native_runtime_authority_v1.rs`

**Estimated implementation delta:** at most 900 changed lines. Keep the
runtime-authority module and request resolver as independently reviewed patches
even though they share this checkpoint.

- [ ] **Step 1: Write the failing request tests**

Require:

- exact projection binding -> `SelectedDatasetGenerationV1` conversion;
- exactly one anchor and one binding per timeframe;
- exact generation/manifest binding compared with the current manifest under
  the publication lock, with no unsealed current selection or substitution;
- current-at-acquisition succeeds; an advanced current manifest returns typed
  `ExactDatasetGenerationConflict` before Data allocation and never opens the
  saved old generation;
- exact startup-Settings identity equals the authority installed before
  Search/Data runtime initialization;
- a private-construction `CanonicalNativeRuntimeInstallReceiptV1` exists only
  after the Gen0-consumed Search and Data installer functions were invoked and
  their current snapshots matched the same startup Settings; compiled-default
  getter values without this receipt are rejected;
- canonical root equals `startup_settings.system.data_dir`; callers cannot
  supply another root;
- contract/settings symbol and account equality;
- only population, population-auto, and max-indicators overrides;
- fixed `FeatureProfile::Standard` and `GenerationZeroOnly`;
- raw and clamped legacy generation counts preserved/hashed as unused
  full-search evidence and not used as scope authority, rejected, or mutated;
- absent session-spread curve;
- configured/default cost band preserved as explicitly unused Gen0 evidence,
  not cleared and not rejected;
- disabled adaptive thresholds, ATR-scaled bounds, minimum-history, and ledger;
- mutable migration is already false at resolve and is rechecked before
  preflight; no route calls `set_migration_enabled(false)` to override mesh
  state;
- no-op row cap and prefilter; and
- named configured/resolved population, term, string/vector, source-count, and
  result-byte caps with checked worst-case result arithmetic; and
- unsupported policy fails before the native preflight callback is invoked.

Runtime-authority tests mutate each Gen0-consumed installed/snapshotted class
independently (Data normalization and feature-cube policy, genetic, strategy
evaluation, backtest, SMC, stop-target/adaptive, gene-bound, and seen-memory
inputs) and require mismatch before preflight. Add a migration-toggle-after-
resolution RED proving the second check catches a federation race.

Target API:

```rust
pub fn resolve_canonical_native_discovery_request_v1(
    startup_settings: &neoethos_core::Settings,
    runtime_install_receipt: &CanonicalNativeRuntimeInstallReceiptV1,
    contract_ref: CanonicalResearchContractArtifactRefV1,
    overrides: CanonicalNativeGenerationZeroOverridesV1,
) -> Result<CanonicalNativeDiscoveryRequestV1, CanonicalNativeDiscoveryRequestErrorV1>;
```

- [ ] **Step 2: Run RED**

Run:

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-1b
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_discovery_request_v1 \
  --test canonical_native_runtime_authority_v1 \
  --no-fail-fast \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-1b/request-red.log
test ${PIPESTATUS[0]} -ne 0
```

Expected: FAIL only on the absent request/runtime-authority contracts.

- [ ] **Step 3: Implement exact-series derivation and checked overrides**

Reuse
`canonical_pinned_source_projection_from_search_receipt_v1`; do not reproduce
its digest formula. Convert projection manifest hashes to canonical lowercase
hex and seal `CanonicalDatasetSeriesReceiptV1`. Build config only through
`DiscoveryConfig::try_from_settings_for_canonical_trendbar_research`. Seal a
distinct typed Gen0 scope and never use `config.generations` as permission to
continue. Preserve/hash the raw and clamped legacy generation counts as unused
full-search evidence; do not require zero and do not change them. Verify
installed runtime authority against the same
startup Settings, derive the root from `settings.system.data_dir`, preserve the
unused cost band, and validate the V1 policy/caps without silently changing it.
The typed `GenerationZeroOnly` scope hardcodes the runner boundary.

Add
`install_and_seal_canonical_native_runtime_authority_v1(settings)` as the only
constructor for `CanonicalNativeRuntimeInstallReceiptV1`. It invokes the
existing Search and Data runtime installers, then compares every Gen0-consumed
current snapshot against the same Settings before sealing a domain-separated
identity. A caller cannot construct a receipt from getters alone. Repeated
same-Settings installation returns the same identity; prior conflicting
installation fails. The resolver consumes a borrow of this receipt and
revalidates its identity/snapshots before artifact read and preflight.

- [ ] **Step 4: Run focused GREEN and source contracts**

Run:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_discovery_contract_ref_v1 \
  --test canonical_native_root_io_v1_contract \
  --test canonical_native_discovery_request_v1 \
  --test canonical_native_runtime_authority_v1 \
  --no-fail-fast \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-1b/request-green.log
test ${PIPESTATUS[0]} -eq 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda --lib \
  resident_population_auto_sizing_receipt_v2_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-1b/v2-regression.log
test ${PIPESTATUS[0]} -eq 0
```

Expected: all GREEN; no raw CPU/GPU receipt equality check.

- [ ] **Step 5: Write the Chunk-1B SHA checkpoint**

Run:

```bash
sha256sum \
  crates/neoethos-search/src/canonical_native_discovery_request_v1.rs \
  crates/neoethos-search/src/canonical_native_runtime_authority_v1.rs \
  crates/neoethos-search/src/lib.rs \
  crates/neoethos-search/tests/canonical_native_discovery_contract_ref_v1.rs \
  crates/neoethos-search/tests/canonical_native_discovery_request_v1.rs \
  crates/neoethos-search/tests/canonical_native_runtime_authority_v1.rs \
  > target/audit-logs/canonical-native-discovery-entry/chunk-1b/post.sha256
```

Dispatch review for Chunk 1B and do not continue until approved.

## Chunk 2: Dual-receipt Generation-0 result and publisher

### Task 3: RED-lock schema semantics and non-authority boundaries

**Files:**

- Create: `crates/neoethos-search/src/canonical_native_generation_zero_result_v1.rs`
- Create: `crates/neoethos-search/tests/canonical_native_generation_zero_result_v1.rs`
- Modify: `crates/neoethos-search/src/canonical_native_root_io_v1.rs`
- Modify: `crates/neoethos-search/src/lib.rs`

**Estimated implementation delta:** at most 900 changed lines. Split the wire
sealer and filesystem publisher into separate reviewed patches if the estimate
rises to 1,000.

- [ ] **Step 1: Record pre-edit hashes**

Capture `lib.rs` and mark both new files `MISSING` in
`chunk-2/pre.sha256`.

- [ ] **Step 2: Write schema/source REDs**

The tests must require private construction and these exact serialized facts:

```text
scope=generation_zero_only
artifact_class=research_only
promotion_eligibility=not_promotion_eligible
authorization_issued=false
consumer_completion_confirmed=true
replay_identity_sealed=false
```

Also require separate fields for financial CPU V2 provenance and evaluated
GPU V3 input; assert the source never assigns one receipt to the other and
never names `CanonicalTrendbarResearchDiscoveryResultV3` or `DiscoveryResult`
as the result payload. Require the startup Settings and runtime-install-receipt
identities, typed Gen0 scope,
raw/clamped legacy generation evidence labeled `unused_full_search`, frozen V1
limits, and an explicit `cost_band_status=unused_generation_zero` without
claiming a robustness measurement.

Require separately labeled `contract_domain_identity_sha256` and
`contract_file_sha256`; a RED must prove substituting one for the other fails.

- [ ] **Step 3: Run RED and save `chunk-2/red.log`**

Run:

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-2
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_generation_zero_result_v1 \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-2/red.log
test ${PIPESTATUS[0]} -ne 0
```

Expected: FAIL only on the missing result type/sealer/publisher.

- [ ] **Step 4: Implement the result and validator**

Use a private production sealer that receives the loaded contract evidence,
full native V3 receipt, the serializable population-sizing evidence/identity,
and successful `ResidentGenerationZeroMilestoneV1`. Snapshot the V5 contract,
native receipt, sizing receipt, and evaluation evidence through existing
getters before consuming V5; no new pre-Gen0 getter is needed. Serialize
`SearchResult::genes` and `[f64; 11]` metric rows in a dedicated wire payload
because `SearchResult` itself is deliberately not serializable. Hash typed,
domain-separated identity material that excludes only its own final hash.

Validation must check independent receipt validity, source-projection equality,
milestone/receipt identity equality, finite/shape-consistent candidates and
metrics, strict native engine, zero parent/adaptive H2D, bounded metrics
readback, confirmed completion, and every ResearchOnly flag.
For resolved population `P`, require `genes.len()==P`, `metrics.len()==P`,
exactly 11 finite `f64` values in every wire row, native readback rows `P`, and
checked readback bytes `P * 104`. Bind 104 to the authoritative
`NeoPopulationMetricRow`/`PopulationMetricsOnlyPlanV1` byte plan; do not derive
it as `11 * size_of::<f64>()`.

Freeze `MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1 = 512 MiB`. After prepared
facts resolve feature count `F`, explicitly map `max_indicators=0` to `F`
before constructing the sizing request, then seal term cap `T`. Compute with checked schema
arithmetic the maximum JSON sizes of wire keys, u64/usize/f64, bounded strategy
IDs, vector terms, receipts/source segments, `P` genes, and `P` metric rows.
Derive `P_cap(T)`; do not hard-cap configured P at 16,384. Require configured P
at most that cap before V5 admission. Inject `P_cap(T)` into the source-
compatible V2 sizing sealer so its effective growth cap is
`min(RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2, P_cap(T))`; never shrink
configured P. The existing hashed `hard_growth_cap` receipt field binds the
effective cap. The capped V5 workspace-plan closure verifies resolved
`P<=P_cap(T)` before workspace binding/native materialization. The final deterministic
compact-JSON counting writer enforces 512 MiB and never creates a second
unbounded full-artifact `Vec`.

- [ ] **Step 5: Implement create-new content-addressed publication**

Publish only to
`research/native-discovery/v1/cngr1-<identity>.json` beneath the checked root.
Traverse/create the output directory without following links. Stream to one
unique same-directory temporary file, flush and fsync it, atomically install
with no-replace semantics, fsync the parent directory, and remove the temporary
name. If another writer wins, bounded-reopen and accept only byte-identical
content. Do not write directly to the final path, overwrite, alias, or write
`current/latest`.

Generalize the proven temp + `sync_all` + hard-link/no-replace pattern from
`historical_search_receipt_prep` into the new root-I/O module; add directory
fsync and bounded identical-winner handling. Do not call `write_json_atomic`,
which replaces an existing target.

- [ ] **Step 6: Run GREEN and tamper tests**

Reject changes to contract ref/hash, either receipt, source projection,
`P_cap(T)`, effective hard-growth cap, population identity, metrics
identity/order/value, H2D counters, completion,
replay flag, artifact class, authorization flag, evidence hash, filename, and
existing nonidentical output bytes. Add crash-before-install, concurrent
identical writers, concurrent nonidentical collision, symlinked output
component, over-cap P/K, arithmetic overflow, counting-writer overflow, and
partial-final-object REDs.
Pre-create each of `research/`, `research/native-discovery/`, and its `v1/`
leaf as a Linux symlink escape in separate tests; every case must fail inside
the sealed root-I/O boundary.

Run:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_generation_zero_result_v1 \
  --test canonical_native_root_io_v1_contract \
  --no-fail-fast \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-2/green.log
test ${PIPESTATUS[0]} -eq 0
```

- [ ] **Step 7: Write `chunk-2/post.sha256` and obtain review approval**

## Chunk 3: One production staged-V5 executor

### Task 4: Connect exact pinned Data to V5 and the result sealer

**Files:**

- Create: `crates/neoethos-search/src/canonical_native_discovery_run_v1.rs`
- Create: `crates/neoethos-search/tests/canonical_native_discovery_entry_v1_contract.rs`
- Create: `crates/neoethos-search/tests/canonical_native_discovery_entry_v1_device.rs`
- Modify: `crates/neoethos-search/src/resident_population_auto_sizing_receipt_v2.rs`
- Modify: `crates/neoethos-search/src/prepared_discovery_run_input_v3.rs`
- Modify: `crates/neoethos-search/tests/prepared_discovery_run_input_v3_contract.rs`
- Modify: `crates/neoethos-search/src/lib.rs`

**Estimated implementation delta:** at most 950 changed lines. Keep the common
from-reference boundary/cancellation seam and CUDA executor as separately
reviewed patches.

- [ ] **Step 1: Record pre-edit hashes and write a production-path RED**

The contract test must require one public adapter boundary:

```rust
pub fn run_canonical_native_discovery_generation_zero_from_ref_v1<F>(
    startup_settings: &neoethos_core::Settings,
    runtime_install_receipt: &CanonicalNativeRuntimeInstallReceiptV1,
    contract_ref: CanonicalResearchContractArtifactRefV1,
    overrides: CanonicalNativeGenerationZeroOverridesV1,
    cancellation: &CanonicalNativeCancellationTokenV1,
    progress: F,
) -> Result<PublishedCanonicalNativeGenerationZeroResearchV1>
where
    F: FnMut(DiscoveryProgress);
```

Assert its source names the exact pin, Data preflight, prepared V5,
Data+population materializer, native V3 receipt constructor, Gen0 runner, and
result publisher. Assert it contains no CPU feature builder, `Auto`,
`run_batch`, legacy holdout/funnel, or fallback.

The non-Linux implementation returns `UnsupportedPlatform` first. On Linux,
the default-feature implementation returns `NativeCudaRequired` before calling
the request resolver/loader/install-receipt validator. The Linux CUDA
implementation checks cancellation,
resolves the one checked request internally, and then consumes only that
request. App/CLI adapters do not call loader -> resolver -> executor as three
separate authority boundaries.

Add a Linux default-feature runtime RED using an unreadable/nonexistent
referenced artifact: the returned error must be `NativeCudaRequired`, proving
no path open or resolver error won the ordering. Add a non-Linux source/
compile contract requiring `UnsupportedPlatform` before the feature gate or
any root/path symbol.

Add a source/order RED requiring
`prepare_gpu_only_feature_materialization_v3` to complete before the call to a
new source-compatible capped V5 wrapper. The wrapper receives that exact
prepared value by move, a checked `max_resolved_population`, and the native
factory; its internal preflight closure may only return/move the prepared value
and cannot call a Data preparer. It calls a new V2 capped sealer. That sealer
rejects configured P above the external cap, resolves its growth cap as
`min(RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2, external_cap)`, and stores
that effective value in the existing identity-hashed `hard_growth_cap` receipt
field. Add a RED where memory/time allow 16,384 but `P_cap=4,096`: auto must
resolve to 4,096, not fail or grow past it. Add receipt-cap mutation,
configured `P=P_cap+1`, and `P_cap=0` REDs; configured overflow must fail before
entering the dispatcher/context. The V5 closure defensively requires
`resolved_population<=max_resolved_population` before workspace bind/native
factory.

Freeze the additive V2 sealer as:

```rust
pub(crate) fn seal_resident_population_auto_for_canonical_trendbar_research_with_hard_cap_v2(
    prepared: &PreparedGpuOnlyFeatureMaterializationV3,
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
    config: &DiscoveryConfig,
    financial_contract: &CanonicalTrendbarResearchExecutionContractV3,
    external_hard_population_cap: usize,
) -> Result<(
    ResidentPopulationAutoSizingReceiptV2,
    SealedDataPopulationGpuWorkspacePlanV1,
), ResidentPopulationAutoSizingErrorV2>;
```

The existing uncapped V2 entry remains source-compatible and supplies its
current 16,384 growth cap. Add a `hard_growth_cap()` getter and bind it through
the receipt's existing computed identity/validation. The private
`ResidentPopulationAutoSizingRequestV2` gains one checked hard-growth-cap fact:
the existing constructor sets 16,384, while only the new Search-owned capped
constructor may set the smaller external value. No app/frontend constructs it.

Freeze the additive same-crate seam as:

```rust
pub(crate) fn prepare_prepared_canonical_trendbar_research_run_input_capped_v5<
    NativeFactory,
>(
    config: &DiscoveryConfig,
    financial_contract: &CanonicalTrendbarResearchExecutionContractV3,
    prepared: PreparedGpuOnlyFeatureMaterializationV3,
    max_resolved_population: usize,
    native_factory: NativeFactory,
) -> Result<PreparedCanonicalDiscoveryRunInputV5>
where
    NativeFactory: FnOnce(
        PreparedGpuOnlyFeatureMaterializationV3,
        AdmittedNativeCudaDataPopulationRunV1,
    ) -> Result<(
        CanonicalGpuResidentSearchInputReceiptV3,
        SealedGpuResidentFeatureStoreV3,
    )>;
```

It shares the existing V5 implementation internally; the existing public V5
signature/behavior remains source compatible.

- [ ] **Step 2: Run RED and save the compiler/source-contract log**

Run:

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-3
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_discovery_entry_v1_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-3/red.log
test ${PIPESTATUS[0]} -ne 0
```

Expected: FAIL only on the absent common boundary/executor.

- [ ] **Step 3: Implement the minimal executor**

Move the exact series into `pin_exact_canonical_series_v1`, preflight with the
contract base timeframe, fixed Standard profile, and exact parent-row budget,
then call `prepare_gpu_only_feature_materialization_v3` before entering V5.
Resolve `max_indicators=0` to exact prepared feature count `F`, seal `T` and
`P_cap(T)`, and pass the already-prepared continuation plus cap through the
source-compatible capped V5 wrapper. Its closure only moves this value; it
does not prepare Data after CUDA admission/context acquisition. The V2 capped
sealer solves auto with `min(existing_growth_cap,P_cap)` and its hashed
`hard_growth_cap` must equal that effective bound. Materialize on
the admitted Data+population run only after configured P and the V2 receipt's
resolved P/K pass the conservative output-size bound. Mint and validate V3
receipt, rebuild/compare the CPU projection
using existing public binding/segment getters plus the checked Data projector,
snapshot the full native receipt/financial contract/sizing evidence before
consuming V5, run Generation 0, seal the dual-receipt result, and publish it.
No new prepared/milestone getter is expected; stop on a compiler RED if current
source contradicts that audited seam.

Implement one cloneable Search cancellation token with typed probe points:
before contract load, exact pin, preflight, materialization, Gen0 launch, and
publication. Revalidate startup/runtime authority and mutable migration before
preflight. Once Gen0 launches, cancellation records intent but cannot return
until the V5 completion lease is confirmed.

The zero-indicator sentinel resolution creates one checked native-Gen0 config
copy with `max_indicators=F`; it preserves the raw `0` and resolved `F` in the
request/result evidence. No other Settings/DiscoveryConfig field is mutated.

- [ ] **Step 4: Run warning-denied non-device GREEN**

Run:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_discovery_entry_v1_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-3/contract-green.log
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search \
  --test canonical_native_discovery_entry_v1_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-3/default-gate.log
```

- [ ] **Step 5: Run the real RTX production-executor test**

The device fixture may publish immutable test generations and a saved contract
artifact, but after that setup it must call only the public from-reference
production boundary. Add an advance-after-contract case that returns the typed
current-generation conflict without entering native preflight.

Run:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  NEOETHOS_RUN_CANONICAL_NATIVE_DISCOVERY_V1_DEVICE_TEST=1 \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_discovery_entry_v1_device \
  -- --nocapture --test-threads=1 \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-3/rtx-green.log
```

Expected evidence: `CudaNativeF64`; configured and resolved population; exact
Stage1 bounds; parent/adaptive H2D `0`; `genes=P`, `metric_rows=P`, 11 finite
values per row, and `readback_rows=P`, `readback_bytes=P*104`; completion
`true`; replay `false`; distinct valid CPU
V2 and GPU V3 receipts; content-addressed result reopens and validates.

- [ ] **Step 6: Run regression graphs and write `chunk-3/post.sha256`**

Run:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda --lib \
  resident_population_auto_sizing_receipt_v2_contract
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test prepared_discovery_run_input_v3_contract
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test resident_generation_zero_v5_device -- --nocapture --test-threads=1
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-search --features gpu-b-adapter --all-targets
sha256sum \
  crates/neoethos-search/src/canonical_native_discovery_run_v1.rs \
  crates/neoethos-search/src/resident_population_auto_sizing_receipt_v2.rs \
  crates/neoethos-search/src/prepared_discovery_run_input_v3.rs \
  crates/neoethos-search/src/lib.rs \
  crates/neoethos-search/tests/canonical_native_discovery_entry_v1_contract.rs \
  crates/neoethos-search/tests/canonical_native_discovery_entry_v1_device.rs \
  crates/neoethos-search/tests/prepared_discovery_run_input_v3_contract.rs \
  > target/audit-logs/canonical-native-discovery-entry/chunk-3/post.sha256
```

Save each complete command log even where the compact listing above omits its
`tee` filename. Obtain independent code and RTX-evidence review before app
integration.

## Chunk 4A: Runtime-install carrier and atomic in-process lease

### Task 5A: Retrofit every existing in-process Search/Training start

**Files:**

- Modify: `crates/neoethos-app/src/app_services/jobs.rs`
- Modify: `crates/neoethos-app/src/app_services/discovery.rs`
- Modify: `crates/neoethos-app/src/app_services/discovery_tests.rs`
- Modify: `crates/neoethos-app/src/app_services/training.rs`
- Modify: `crates/neoethos-app/src/app_services/validation.rs`
- Modify: `crates/neoethos-app/src/server/state.rs`
- Modify: `crates/neoethos-app/src/server/engines_control.rs`
- Modify: `crates/neoethos-app/src/server/federation.rs`
- Modify: `crates/neoethos-app/src/lib.rs`
- Modify: `crates/neoethos-app/src/main.rs`
- Modify: `desktop/src-tauri/src/lib.rs`
- Create: `crates/neoethos-app/tests/search_runtime_lease_contract.rs`
- Modify: focused inline/app/server tests for every changed call site

**Estimated implementation delta:** at most 850 changed lines. The lease/
startup-carrier retrofit is a reviewed patch and must not include the new native
job lane.

- [ ] **Step 1: Record Chunk-4A prehashes and RED-lock every bypass**

Current source has independent Discovery/Training slots, a non-atomic slot
precheck, and an unguarded federation setter. Add compile/source REDs proving
every production call to `start_discovery_job` and `start_training_job` in app
main, `server::engines_control`, `app_services::validation`, and the legacy
Discovery-to-Training auto-chain must supply a move-only
`InProcessSearchRuntimeLeaseV1`. No public/default start overload may acquire a
different manager or omit the token.

Add a crate-private-production, process-wide
`InProcessSearchRuntimeLeaseManagerV1` in `app_services::jobs`. Production
startup obtains its one `Arc` from one `OnceLock`; explicit isolated managers
are `cfg(test)` only. `AppApiState` may carry that `Arc` for routing/status, but
it is not an authorization source and cannot mint tokens. The manager mutex
serializes owners `LegacyDiscovery`, `Training`, and
`CanonicalNativeResearch`. A lease is move-only and releases exactly once on
drop.

`migration_enable` must call a manager method that holds this same mutex while
it checks the active owner and invokes `neoethos_search::set_migration_enabled`
instead of setting migration directly. Native acquisition checks
`neoethos_search::migration_enabled()` while holding the mutex. Source REDs
permit no other production app call to `set_migration_enabled(true)`. This
closes app-process races only; it is not an OS-wide/GPU-wide lock, and an
unrelated CLI process still fails closed through CUDA admission/allocation/OOM.

Add barrier REDs for legacy-vs-native, training-vs-native,
native-vs-native, and migration-enable-vs-native. Exactly one side may win;
native cannot acquire after migration wins, migration cannot enable after any
lease wins, and all losers perform no Search/Data work.

- [ ] **Step 2: Preserve the legacy Discovery-to-Training handoff without a gap**

Extend `DiscoveryJobHandle` with a one-shot, move-only lease-completion
receiver. On clean legacy Discovery success, the worker sends the original
lease to the drainer; the drainer relabels that same token to `Training` and
passes it to a lease-requiring training start without ever making the manager
slot vacant. If the receiver/drainer is gone, the failed send drops the token.
Failed, degraded, cancelled, and panic paths drop it without chaining. RED-lock
that a competing native start cannot enter between Discovery success and the
Training worker, and that no unconsumed continuation leaks ownership.

- [ ] **Step 3: Retain the same installed Settings authority everywhere**

`AppApiState` stores `Arc<Settings>`,
`Arc<CanonicalNativeRuntimeInstallReceiptV1>`, and the production manager Arc.
Headless main passes its already-loaded Settings/install receipt;
`PreparedDesktopStartup` carries the same objects through `backend::start()`
into state. Test-only constructors install an explicit fixture triple.
Production handlers never reload `config_path` and never infer installation
from default-valued getters.

Change the app's existing `install_runtime_overrides_from_settings` aggregate
to return the typed receipt after Search/Data installation and snapshot checks;
update every app/Tauri caller to retain it. CLI startup calls the same
Search-owned installer/sealer immediately after its existing installation
sequence and retains the receipt through dispatch. Do not add a second config
load.

- [ ] **Step 4: Run warning-denied GREEN and race regressions**

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-4a
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-app --features gpu-nvidia --lib search_runtime_lease \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-4a/green.log
test ${PIPESTATUS[0]} -eq 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-app --features gpu-nvidia \
  --test search_runtime_lease_contract \
  --test server_contract_tests --no-fail-fast \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-4a/races.log
test ${PIPESTATUS[0]} -eq 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-app --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-app --features gpu-nvidia --all-targets
```

- [ ] **Step 5: Write `chunk-4a/post.sha256` and obtain review approval**

## Chunk 4B: Separate non-promotable application job lane

### Task 5B: Add `CanonicalNativeResearch` lifecycle and cancellation

**Files:**

- Create: `crates/neoethos-app/src/app_services/canonical_native_discovery.rs`
- Create: `crates/neoethos-app/src/app_services/canonical_native_discovery_tests.rs`
- Modify: `crates/neoethos-app/src/app_services/mod.rs`
- Modify: `crates/neoethos-app/src/app_services/jobs.rs`
- Modify: `crates/neoethos-app/src/server/state.rs`
- Modify: `crates/neoethos-app/src/server/system_status.rs`
- Modify: `crates/neoethos-app/src/main.rs`
- Modify: focused app/server lifecycle tests

**Estimated implementation delta:** at most 650 changed lines. This chunk uses
the reviewed manager/receipt carrier from Chunk 4A and does not add public HTTP
or CLI arguments.

- [ ] **Step 1: Write lifecycle REDs before the service implementation**

Require `JobKind::CanonicalNativeResearch`, a distinct `ServiceEvent` and state
slot, stage-labeled failure reports, exact result path/receipt/counter
highlights, and no model-target writer, promotion state, or Training auto-chain.
Require the start function to take the startup Settings/install receipt and a
move-only canonical-native lease from the shared manager. There is no overload
that accepts raw config/root or resolves the contract before a cancellation
handle exists.

- [ ] **Step 2: Implement `start_canonical_native_discovery_job_v1`**

Create the Search cancellation token first, acquire the manager lease and
install the job slot, then spawn the worker. Inside the worker call only
`run_canonical_native_discovery_generation_zero_from_ref_v1`; never call the
loader/resolver separately. Check cancellation before contract load, exact pin,
Data preflight, materialization, Gen0 launch, and publication. Once Gen0 has
launched, retain both CUDA and manager ownership until consumer completion;
then report cancelled and skip publication if cancellation was requested.
Every success/error/panic/cancel terminal path releases exactly once.

Add lifecycle REDs for stop-before-load, stop-before-materialize,
stop-during-native (status remains active until completion), panic/error
cleanup, and successful completion without Training auto-chain. The separate
native success event must not be representable as legacy Discovery success.

- [ ] **Step 3: Run focused GREEN and legacy snapshot regressions**

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-4b
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-app --features gpu-nvidia --lib canonical_native_discovery \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-4b/green.log
test ${PIPESTATUS[0]} -eq 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-app --features gpu-nvidia \
  --test native_discovery_workspace_bridge_source_contract \
  --test server_contract_tests --no-fail-fast \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-4b/legacy-regression.log
test ${PIPESTATUS[0]} -eq 0
```

- [ ] **Step 4: Write `chunk-4b/post.sha256` and obtain review approval**

## Chunk 4C: Explicit API and headless inputs

### Task 6: Add explicit API and headless inputs

**Files:**

- Modify: `crates/neoethos-app/src/server/engines_control.rs`
- Modify: `crates/neoethos-app/src/server/mod.rs`
- Modify: `crates/neoethos-app/src/main.rs`
- Create: `crates/neoethos-app/tests/canonical_native_discovery_entry_contract.rs`
- Modify: `crates/neoethos-app/tests/server_contract_tests.rs`

**Estimated implementation delta:** at most 750 changed lines. Keep route DTOs
and headless argument wiring as separate reviewed patches.

- [ ] **Step 1: Write route/argument REDs**

Require strict body fields for schema/version/relative path/SHA and only the
three allowed overrides. Require a separate
`POST /engines/discovery/canonical-native/start` and a matching
`POST /engines/discovery/canonical-native/stop`. Require explicit headless
`--canonical-native-discovery`, `--canonical-research-contract`, and
`--canonical-research-contract-sha256`; reject partial sets and conflicts with
legacy auto-discovery/training/validation modes.

Run the RED immediately after writing it:

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-4c
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-app --features gpu-nvidia \
  --test canonical_native_discovery_entry_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-4c/red.log
test ${PIPESTATUS[0]} -ne 0
```

- [ ] **Step 2: Prove legacy sources remain distinct**

The source contract must still find:

- `/engines/discovery/start` -> `start_discovery_job`;
- `--auto-discovery` -> legacy `DiscoveryRequest`;
- no native request inside the legacy handler; and
- no `DiscoveryResult`/model-target/training handoff inside the native handler.

- [ ] **Step 3: Implement the adapters and named non-CUDA refusal**

Both API and headless paths receive the exact startup Settings snapshot and
private runtime-install receipt produced after immutable Search/Data runtime
installation, create the same reference and override structs, and call the app
service. They never reload a mutable config file or infer installation from
getter defaults. A build without `gpu-nvidia` returns a typed/actionable error
before receipt validation, request resolution, or opening the contract file.

Stop looks up only the dedicated native job token. It reports
`cancellation_requested` while CUDA completion is pending and must not call a
legacy Discovery stop handler, drop ownership, or mark completion early.

- [ ] **Step 4: Run warning-denied app tests/checks**

Run focused tests and:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-app --features gpu-nvidia \
  --test canonical_native_discovery_entry_contract \
  --test server_contract_tests \
  --no-fail-fast \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-4c/green.log
test ${PIPESTATUS[0]} -eq 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-app --features gpu-nvidia --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-app --all-targets
```

Expected: both graphs warning clean; non-GPU graph has an honest refusal, not
a hidden CPU path.

- [ ] **Step 5: Write `chunk-4c/post.sha256` and obtain review approval**

## Chunk 5A: Search-owned contract carrier exporter

### Task 7A: Stream and publish one supported carrier schema

**Files:**

- Create: `crates/neoethos-search/src/canonical_research_contract_export_v1.rs`
- Create: `crates/neoethos-search/tests/canonical_research_contract_export_v1.rs`
- Modify: `crates/neoethos-search/src/canonical_native_root_io_v1.rs`
- Modify: `crates/neoethos-search/src/lib.rs`
- Modify: `crates/neoethos-cli/src/canonical_full_run.rs`
- Modify: `crates/neoethos-cli/tests/canonical_trendbar_full_run_source_contract.rs`

**Estimated implementation delta:** at most 850 changed lines.

- [ ] **Step 1: RED-lock the source schema and one-handle streaming boundary**

Freeze
`MAX_CANONICAL_RESEARCH_CONTRACT_CARRIER_BYTES_V1 = 512 MiB` and exactly one
V1 carrier: schema `neoethos.canonical-trendbar-full-run.v1`, version `1`, the
current private `CanonicalResearchDiscoveryArtifactV1` in
`canonical_full_run.rs`. Move/share only its schema constant; do not expose the
private full artifact type.

Freeze the supported top-level names to the current envelope exactly:
`schema`, `version`, `artifact_class`, `promotion_eligibility`,
`authorization_issued`, `plan_sha256`, `matrix_sha256`,
`research_contract_sha256`, `discovery_evidence_sha256`, `resolved_settings`,
`cost_assumption_exact_utf8`, `settings_source_exact_utf8`,
`broker_symbol_contract_exact_utf8`, `cost_assumptions`, `research_contract`,
`discovery_result`, `training_oos_from_ms`, `planned_models`,
`completed_models`, `failed_models`, `training_label_round_trip_cost_pips`, and
`model_artifacts`. Require them once each; schema evolution needs a new
explicit carrier version.

Require a typed streaming map visitor over the single handle-rooted opened
file. It recognizes the exact current top-level field set, rejects unknown/
duplicate/missing required fields, extracts only `research_contract` and
`research_contract_sha256`, consumes every other field as `IgnoredAny`, calls
`Deserializer::end()`, then drains the same bounded reader through EOF while
hashing the exact source bytes. It must not call
`serde_json::from_value`, deserialize `serde_json::Value`, read a second path
handle, or allocate a 512 MiB source `Vec`. Add `512 MiB + 1`, wrong source
SHA, mid-read mutation, malformed/unknown carrier, wrong embedded domain
identity, and large ignored-discovery-field REDs.

The public Search seam is:

```rust
pub fn export_canonical_research_contract_from_carrier_v1(
    startup_settings: &neoethos_core::Settings,
    source_ref: CanonicalResearchContractCarrierRefV1,
) -> Result<PublishedCanonicalResearchContractV1>;
```

It calls both contract validators and publishes deterministic standalone bytes
with the sealed root-I/O no-replace protocol. Only the typed published path,
source-file SHA, contract-domain identity, standalone-file SHA, and byte count
leave Search; no sealed-root or raw publisher primitive is public.

- [ ] **Step 2: Run RED**

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-5a
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search \
  --test canonical_research_contract_export_v1 \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-5a/red.log
test ${PIPESTATUS[0]} -ne 0
```

- [ ] **Step 3: Implement, run GREEN, and verify the full-run producer**

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search \
  --test canonical_research_contract_export_v1 \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-5a/green.log
test ${PIPESTATUS[0]} -eq 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-cli \
  --test canonical_trendbar_full_run_source_contract
```

- [ ] **Step 4: Write `chunk-5a/post.sha256` and obtain review approval**

## Chunk 5B: Thin CLI native/export adapters

### Task 7B: Add the explicit CLI subcommands

**Files:**

- Create: `crates/neoethos-cli/src/canonical_native_discovery.rs`
- Create: `crates/neoethos-cli/src/canonical_research_contract_export.rs`
- Modify: `crates/neoethos-cli/src/main.rs`
- Create: `crates/neoethos-cli/tests/canonical_native_discovery_entry_contract.rs`
- Create: `crates/neoethos-cli/tests/canonical_research_contract_export.rs`

**Estimated implementation delta:** at most 650 changed lines. Chunk 5A's
Search exporter must already be approved; this chunk owns only flag parsing,
startup receipt propagation, typed Search calls, and output formatting.

- [ ] **Step 1: Record prehashes and write argument/source REDs**

Require exactly one contract/SHA flag, canonical lowercase SHA, only three
optional overrides, no `--root`, `--config`, or `--device`, and no invocation
of `batch-discover`/`DiscoveryOrchestrator`/CPU feature preparation. Assert the
handler receives `&startup_settings` from `main` after the existing startup
load/install boundary.

Add thin-exporter REDs for one root-relative source artifact plus exact source-
file SHA, one call to Search's
`export_canonical_research_contract_from_carrier_v1`, and distinct printed
contract-domain identity / standalone-file SHA. Assert CLI source contains no
Serde carrier parsing, root-handle traversal, raw publisher, or financial-
contract validation.

Freeze the exporter CLI spelling as
`canonical-research-contract-export --source <root-relative-carrier> --source-sha256 <sha256>`.

- [ ] **Step 2: Run RED in GPU and default feature graphs**

Run:

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-5b
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-cli --features gpu-nvidia \
  --test canonical_native_discovery_entry_contract \
  --test canonical_research_contract_export \
  --no-fail-fast \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-5b/gpu-red.log
test ${PIPESTATUS[0]} -ne 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-cli \
  --test canonical_native_discovery_entry_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-5b/default-red.log
test ${PIPESTATUS[0]} -ne 0
```

- [ ] **Step 3: Implement the thin adapter**

Use the already-loaded startup Settings, derive its configured data root, build
the shared reference/overrides, call only the public from-reference executor,
and print result path/identity, both receipt identities, resolved population,
H2D/readback counters, completion, and replay status. Default builds return the
named NativeCuda-required error before resolution/artifact read.

Call the Search-owned exporter with the already-loaded startup Settings; do not
expose or call root-I/O/publication helpers from CLI. CLI startup retains the
private runtime-install receipt for the native command; the exporter needs only
the same startup Settings/root because it performs no runtime execution. The
native command accepts the resulting root-relative standalone artifact and
exact file SHA; it does not parse a full Discovery artifact during launch.

- [ ] **Step 4: Run focused GREEN and warning-denied CLI checks**

Run:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-cli --features gpu-nvidia \
  --test canonical_native_discovery_entry_contract \
  --test canonical_research_contract_export \
  --no-fail-fast \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-5b/gpu-green.log
test ${PIPESTATUS[0]} -eq 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-cli --features gpu-nvidia --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-cli --all-targets
```

- [ ] **Step 5: Write `chunk-5b/post.sha256` and obtain review approval**

## Chunk 5C: Explicit TUI route

### Task 8: Add an explicit TUI route without changing legacy default

**Files:**

- Modify: `crates/neoethos-cli/src/tui/form.rs`
- Modify: `crates/neoethos-cli/src/tui/pages/discover.rs`
- Modify: `crates/neoethos-cli/src/tui/jobs.rs`
- Modify: `crates/neoethos-cli/src/tui/app.rs` for mandatory focus/viewport,
  scroll, exact child-group ownership, and cooperative stop handling.
- Modify: `crates/neoethos-cli/src/canonical_native_discovery.rs`
- Modify: `crates/neoethos-cli/Cargo.toml`
- Modify if dependency resolution changes exact bytes: `Cargo.lock`
- Create: `crates/neoethos-cli/tests/canonical_native_discovery_tui_contract.rs`

**Estimated implementation delta:** at most 650 changed lines.

- [ ] **Step 1: Write TUI REDs**

Require default route `legacy_batch`, native route `native_gen0`, mandatory
contract path/SHA for native, dynamic launch label, exact new CLI argv, and
unchanged legacy argv snapshots. Require native stop to be cooperative: it
must not call the existing hard-kill path while a CUDA completion lease may be
live. Add a short-terminal RED proving focused path, SHA, launch, status, and
stop controls remain visible/reachable instead of being truncated.

Run:

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-5c
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-cli --features gpu-nvidia \
  --test canonical_native_discovery_tui_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-5c/red.log
test ${PIPESTATUS[0]} -ne 0
```

- [ ] **Step 2: Implement route-aware validation and spawn**

The native label must read `Native CUDA Gen0 — ResearchOnly`. It must use the
same current executable and subprocess manager. It must never append native
flags to `batch-discover`.

Spawn a native child in a dedicated process group and retain its exact child
handle/group identity. Before artifact resolution, the native CLI installs a
`SIGINT` handler on Linux (and the future Windows implementation uses
`CTRL_BREAK`) that only flips the Search cancellation token. Native stop first
checks the retained child is still live, sends one cooperative `SIGINT` to the
owned group, records `cancellation_requested`, and waits/reaps the child. A
child that exited first is reaped without signaling; repeated stop is
idempotent; signal-delivery failure is surfaced and never falls back to
`SIGKILL`, `kill_pid`, or `taskkill /F`. V1's non-Linux native entry remains
`UnsupportedPlatform`; CTRL_BREAK support is documentary follow-up, not a V1
claim. Add cleanup/race REDs and prove stop-during-CUDA waits for completion.
Use direct pinned signal/libc dependencies in `neoethos-cli` rather than naming
transitive crates; keep the async-signal handler limited to an atomic flag.

- [ ] **Step 3: Run TUI tests and both CLI feature checks**

Run:

```bash
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-cli --features gpu-nvidia \
  --test canonical_native_discovery_tui_contract \
  --test canonical_native_discovery_entry_contract \
  --no-fail-fast \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-5c/green.log
test ${PIPESTATUS[0]} -eq 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-cli --features gpu-nvidia --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-cli --all-targets
```

- [ ] **Step 4: Write `chunk-5c/post.sha256` and obtain review approval**

## Chunk 6: Desktop UI adapter

### Task 9: Add strict API types and a separate visible action

**Files:**

- Modify: `desktop/src/apiContracts.ts`
- Modify: `desktop/src/api.ts`
- Modify: `desktop/src/screens/Discovery.tsx`
- Modify: `desktop/test/apiContracts.test.ts`
- Create: `desktop/test/canonicalNativeDiscoveryState.test.ts`
- Modify: `desktop/src-tauri/src/lib.rs`

**Estimated implementation delta:** at most 750 changed lines. The startup
Settings/runtime-install-receipt carrier was introduced in the app-state
chunk; this chunk only wires and verifies the visible Desktop adapter/stop
action. V1 enables the native action on Linux only and shows the typed
unsupported-platform state elsewhere.

- [ ] **Step 1: Write TypeScript REDs**

Require the exact artifact-ref schema/version/path/SHA, reject missing/unknown
fields in the constructor, and ensure only population/population-auto/max-
indicators are forwarded. Require the new start/stop endpoints and separate
status fields.

Run:

```bash
mkdir -p /workspace/forex-ai/target/audit-logs/canonical-native-discovery-entry/chunk-6
cd /workspace/forex-ai/desktop
node --test test/apiContracts.test.ts test/canonicalNativeDiscoveryState.test.ts \
  2>&1 | tee ../target/audit-logs/canonical-native-discovery-entry/chunk-6/red.log
test ${PIPESTATUS[0]} -ne 0
```

- [ ] **Step 2: Implement the separate ResearchOnly panel/action**

Keep the current Discovery button and queue unchanged. Show required contract
relative path and SHA inputs, explicit Native CUDA/Gen0/ResearchOnly warnings,
resolved population and transfer counters, result path, completion status, and
`Replay identity: not sealed`. While a native stop is pending, show that CUDA
completion is still awaited. Never show portfolio/training/promotion success.

- [ ] **Step 3: Run complete Desktop verification**

Run:

```bash
cd /workspace/forex-ai/desktop
node --test test/apiContracts.test.ts test/canonicalNativeDiscoveryState.test.ts
npm run lint
npm run build
cd /workspace/forex-ai
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-desktop --features gpu-nvidia --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-desktop --all-targets
```

Expected: tests, lint, TypeScript, Vite build, and both Rust/Tauri feature
graphs GREEN and warning clean.

- [ ] **Step 4: Write `chunk-6/post.sha256` and obtain review approval**

## Chunk 7A: Cross-frontend routing and fail-closed evidence

### Task 10: Prove shared routing and fail-closed behavior

**Files:**

- Create: `crates/neoethos-search/tests/canonical_native_discovery_frontend_contract.rs`
- Modify focused app/CLI/Desktop tests only if review finds an uncovered edge.

**Estimated implementation delta:** at most 400 changed lines.

- [ ] **Step 1: Add a workspace source contract**

Assert CLI, TUI, app headless, and app API all name
`CanonicalResearchContractArtifactRefV1` and the same Search executor. Assert
legacy `batch-discover`, `/engines/discovery/start`, and `--auto-discovery`
still name their old backends and do not import the native executor.
Assert every in-process app start reaches the atomic
`InProcessSearchRuntimeLeaseV1`, every native adapter passes the startup
runtime-install receipt, and no frontend imports Search root-I/O/publisher
internals. Assert V1 non-Linux adapters reach `UnsupportedPlatform` before any
artifact loader symbol.

- [ ] **Step 2: Run the standalone feature matrix**

Run:

```bash
mkdir -p target/audit-logs/canonical-native-discovery-entry/chunk-7a
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 test \
  -p neoethos-search --features gpu-cuda \
  --test canonical_native_discovery_frontend_contract \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/chunk-7a/frontend-contract.log
test ${PIPESTATUS[0]} -eq 0
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-search --features gpu-cuda --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-search --features gpu-b-adapter --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-app --features gpu-nvidia --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-app --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-cli --features gpu-nvidia --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-cli --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-desktop --features gpu-nvidia --all-targets
env CARGO_INCREMENTAL=0 RUSTFLAGS=-Dwarnings \
  /root/.cargo/bin/cargo +nightly-2026-04-07 check \
  -p neoethos-desktop --all-targets
```

Do not rely on workspace feature unification to cover these.

- [ ] **Step 3: Write `chunk-7a/post.sha256` and obtain review approval**

## Chunk 7B: Real RTX and application-path acceptance

### Task 11: Run real RTX and application-path acceptance

**Estimated implementation delta:** at most 250 changed test/support lines;
the rest of this chunk is execution evidence.

- [ ] **Step 1: Rerun the Search production executor on RTX**

Run the exact Chunk-3 device command and retain the complete log.

- [ ] **Step 2: Export, then run the built CLI against an exact standalone contract**

Use the standalone exporter first against an operator-approved, known-schema
canonical research/full-run carrier under the configured root. Supply its
root-relative path and exact source-file SHA-256. Capture and verify all three
printed facts: extracted contract domain identity, standalone output-file
SHA-256, and the automatically selected content-addressed path
`research/contracts/v3/crcv3-<standalone_file_sha256>.json`. Use that exact
standalone path/SHA for the native run:

```bash
export CONFIG_FILE=/absolute/path/to/operator-approved-config.yaml
export SOURCE_REL=research/source-carrier.json
export SOURCE_SHA=<exact-lowercase-source-file-sha256>
/workspace/forex-ai/target/debug/neoethos-cli \
  canonical-research-contract-export \
  --source "$SOURCE_REL" \
  --source-sha256 "$SOURCE_SHA" \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/final/export.log
export CONTRACT_REL=<printed-root-relative-standalone-contract-path>
export CONTRACT_SHA=<printed-exact-standalone-file-sha256>
/workspace/forex-ai/target/debug/neoethos-cli \
  canonical-native-discover \
  --contract "$CONTRACT_REL" \
  --contract-sha256 "$CONTRACT_SHA" \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/final/cli.log
```

Start the process with the operator-approved canonical Settings file through
the CLI's existing startup configuration mechanism. Confirm the root printed
by the command equals `startup_settings.system.data_dir`; no second root or
config is accepted by the subcommand.

Expected: one content-addressed ResearchOnly result; no CPU fallback; zero
parent/adaptive H2D; `genes=P`; `metric_rows=P`; 11 finite values per metric
row; `readback_bytes=P*104`; completion true; replay false.

- [ ] **Step 3: Run the built headless app through its new explicit option**

Use the same root/path/SHA/settings and capture the full app log. Confirm the
new `CanonicalNativeResearch` status lane preserves the same stable contract-
file/domain identities, source projection, receipt semantics, and route/
counter invariants. Its `cngr1` result identity may differ and must validate
independently because replay identity is unsealed. Confirm it does not emit
Training-start, model-target, promotion, or legacy Discovery completion events.

```bash
/workspace/forex-ai/target/debug/neoethos-app \
  --headless \
  --config "$CONFIG_FILE" \
  --canonical-native-discovery \
  --canonical-research-contract "$CONTRACT_REL" \
  --canonical-research-contract-sha256 "$CONTRACT_SHA" \
  2>&1 | tee target/audit-logs/canonical-native-discovery-entry/final/headless.log
```

- [ ] **Step 4: Exercise the API/Desktop adapter against the same run contract**

POST the strict request, observe the dedicated status fields to completion,
and compare stable contract/source/receipt/route invariants with Search/CLI
evidence. Require a distinct valid content-addressed result per invocation
when genes/metrics/hardware sizing differ; never require cross-run result-
identity equality while `replay_identity_sealed=false`. The Desktop button is
an adapter proof only when the server log confirms the same Search executor.

Build and run the actual Tauri Rust shell with `gpu-nvidia`, start the request
from the visible panel, and exercise its stop action once. Node/Vite alone is
not Desktop acceptance. Confirm the shell-carried `Arc<Settings>` and
`Arc<CanonicalNativeRuntimeInstallReceiptV1>` identities match the installed
runtime authority and the published result.

With the headless server running on its configured port, exercise the API with
the frozen V1 body shape:

```bash
curl --fail-with-body --silent --show-error \
  -H 'content-type: application/json' \
  -X POST http://127.0.0.1:7423/engines/discovery/canonical-native/start \
  --data-binary "{\"schema\":\"neoethos.canonical-native-discovery-start.v1\",\"version\":1,\"contract\":{\"schema\":\"neoethos.canonical-research-contract-artifact-ref.v1\",\"version\":1,\"relative_path\":\"$CONTRACT_REL\",\"expected_sha256\":\"$CONTRACT_SHA\"},\"overrides\":{}}"
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:7423/system/status
curl --fail-with-body --silent --show-error \
  -X POST http://127.0.0.1:7423/engines/discovery/canonical-native/stop
```

On a GPU-equipped Desktop test host with the same canonical Settings/root,
launch the actual shell (not only Vite):

```bash
cd /workspace/forex-ai/desktop
CONFIG_FILE="$CONFIG_FILE" npx tauri dev --features gpu-nvidia \
  2>&1 | tee ../target/audit-logs/canonical-native-discovery-entry/final/tauri-gpu.log
```

- [ ] **Step 5: Test every failure before allocation**

Repeat with wrong SHA, escaped path, symlink, altered contract, mismatched
settings/runtime authority, missing/forged runtime-install receipt, advanced
current generation, migration toggled, racing legacy/training/native starts,
unsupported V5 policy, over-512-MiB result envelope, symlinked output parent,
non-Linux platform, and Linux non-CUDA build. Confirm no Data materialization
and no CPU work begin.

- [ ] **Step 6: Record final manifests and reviewer verdict**

Create:

```text
target/audit-logs/canonical-native-discovery-entry/final/source.sha256
target/audit-logs/canonical-native-discovery-entry/final/logs.sha256
target/audit-logs/canonical-native-discovery-entry/final/review.md
```

Then hash the exact source/result/log evidence:

```bash
cd /workspace/forex-ai
find crates desktop docs/superpowers -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > target/audit-logs/canonical-native-discovery-entry/final/source.sha256
find target/audit-logs/canonical-native-discovery-entry -type f \
  ! -path '*/final/logs.sha256' -print0 \
  | sort -z | xargs -0 sha256sum \
  > target/audit-logs/canonical-native-discovery-entry/final/logs.sha256
```

The final report must distinguish:

- built;
- warning-clean;
- source-contract tested;
- RTX device validated;
- CLI/TUI/headless/Desktop connected;
- persisted ResearchOnly artifact validated; and
- still not replay-sealed/full Discovery/OOS/promotion/training.

Do not claim full Search completion until a later version seals replay identity
and implements the remaining funnel/validation stages.
