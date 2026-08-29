# Canonical Native Discovery Entry Design

**Date:** 2026-08-27
**Decision:** Option 1 — explicit saved canonical research contract reference
**Status:** approved for implementation planning
**Current execution boundary:** native CUDA, staged V5, Generation 0 only

## Problem

The resident CUDA Data-plus-population path now executes a real Generation 0
on an RTX card, but no production application entrypoint can supply its exact
financial contract and reach it. The existing application, headless,
`batch-discover`, and TUI paths still enter legacy Discovery. On a physical GPU
the app path stops before native Data materialization; the CLI batch path
eagerly materializes host inputs and runs the legacy orchestrator.

The new entry must connect the already-proven staged V5 backend without
pretending that the bounded Generation-0 milestone is a complete Discovery
result. It must also preserve the independent CPU V2 financial-provenance
receipt and GPU V3 evaluated-input receipt instead of aliasing either identity
to the other.

## Approved scope

Build one new, versioned, explicit Native CUDA research entry shared by:

- a new CLI subcommand;
- an explicit TUI route;
- an explicit headless-app option; and
- a separate Desktop UI/API action.

All four frontends construct the same Search-owned request and call the same
Search-owned executor. The request names a previously saved
`CanonicalTrendbarResearchExecutionContractV3` artifact by a path contained by
the configured canonical root and by the expected SHA-256 of its exact bytes.

The first version ends after Generation 0 and publishes a new
`ResearchOnly`, `NotPromotionEligible`, `GenerationZeroOnly` result. It does
not claim replay identity, a funnel, portfolio, holdout, validation, promotion,
training handoff, or a full Discovery completion.

## Explicit non-goals

- Do not repoint or reinterpret `batch-discover`.
- Do not repoint `/engines/discovery/start`, `--auto-discovery`, or the current
  legacy TUI launch action.
- Do not return or persist `CanonicalTrendbarResearchDiscoveryResultV3`; that
  type asserts that the embedded CPU V2 input was evaluated.
- Do not construct a CPU feature frame, host parent, or host adaptive-base
  series on the Native CUDA route.
- Do not add `auto`, CPU fallback, GPU-absence fallback, inline financial
  values, environment-variable authority, mutable `current/latest` pointers,
  or a caller-selected raw CUDA ordinal.
- Do not load a second Settings file or accept a second canonical root after
  process-wide Search/Data runtime authority has been installed. The route
  uses the exact startup `Settings` value and its configured data root.
- Do not trigger Training, promotion, live trading, OOS, autoresearch, or the
  legacy Discovery funnel from a Generation-0 success.
- Do not call this run reproducible while
  `replay_identity_sealed == false`.
- V1 production execution is Linux-only. Non-Linux frontends return typed
  `UnsupportedPlatform` before artifact read; Windows handle-relative reparse
  support and junction evidence are a later separately reviewed version.

## Chosen architecture

### 1. Search owns the shared request and execution boundary

Add focused Search modules for:

- the saved-contract reference and bounded loader;
- the Search-owned bounded carrier exporter;
- a typed proof that Gen0-consumed Search/Data runtime installers ran against
  the same startup Settings;
- the checked runtime request;
- the production staged-V5 executor; and
- the dual-receipt Generation-0 result and content-addressed publisher.

`neoethos-app`, `neoethos-cli`, the TUI, and Desktop UI are adapters only. They
may load operator settings and collect explicitly allowed Generation-0 knobs,
but they do not implement Data preparation, CUDA admission, receipt binding,
financial-contract interpretation, or result identity hashing.

They call one Search-owned from-reference boundary. Its non-CUDA form refuses
before request resolution or artifact read; its CUDA form checks cancellation,
loads/resolves the private checked request, and continues into the staged
executor. No frontend can reorder loader, resolver, admission, or publisher.

### 2. The saved contract is a reference, not ambient state

The versioned reference is:

```rust
#[serde(deny_unknown_fields)]
pub struct CanonicalResearchContractArtifactRefV1 {
    schema: CanonicalResearchContractArtifactRefSchemaV1,
    version: u16,
    relative_path: String,
    expected_sha256: String,
}
```

The concrete wire values are:

- schema: `neoethos.canonical-research-contract-artifact-ref.v1`;
- version: `1`;
- a non-empty UTF-8 relative path containing only normal components; and
- one canonical lowercase 64-hex SHA-256.

The canonical root is `settings.system.data_dir` from the exact in-memory
`Settings` value that was loaded before, and used for, process-wide
Search/Data runtime installation. It is not a request field, is never taken
from the contract file, and is not silently replaced by the process working
directory. CLI and TUI therefore do not accept another `--root` or `--config`
for this command. The app's existing startup `--config` remains authoritative
because it is parsed before runtime installation.

### 3. Bounded, root-contained, single-read loading

`load_canonical_research_contract_artifact_v1` performs all checks before CUDA
probing, allocation, or feature materialization:

1. Require one existing physical canonical-root directory and retain its
   stable handle/identity for the whole traversal.
2. Reject an absolute path, prefix, empty component, `.`, `..`, control
   character, or symlink component.
3. Traverse/open relative to the retained root with no-follow semantics. After
   open, resolve and validate the final handle, stable file identity, regular
   file type, and containment beneath the same physical root. Path-only
   `canonicalize` followed by ordinary `open` is forbidden because it leaves a
   symlink swap race.
4. Bound the opened regular file to
   `MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1 = 8 MiB`.
5. Read that opened handle once, including an extra-byte overflow check.
6. Hash those exact bytes and compare them to `expected_sha256`.
7. Deserialize those same bytes as exactly
   `CanonicalTrendbarResearchExecutionContractV3`; unknown fields and unknown
   schema versions fail.
8. Run both `contract.validate()` and
   `contract.validate_against_receipt(contract.input_receipt())`, then derive
   its identity and node-name-independent pinned-source projection.

The returned loaded value owns the parsed contract, normalized relative path,
exact artifact SHA-256, byte length, contract identity, and source projection.
It is move-only and has no public raw constructor. Later stages consume it;
they never reopen the path or consult an ambient installed contract.

The loader calls both `contract.validate()` and
`contract.validate_against_receipt(contract.input_receipt())`. The latter is
required to cross-check the financial symbol against the receipt anchor. The
contract's domain identity (`contract.identity_sha256()`, historically named
`research_contract_sha256`) and the exact standalone JSON file SHA-256 are
different facts and are stored and labeled separately.

V1 also supplies a Search-owned standalone export/extraction operation. Its
only supported source carrier is
`neoethos.canonical-trendbar-full-run.v1`, version `1`, the current private
`CanonicalResearchDiscoveryArtifactV1` envelope in
`neoethos-cli/src/canonical_full_run.rs`. The source cap is the explicit
`MAX_CANONICAL_RESEARCH_CONTRACT_CARRIER_BYTES_V1 = 512 MiB`, separate from
the 8 MiB standalone-contract cap. Search opens the source once through the
same handle-rooted boundary, streams exact-byte SHA-256 and a typed top-level
Serde map visitor over that one bounded handle, rejects duplicate/unknown
top-level fields, and extracts only `research_contract` plus
`research_contract_sha256`; large unrelated fields are consumed as
`IgnoredAny`, never as `serde_json::Value` or a full in-memory carrier. The
visitor consumes through EOF before the source-file hash is accepted.

Search then runs both contract validations, requires the carrier's
`research_contract_sha256` to equal the contract domain identity, serializes
the contract alone as deterministic compact JSON, and publishes it with the
same safe content-addressed create-new protocol at
`research/contracts/v3/crcv3-<standalone_file_sha256>.json`. It prints the
contract domain identity, the standalone exact-file SHA-256, and the
root-relative output path. Extraction validates the source file hash and the
supported carrier schema/version but does not reinterpret the carrier's CPU
Discovery result as native evidence. The CLI only parses flags and prints the
typed Search result; it never receives raw root-I/O primitives or interprets
the carrier. Native Discovery never scrapes a large result artifact implicitly.

### 4. The request derives exact source generations from the contract

The CPU V2 receipt embedded in the contract is financial provenance only; it
is never relabeled as a native feature receipt. Its validated, node-name-
independent source projection may, however, identify the exact immutable
dataset generations that must be pinned.

The request resolver converts each projection binding into a
`SelectedDatasetGenerationV1` using the exact dataset identity, generation id,
and manifest-binding SHA-256, then seals one
`CanonicalDatasetSeriesReceiptV1`. It requires exactly one anchor, one source
per timeframe, exact anchor coverage, and the same source/account/symbol/
timestamp convention.

The present Data opener is narrower than an arbitrary historical replay:
`pin_exact_canonical_series_v1` ultimately calls
`open_exact_dataset_generation`, which accepts the saved selection only when
it still equals the current manifest under the publication lock. V1 is
therefore explicitly **current-at-acquisition only**. If any pointer advanced
after the contract was created or exported, resolution returns the typed exact-
generation conflict before Data allocation. It never substitutes the new
`current`, and it never claims it reopened an arbitrary old generation. A
later historical V2 needs a retained-generation index/lease policy because
current GC protects only current and previous generations.

The shared checked request is conceptually:

```rust
pub struct CanonicalNativeDiscoveryRequestV1 {
    loaded_contract: LoadedCanonicalResearchContractV1,
    startup_settings_sha256: String,
    runtime_install_receipt: CanonicalNativeRuntimeInstallReceiptV1,
    canonical_root: SealedCanonicalRootV1,
    exact_series: CanonicalDatasetSeriesReceiptV1,
    config: DiscoveryConfig,
    scope: CanonicalNativeGenerationZeroScopeV1,
    limits: CanonicalNativeGenerationZeroLimitsV1,
    feature_profile: FeatureProfile, // V1 is fixed to Standard
}
```

Construction is private to
`resolve_canonical_native_discovery_request_v1(startup_settings,
runtime_install_receipt, contract_ref, overrides)`. Startup first calls one
typed installer/sealer that invokes every Gen0-consumed Search/Data installer,
compares the installed snapshots with the same Settings, and returns a private-
construction `CanonicalNativeRuntimeInstallReceiptV1`. This avoids treating
getter defaults as evidence that `OnceLock` installation occurred. The
resolver revalidates the receipt and installed snapshots before reading the
contract and again before preflight. CLI and App/Tauri state retain the same
in-memory Settings and receipt rather than rereading a mutable config file.
The small override object permits only checked
Generation-0 knobs that are actually consumed: configured population,
population-auto, and maximum indicators. It cannot override symbol, account
currency, source identity, financial values, feature profile, device route,
output class, or execution scope.

The explicit entry type itself supplies Generation-0 intent. The resolver
builds `DiscoveryConfig` through
`try_from_settings_for_canonical_trendbar_research`. That existing adapter
clamps `prop_search_generations` with `max(1)`, so its resolved
`config.generations` field is not the Generation-0 scope authority. The new
typed scope is. The executor always calls the bounded Generation-0 runner and
never the configured multi-generation loop. The raw and clamped legacy
generation settings are preserved and hashed as unused full-search evidence;
ordinary multi-generation production Settings are neither rejected nor
mutated merely to launch this explicit Gen0 route.

It fails before Data when the current V5 limitations are not explicitly
satisfied. In V1:

- raw/clamped legacy generation counts are recorded as unused full-search
  evidence and never interpreted as permission for another generation;
- the session-spread curve must be absent because V5 cannot bind it;
- the cost band is preserved in the Settings/config evidence but is explicitly
  unused and unclaimed by `GenerationZeroOnly`; ordinary configured/default
  cost bands are neither cleared nor rejected;
- adaptive thresholds and ATR-scaled gene bounds must be disabled;
- minimum-history and discovery-ledger paths must be disabled;
- row trimming and feature prefiltering must be no-ops; and
- process migration must already be disabled and remain disabled; the route
  does not turn off an active federation process; and
- the payoff floor must be reachable.

No unsupported setting is silently changed to make the run pass.

The request also carries a typed
`CanonicalNativeGenerationZeroRuntimeAuthorityV1`. Search derives it from the
same startup Settings and compares it with every already-installed runtime
override consumed by Gen0, including Data normalization/feature-cube policy,
genetic, evaluation, backtest, SMC, stop-target, adaptive, gene-bound, and
seen-memory settings. It checks mutable federation migration is false. The
authority identity is revalidated before preflight; the existing V2/V5
receipt and drift checks remain in force through completion. A file reload or
mesh toggle cannot silently change the run.

### 5. Bounded request and persisted-result envelope

V1 fixes `MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1 = 512 MiB`. It does not
hard-code resolved population to the current auto-growth value of 16,384.
After prepared feature facts reveal exact feature count `F`, zero
`max_indicators` is explicitly resolved to `F` before constructing the V2
sizing request and the effective term cap `T` is sealed; `usize::MAX` is never
carried into sizing or serialization. Checked
schema arithmetic includes the maximum JSON representation of every wire key,
u64/usize/f64, bounded strategy ID, gene vector term, receipt, source binding/
segment, `P` genes, and `P` metric rows. It derives the largest persistable
`P_cap(T)` under 512 MiB.

Raw override syntax/count bounds are checked before the contract is opened.
Prepared Data facts, resolved `T`, and `P_cap(T)` are produced before entering
the V5 admission/context dispatcher. Configured population must be at most
`P_cap(T)` or the request fails without entering the dispatcher. A source-
compatible V2 sizing sealer injects the typed external cap and resolves
`effective_growth_cap = min(existing_hard_growth_cap, P_cap(T))`; configured P
is never silently shrunk. The existing receipt's hashed `hard_growth_cap`
field binds that effective cap. The capped V5 wrapper also verifies the sealed
resolved population is at most `P_cap(T)` inside the workspace-plan closure,
before workspace binding or the native materializer. Thus a valid auto run can
grow to the persistable cap instead of failing merely because the old 16,384
ceiling was larger. A counting writer
enforces the 512 MiB final cap while emitting deterministic compact JSON, so
the publisher never constructs a second unbounded result buffer. Values
outside the frozen V1 envelope fail by name; they do not truncate genes/
metrics or silently alter the requested population.

### 6. One staged V5 production executor

After the common from-reference gate resolves the checked request, the executor
continues in this order:

1. Pin the exact series receipt under the canonical root.
2. Run `preflight_gpu_only_feature_workspace_v3` with the contract anchor
   timeframe, fixed `FeatureProfile::Standard`, and the exact parent-row count.
3. Call `prepare_gpu_only_feature_materialization_v3` and obtain one
   `PreparedGpuOnlyFeatureMaterializationV3` before entering the V5 dispatcher
   or acquiring its CUDA context. Resolve `F`, the zero-indicator sentinel,
   `T`, and `P_cap(T)` from this exact prepared value.
4. Call the additive source-compatible
   `prepare_prepared_canonical_trendbar_research_run_input_capped_v5` wrapper
   around the staged V5 implementation. Its
   preflight closure may only move the already-prepared value; it must not
   prepare or reopen Data after admission/context acquisition.
5. Let V5 resolve population from the same pre-materialization free-memory
   snapshot and bind the Data-plus-population workspace once.
6. Materialize only through
   `materialize_prepared_gpu_only_feature_store_for_data_population_v3`.
7. Mint `CanonicalGpuResidentSearchInputReceiptV3` from that sealed store and
   validate it against the store, pinned-source projection, plan-bound strict
   GPU execution authority, and contract-derived financial authority.
8. Run `run_prepared_canonical_trendbar_research_generation_zero_v5`.
9. Require consumer completion, zero parent H2D, zero adaptive-base H2D,
   metrics-only readback bounds, and native engine `CudaNativeF64`.
10. Seal and publish the new Generation-0 result.

There is no caller-visible or functional CPU factory in this public route. A
common feature-gated entry checks Native CUDA capability before request
resolution or contract read. A missing CUDA build/runtime, zero compatible
devices, admission failure, OOM, receipt mismatch, or kernel failure is a
named error.

### 7. Dual-receipt ResearchOnly result

Add `CanonicalNativeGenerationZeroResearchResultV1`, with deny-unknown-fields
serialization and an identity hash over all identity-affecting fields. It
contains:

- schema/version and explicit `GenerationZeroOnly` scope;
- `ResearchOnly`, `NotPromotionEligible`, and `authorization_issued = false`;
- the normalized contract artifact reference, exact artifact SHA-256 and byte
  length;
- the separately labeled contract domain identity and standalone exact-file
  SHA-256;
- startup Settings, runtime-install-receipt, and Gen0 runtime-authority
  identities, the typed Gen0 scope,
  raw/clamped legacy generation evidence labeled `unused_full_search`, the
  frozen 512 MiB envelope, and `cost_band_status=unused_generation_zero`;
- the validated `CanonicalTrendbarResearchExecutionContractV3` and its
  financial CPU V2 receipt identity, labeled **financial provenance only**;
- the full `CanonicalGpuResidentSearchInputReceiptV3`, labeled **evaluated
  native input**;
- population-sizing receipt identity, Stage1 range, resolved population,
  term cap, `P_cap(T)`, effective V2 hard-growth cap, metrics receipt
  identities, and optional adaptive-token identity;
- a serializable copy of Generation-0 genes and metric rows;
- explicit residency counters, including parent/adaptive upload bytes and
  metrics readback rows/bytes;
- `consumer_completion_confirmed = true`;
- `replay_identity_sealed = false`; and
- a domain-separated `evidence_identity_sha256`.

Validation requires both receipts to validate independently, the contract's
source projection to equal the native receipt's pinned-source projection, the
milestone receipt identities to match the embedded receipts, finite and
shape-consistent result rows, zero forbidden H2D counters, confirmed
completion, and all ResearchOnly/non-promotion flags. The validator must never
compare CPU feature-plan/content identity to GPU feature-plan/Merkle identity.
For resolved population `P`, it requires exactly `P` genes, exactly `P` metric
rows, exactly 11 finite `f64` values per row, native readback rows `P`, and
checked readback bytes `P * 104`, where 104 is the authoritative native
`NeoPopulationMetricRow` width rather than `11 * 8`.

The result is written under the canonical root at:

`research/native-discovery/v1/cngr1-<evidence_identity_sha256>.json`

Publication is create-new and content-addressed. An existing path is accepted
only when its bounded bytes are exactly identical; it is never overwritten and
there is no mutable `current/latest` pointer. The publisher traverses the
output directory beneath the same sealed root without following links, writes
a unique same-directory temporary file through the bounded counting writer,
flushes and `fsync`s it, atomically installs it with no-replace semantics,
`fsync`s the parent directory, then removes the temporary name. A racing
winner is bounded-reopened and accepted only when byte-identical. A crash must
never leave a partial final content-addressed object.

## Frontend adapters

### CLI

Add `canonical-research-contract-export` as the explicit producer for the
standalone artifact. It accepts one root-relative source artifact plus its
exact file SHA-256, calls only the Search-owned bounded exporter, and prints
its typed output path, contract domain identity, and exact output-file SHA-256.

Add a new `canonical-native-discover` subcommand with exactly one each of:

- `--contract <root-relative-json>`; and
- `--contract-sha256 <lowercase-64-hex>`.

Optional Generation-0 overrides are limited to population,
population-auto, and maximum indicators. The command receives the already
loaded `startup_settings` from CLI `main`; it neither reloads config nor
accepts another data root. A non-CUDA build fails by name before resolving the
request or reading the referenced artifact. The command prints the
content-addressed result path, result identity, both receipt identities,
resolved population, H2D counters, metric rows, completion state, and replay
state.

### TUI

Keep `batch-discover` as the default route. Add an explicit route selector
`legacy_batch | native_gen0` plus required contract-relative-path and SHA-256
fields for `native_gen0`. The launch label and command preview must say
`Native CUDA Gen0 — ResearchOnly`; selecting it spawns the new CLI subcommand.
It never changes `batch-discover` arguments or implies a full Discovery
portfolio. The Linux V1 native child is placed in its own process group.
Before any artifact read it installs a `SIGINT` handler that flips the same
Search cancellation token. Native stop signals that exact live group once,
retains the child handle, and waits for the child
to report completion/cancelled; it never uses `kill_pid`, `SIGKILL`, or
`taskkill /F`. Exit-before-stop, repeated-stop, signal-delivery failure, PID/
group cleanup, and stop-during-CUDA are typed states. TUI viewport/scroll and
focus navigation are mandatory so path, SHA, launch, and stop remain visible
and reachable in a short terminal. Windows `CTRL_BREAK` support belongs to the
later Windows-native version; V1 fails before spawning that route there.

### Headless app

Add explicit `--canonical-native-discovery` plus the required contract path
and SHA-256 arguments. It is separate from `--auto-discovery` and mutually
exclusive with legacy auto-discovery/training/validation modes. It constructs
the same request and starts the same dedicated app service. A one-shot failure
is logged and surfaced; no CPU retry occurs.

### Desktop UI/API

Add `POST /engines/discovery/canonical-native/start` with a strict body that
requires the artifact-ref schema/version/path/hash and exposes only the three
allowed Generation-0 overrides, plus
`POST /engines/discovery/canonical-native/stop` for its cancellation token.
Add a separate Discovery-screen panel/action labeled
`Native CUDA Gen0 — ResearchOnly`; do not change the current Discovery button.

The new service uses a separate `CanonicalNativeResearch` job/event/status
lane. There is no existing cross-lane exclusion primitive, so V1 adds one
process-wide, in-process `InProcessSearchRuntimeLeaseManagerV1` in the
app-services boundary. Its production constructor is crate-private and every
direct `start_discovery_job`, `start_training_job`, and canonical-native start
requires a move-only lease obtained from that one manager; `AppApiState` is a
status carrier, not an alternate authorization path. The headless and Desktop
startup paths retain the same manager instance, and compile/source contracts
cover the direct app-main, engine-control, validation, and auto-chain call
sites so none can construct or bypass a lease.

Lease acquisition and federation migration-enable use the same manager mutex.
Concurrent Legacy Discovery, Training, and canonical-native starts cannot both
pass; migration-enable rejects while any lease is held, and native start
rejects if migration is already enabled. On successful legacy Discovery, the
worker hands the original move-only lease through a one-shot continuation to
the drainer, which relabels it for Training without releasing the manager slot;
if the continuation is not consumed, dropping it releases the lease. Every
other terminal/error/panic/cancellation path releases exactly once. A
successful canonical-native Generation-0 milestone never emits the legacy
Discovery-success condition and therefore never starts Training.

This lease protects one process's runtime globals; it is not described as an
OS-wide or GPU-global lock. The TUI parent owns at most one exact child handle,
but an unrelated CLI process can still compete for the card. CUDA admission,
allocation, or OOM must then fail closed. Cross-process serialization would
require a separately designed root-scoped OS lock and is not claimed by V1.

The API/UI also expose a dedicated native stop action. Stop sets the Search
cancellation token and reports `cancellation_requested`; it does not detach or
free a running CUDA session. The worker remains in the native lane until the
completion lease is ready, then reports cancelled and skips publication.

## Failure and cancellation flow

Failures are stage-labeled and fail closed:

- `native_capability_gate`;
- `runtime_install_receipt`;
- `search_gpu_execution_lease`;
- `contract_reference_validation`;
- `contract_artifact_read`;
- `contract_artifact_hash`;
- `contract_schema_validation`;
- `exact_source_pin`;
- `native_preflight`;
- `native_admission`;
- `resident_data_materialization`;
- `native_receipt_binding`;
- `generation_zero_evaluation`;
- `consumer_completion`; and
- `result_publication`.

The app creates/spawns the cancellable job before request resolution and checks
cancellation before contract loading, before Data preflight, before
materialization, and before Generation-0 launch. Resolution is therefore job
work, not synchronous work performed before a cancellation handle exists.
Once a native launch has started, it waits for the completion lease; it does
not drop GPU ownership or report cancellation early. A cancellation observed
after completion may skip publication and report a clean cancelled state.

## Acceptance criteria

1. All four new frontend adapters submit the same versioned artifact-reference
   and override shapes to one Search-owned from-reference boundary; only
   Search constructs the private checked request and reaches the executor.
2. Legacy CLI/app/TUI/UI entrypoints remain source- and behavior-distinct.
3. Contract path traversal, Linux symlink escape or swap race, non-regular files,
   oversized bytes, hash mismatch, unknown fields/version, data-generation
   mismatch, settings mismatch, and non-CUDA builds fail before Data
   allocation.
   A saved selection that is no longer the current manifest returns the typed
   exact-generation conflict; V1 makes no arbitrary historical-replay claim.
   Linux no-follow/swap tests pass. A non-Linux build returns typed
   `UnsupportedPlatform` before reference resolution/read; no unverified
   Windows containment claim is made by V1.
4. A real RTX test reaches the production executor and reports
   `CudaNativeF64`, parent H2D `0`, adaptive H2D `0`, exact resolved population,
   `P` genes, `P` metric rows of 11 finite values, readback rows `P`, readback
   bytes `P * 104`, and confirmed consumer completion.
5. The persisted result contains distinct CPU V2 financial-provenance and GPU
   V3 evaluated-input receipts and never claims they are equal.
6. The persisted result always says `GenerationZeroOnly`, `ResearchOnly`,
   `NotPromotionEligible`, `authorization_issued=false`, and
   `replay_identity_sealed=false`.
7. No successful native result can auto-start Training, enter promotion, or be
   loaded as a legacy `DiscoveryResult`.
8. A real app/headless invocation against an operator-supplied saved contract
   is required before claiming application-level completion. A synthetic
   fixture proves mechanics only.
9. The standalone export prints different, correctly validated values for the
   contract domain identity and exact contract-file SHA-256.
10. Native stop/cancellation is proven before load and while waiting for native
    completion; no cancellation path drops a live completion lease.
11. Desktop acceptance includes the Rust Tauri shell linked with
    `gpu-nvidia`, not only Node/Vite tests.
12. Separate Search/CLI/headless/API/Desktop invocations compare stable
    contract-file/domain identities, source projections, typed receipts, and
    route/counter invariants. Each may publish a distinct valid `cngr1` result;
    equality is never required while replay identity remains unsealed.
13. A request cannot be resolved without the private-construction runtime
    install receipt from the same startup Settings, even when all configured
    values equal compiled defaults.

## Repository evidence policy

`/workspace/forex-ai` currently has no `.git` metadata. A branch commit cannot
be created or cited truthfully. Every implementation chunk therefore records:

- pre-edit SHA-256 for every touched file;
- RED and GREEN command logs;
- post-edit SHA-256 for every touched file; and
- an independent review verdict.

These manifests replace an impossible commit checkpoint; they do not imply
that the changes are merged or version-controlled elsewhere.
