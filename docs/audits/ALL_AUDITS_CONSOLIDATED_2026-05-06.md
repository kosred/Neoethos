# All Audits Consolidated Master Report

Created: 2026-05-06 Europe/Berlin
Repository: `kosred/forex-ai`
Source directory: `docs/audits`
Source audits: 33 markdown files

This file consolidates the 33 audit documents under `docs/audits` into one operational master report.

The original audit files remain the detailed evidence. This file is the unified working map: what the audits collectively say, what must be preserved, what must be fixed first, and how the refactor should be sequenced.

---

## Source audit manifest

1. `architecture_unification_duplicate_code_cleanup_audit_2026-05-03.md`
2. `artifact_intent_clarification_training_vs_search_resume_2026-05-03.md`
3. `core_config_domain_modularization_audit_2026-05-04.md`
4. `cpu_gpu_semantic_parity_requirement_2026-05-03.md`
5. `custom_cuda_kernel_preservation_audit_2026-05-03.md`
6. `dead_code_and_stale_artifacts_audit_2026-05-03.md`
7. `deep_duplicate_logic_and_unified_scheduler_audit_2026-05-03.md`
8. `deep_search_engine_state_audit_pass2_2026-05-03.md`
9. `evaluation_contract_deep_audit_pass3_2026-05-03.md`
10. `evolution_neat_crfmnes_gpu_first_audit_2026-05-03.md`
11. `feature_timestamp_mtf_causality_deep_audit_pass5_2026-05-03.md`
12. `forex_data_functional_audit_2026-05-04.md`
13. `forex_models_functional_audit_2026-05-04.md`
14. `forex_search_functional_audit_2026-05-04.md`
15. `generic_scheduler_small_files_refactor_note_2026-05-04.md`
16. `gpu_cuda_hpc_parity_deep_audit_pass4_2026-05-03.md`
17. `gpu_first_kernel_everywhere_report_2026-05-03.md`
18. `hardware_autodetect_config_ui_architecture_2026-05-03.md`
19. `model_runtime_backend_fragmentation_audit_2026-05-03.md`
20. `modularization_maintainability_refactor_principle_2026-05-04.md`
21. `python_pyo3_legacy_audit_2026-05-03.md`
22. `quality_challenge_validation_refactor_audit_2026-05-04.md`
23. `rust_env_flags_config_debt_audit_2026-05-03.md`
24. `search_backtest_forward_cpu_gpu_audit_2026-05-03_14-46.md`
25. `search_checkpoint_resume_requirement_2026-05-03.md`
26. `search_discovery_pipeline_audit_2026-05-03.md`
27. `search_gpu_discovery_scheduler_audit_2026-05-03.md`
28. `search_orchestration_refactor_audit_2026-05-03.md`
29. `search_portfolio_artifact_contract_audit_2026-05-03.md`
30. `search_to_live_bridge_audit_2026-05-03.md`
31. `training_model_artifact_contract_audit_2026-05-03.md`
32. `unified_module_logic_architecture_2026-05-04.md`
33. `universal_hardware_parity_requirement_2026-05-03.md`

---

## Executive conclusion

The 33 audits converge on one diagnosis:

`forex-ai` is not weak because it lacks capability. It is risky because capability is fragmented across too many local contracts, local runtime decisions, local environment flags, local hardware assumptions, and overloaded artifact meanings.

The target architecture is:

```text
one canonical runtime contract
+ one canonical hardware scheduler
+ one canonical artifact/provenance contract
+ one canonical data/timestamp/feature availability policy
+ one canonical search/backtest/forward/live semantic bridge
+ deterministic execution when requested
+ explicit CPU/GPU/HPC parity guarantees
+ preserved custom GPU kernels behind clean adapters
+ small modules with clear ownership
```

Do not delete advanced functionality. Preserve the useful parts and remove duplicated ownership.

---

## Highest-level architecture rule

Every subsystem must answer five questions in a typed and testable way:

1. What data, features, symbols, timeframes, and timestamps did it consume?
2. What config, search, risk, and runtime assumptions did it use?
3. Which backend and hardware path did it run on?
4. Is the output canonical, approximate, fallback, degraded, diagnostic-only, or live-ready?
5. Is the output allowed to influence live trading?

If the answer is hidden in env vars, local helper code, filenames, or tribal knowledge, it must be moved into a typed contract.

---

## Consolidated P0 requirements

### P0-1: Centralize hardware and execution scheduling

All CPU, GPU, and HPC decisions must flow through one scheduler-owned execution plan.

Current repeated decision points appear across search, model runtime capabilities, tree model config, Burn/WGPU selection, CUDA discovery, parallel trainer, registry, and environment flag parsing.

Required target types:

```text
HardwareProfile
WorkloadExecutionPlan
DeviceAssignment
CpuBudget
GpuBudget
PrecisionPolicy
DeterminismPolicy
BackendKind
RuntimeDegradedReason
```

Model and search modules should receive resolved assignments. They should not probe hardware or parse production behavior from environment variables locally.

### P0-2: Preserve all custom CUDA/GPU kernels

Custom GPU kernels are project assets, not cleanup noise.

Preserve and wrap:

- statistical linear softmax GPU kernels,
- CRFMNES candidate-loss GPU kernels,
- NEAT population metric GPU kernels,
- search/discovery GPU kernels,
- backtest/evaluation GPU kernels,
- future GPU-first execution routes.

CPU implementations should remain as reference/fallback paths where useful.

Runtime must record whether execution was native GPU, native CPU, CUDA kernel, WGPU backend, CPU reference, surrogate fallback, degraded runtime, approximate mode, or unavailable backend.

### P0-3: Enforce CPU/GPU semantic parity

CPU and GPU paths are interchangeable only when their semantics are proven equivalent.

Required parity fields:

```text
feature_schema_hash
timestamp_policy_hash
feature_availability_policy_hash
label_policy_hash
cost_model_hash
risk_model_hash
execution_semantics_hash
seed_or_rng_policy
```

If a GPU path is approximate or uses a different rule, the artifact must explicitly say so.

### P0-4: Create one provenance-complete artifact contract

Training, model, search, portfolio, checkpoint, runtime, and live artifacts need shared provenance.

Required fields:

```text
artifact_kind
artifact_schema_version
feature_schema_hash
dataset_fingerprint
symbol_universe_hash
timeframe_set_hash
timestamp_unit
timestamp_policy_hash
feature_availability_policy_hash
label_policy_hash
training_config_hash
search_config_hash
runtime_config_hash
risk_config_hash
seed
hardware_profile_id
device_assignment
backend_kind
canonical_or_approx_mode
runtime_degraded_reason
created_at
source_commit
```

Artifacts without required provenance should not be accepted for live execution.

### P0-5: Separate training artifacts from search/resume artifacts

Training artifacts, search checkpoints, and portfolio/live artifacts have different intent.

Training artifacts answer what model was trained, on which data/features/config, and how it should be loaded.

Search/resume artifacts answer where the optimizer can resume, which candidates exist, and what has already been evaluated.

Portfolio/live artifacts answer which strategies survived validation, what risks/correlations were accepted, and whether they are live-safe.

Do not allow these artifact types to masquerade as each other.

### P0-6: Make search checkpoint/resume crash-safe and strict

Search is expensive, so resume must be first-class.

Required behavior:

- atomic checkpoint writes,
- config hash validation,
- dataset fingerprint validation,
- search-space hash validation,
- symbol/timeframe validation,
- generation/iteration cursor,
- evaluated candidate ledger,
- RNG state or deterministic seed chain,
- partial result ledger,
- hardware/runtime plan recorded,
- explicit invalidation when semantics change.

### P0-7: Fix timestamp, MTF, and causality policy as shared contracts

Timestamp and multi-timeframe feature availability are critical trading risks.

Required target:

```text
one timestamp unit
one candle open/close policy
one MTF availability policy
one feature alignment policy
one embargo/walk-forward policy
one live-readiness policy
```

No model, search, backtest, or live path should infer these independently.

### P0-8: Remove production semantics from environment variables

Environment variables are acceptable for local diagnostics. They are dangerous when they silently change production semantics.

Move env-driven decisions into typed config:

```text
Settings
RuntimeConfig
HardwareConfig
TrainingConfig
SearchConfig
LiveExecutionConfig
```

Then resolve once into a validated runtime plan.

### P0-9: Make randomness deterministic when requested

Unseeded randomness appears across genetic search, mutation/crossover, RL, exit-agent exploration, sampling, and fallback algorithms.

Required target:

```text
DeterminismPolicy::Deterministic { seed }
DeterminismPolicy::BestEffort
DeterminismPolicy::NonDeterministicAllowed
```

For research, regression tests, and prop-firm validation, deterministic mode must be reproducible.

### P0-10: Clarify canonical evaluation versus model sanity evaluation

Model sanity scoring must not be confused with canonical trading evaluation.

Required separation:

```text
model_sanity_backtest
canonical_strategy_backtest
walk_forward_validation
forward_test_validation
live_execution_simulation
prop_firm_risk_validation
```

Only canonical strategy/backtest/live contracts should gate production decisions.

---

## Consolidated P1 refactor plan

### P1-1: Split large files into small ownership modules

Large-file risks appear in training orchestration, Burn/deep models, ensemble, RL, exit-agent, streaming/adaptive models, search orchestration, and base helpers.

Refactor principle:

```text
one file = one concept
one module = one ownership boundary
no god files
no duplicate private mini-frameworks
```

Suggested module families:

```text
runtime/
training/
search/
scheduler/
artifacts/
provenance/
evaluation/
live_bridge/
risk/
models/tree/
models/statistical/
models/evolution/
models/deep/
models/burn/
models/ensemble/
models/rl/
models/exit/
models/streaming/
models/anomaly/
```

### P1-2: Use backend adapters for backend-specific behavior

Backend complexity should be isolated behind explicit adapters.

Adapters should exist for CPU reference, CUDA kernels, WGPU/Burn, native tree libraries, local fallback/surrogate modes, and external runtimes.

### P1-3: Introduce explicit degraded-mode metadata

Fallbacks must be visible.

Examples:

```text
native_lightgbm_gpu
native_lightgbm_cpu
local_surrogate_tree_fallback
burn_wgpu
burn_cpu
cuda_kernel
cpu_reference
external_swarm_runtime
diagonal_profile_fallback
```

A degraded artifact must never look equivalent to a native artifact.

### P1-4: Unify search, model, portfolio, and live bridge contracts

The dangerous boundary is:

```text
search result -> selected strategy/model -> portfolio artifact -> live execution
```

Required contracts:

```text
LiveReadyStrategyArtifact
PortfolioSelectionArtifact
ModelRuntimeArtifact
ExecutionRiskContract
FeatureAvailabilityContract
```

Live execution must reject artifacts when hashes, timestamp policy, symbol universe, cost model, risk model, execution assumptions, or backend semantics do not match.

### P1-5: Clean stale and dead code after contracts exist

Do not delete blindly.

Safe cleanup order:

1. Define canonical contract.
2. Mark old paths as legacy/deprecated.
3. Add tests proving the new path covers old behavior.
4. Remove stale files and obsolete artifacts.
5. Add CI checks to prevent reintroduction.

---

## Consolidated P2 improvements

### P2-1: Hardware autodetect and UI architecture

Hardware detection should feed the same canonical scheduler used by training/search/live.

UI should expose detected CPU cores, RAM, GPUs, VRAM, CUDA/WGPU availability, supported precision modes, recommended execution plan, and warnings when workload exceeds hardware budget.

### P2-2: Python/PyO3 legacy boundary

Python and PyO3 remain useful, but must not become a semantic fork.

Policy:

- Rust owns canonical contracts.
- Python bindings expose those contracts.
- Python helpers do not redefine timestamp, feature, evaluation, search, artifact, or risk semantics.
- PyO3 wrappers should be thin and tested against Rust contract fixtures.

### P2-3: Documentation and operator safety

Each critical artifact type should have a short operator-facing document explaining what it is, how it is produced, required hashes/provenance fields, whether it is live-safe, how it is invalidated, and how to resume or rebuild it.

---

## Distilled findings by source audit

1. Architecture unification / duplicate code cleanup: consolidate by semantic ownership, not convenience.
2. Artifact intent clarification: separate training/model, search resume, portfolio, and live artifacts.
3. Core config/domain modularization: split config/domain logic from execution logic.
4. CPU/GPU semantic parity: prove equivalent semantics or explicitly mark approximate/degraded paths.
5. Custom CUDA kernel preservation: preserve kernels and wrap them behind adapters.
6. Dead code and stale artifacts: delete only after canonical replacement and tests exist.
7. Deep duplicate logic and unified scheduler: collapse duplicate device decisions into one scheduler.
8. Deep search engine state: search state must be explicit, resumable, and hash-bound.
9. Evaluation contract: separate model sanity, canonical backtest, walk-forward, forward test, and live simulation.
10. Evolution NEAT/CRFMNES GPU-first: keep GPU-first execution with CPU reference and deterministic evaluation.
11. Feature timestamp MTF causality: use one availability/causality policy; no implicit lookahead.
12. Forex data: canonicalize data loading, timestamp units, symbol/timeframe identity, and features.
13. Forex models: model coverage is strong, but runtime/device/provenance logic is fragmented.
14. Forex search: search/discovery/backtest/portfolio selection need strict contracts.
15. Generic scheduler small files: scheduler code should remain small and composable.
16. GPU CUDA HPC parity: preserve semantics, resource budgeting, and deterministic behavior.
17. GPU-first kernel everywhere: make GPU kernels first-class execution backends.
18. Hardware autodetect config UI: hardware detection should feed typed config and execution plans.
19. Model runtime backend fragmentation: use backend adapters and shared runtime metadata.
20. Modularization maintainability: split by responsibility, not arbitrary line count.
21. Python PyO3 legacy: bindings must expose Rust semantics, not redefine them.
22. Quality challenge validation: prop-firm validation needs strict risk and live-safety gates.
23. Rust env flags config debt: move env-driven production behavior into typed config.
24. Search backtest forward CPU/GPU: search/backtest/forward tests must share one semantic contract.
25. Search checkpoint resume: add atomic strict resumable checkpoints.
26. Search discovery pipeline: make stages and artifacts explicit.
27. Search GPU discovery scheduler: GPU discovery must be scheduler-managed and resource-aware.
28. Search orchestration: orchestration coordinates stages; it should not own all business logic.
29. Search portfolio artifact: selected portfolios need explicit risk/correlation/validation semantics.
30. Search to live bridge: live execution must reject mismatched or non-live-safe artifacts.
31. Training model artifact: model artifacts need stable schemas, metadata, provenance, and load-time validation.
32. Unified module logic architecture: one concept should have one owning module and many backend implementations.
33. Universal hardware parity: hardware-specific execution must be equivalent or explicitly degraded/approximate.

---

## Recommended implementation sequence

Completed items are removed from this active sequence after implementation. Phase 1 contract items were completed on 2026-05-06; see the execution log for the historical record.

The original active sequence is complete. A new follow-on audited sequence has started from the remaining consolidated requirements, beginning with shared timestamp/MTF causality contracts before deeper UI/PyO3/operator-safety work.

---

## Required test matrix

### Artifact contract tests

- artifact refuses missing provenance,
- artifact refuses feature schema mismatch,
- artifact refuses timestamp policy mismatch,
- artifact refuses dataset fingerprint mismatch,
- artifact refuses wrong artifact kind,
- degraded artifact cannot be loaded as canonical.

### Scheduler tests

- device assignment is respected,
- CPU budget is respected,
- GPU memory budget is respected,
- env override cannot silently change production plan,
- unavailable GPU path records explicit degraded reason.

### CPU/GPU parity tests

- statistical model fixture parity,
- evolution candidate loss parity,
- search discovery parity,
- backtest/evaluation parity,
- deterministic seed parity.

### Timestamp/feature tests

- no MTF lookahead,
- candle close/open policy enforced,
- missing timestamp rejected where required,
- feature alignment uses canonical policy,
- live input schema matches training/search schema.

### Search/resume tests

- checkpoint resumes after interruption,
- checkpoint rejects changed config,
- checkpoint rejects changed dataset,
- checkpoint rejects changed search space,
- evaluated candidates are not repeated unless explicitly allowed,
- RNG sequence is reproducible under deterministic mode.

### Live bridge tests

- live rejects model sanity artifacts,
- live rejects search checkpoints,
- live rejects stale portfolio artifacts,
- live rejects cost/risk mismatch,
- live accepts only live-ready artifacts.

---

## Non-negotiable invariants

1. No hidden production semantics in env vars.
2. No silent CPU/GPU semantic drift.
3. No custom GPU kernel deletion during refactor.
4. No artifact without provenance.
5. No search checkpoint treated as model artifact.
6. No model sanity score treated as canonical backtest.
7. No live execution without artifact validation.
8. No timestamp/MTF feature path without causality policy.
9. No unseeded randomness in deterministic mode.
10. No new god files while trying to remove old god files.

---

## Bottom line

The 33 audits collectively recommend a contract-first refactor.

Do not start by deleting code. Start by defining shared contracts, scheduler ownership, provenance, timestamp policy, deterministic execution, and the live-readiness gate. Then move existing capabilities behind those contracts.

The strongest parts of the project are the GPU work, the breadth of model/search capability, and the existing validation instincts. The weakest parts are fragmented ownership, implicit runtime behavior, overloaded artifacts, and duplicated device/backend logic.

The correct direction is not smaller by losing power. It is smaller by making every module own exactly one thing, every artifact say exactly what it means, and every execution path prove whether it is canonical, approximate, or degraded.

---

## Execution log


### 2026-05-10: Follow-on Phase 51 completed — typed determinism policy in DiscoveryRunProfile

Bridged Phase 26's `current_determinism_policy()` accessor and the Phase 50 promotion runbook by recording the resolved [`DeterminismPolicy`] directly in the persisted `DiscoveryRunProfile`:

- imported `DeterminismPolicy` into `forex-search::discovery` from `forex_core::contracts`;
- added `determinism_policy: DeterminismPolicy` to `DiscoveryRunProfile` so the persisted `*_profile.json` documents whether the genetic search ran under `Deterministic { seed }`, `BestEffort`, or `NonDeterministicAllowed`. The field uses Codex's serde rename (`mode: "deterministic" | "best_effort" | "non_deterministic_allowed"`) so consumers can render it directly;
- updated `build_discovery_profile` to populate the field via `crate::genetic::current_determinism_policy()` — the OnceLock-cached typed view that Phase 26 added on top of the legacy `seed: Option<u64>` field;
- added a discovery-test asserting the profile carries one of the three legal variants (the OnceLock may already be installed by another test in the same process, so the test pins the variant set rather than the specific value).

Phase 50's promotion-readiness runbook documents `PromotionRejectedDeterminism` as one of the gate's rejection reasons; this phase makes the underlying field visible in the profile JSON so operators can diagnose the rejection without re-reading source code or re-running discovery.

Next follow-on targets: populate `live_sim_runtime_model_hash` once a live-execution simulator is in place, explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` (P0-9), UI exposure of scheduler/hardware plans (P2-1).


### 2026-05-10: Follow-on Phase 50 completed — operator promotion readiness reference

Closed item 39 from Codex's Phase 30-40 doc handoff list ("Add operator docs for promotion rejection reasons and remediation") by writing [`docs/operator/promotion_readiness.md`](../operator/promotion_readiness.md). The document is the runbook for reading the `LivePromotionGate` verdict and acting on every rejection reason it can produce; it complements [`artifact_safety.md`](../operator/artifact_safety.md) which covers the per-artifact contracts.

The runbook covers:

- the gate-at-a-glance table mapping the four orthogonal checks (validation evidence, runtime safety, live execution contract, determinism requirement) to the report fields they populate;
- how to read the new `validation_evidence_*` fields surfaced in `*_profile.json` from Phase 49, including a worked example showing the always-missing `live_execution_simulation` hash;
- one section per rejection variant — `MissingValidationEvidence`, `LiveRejectedRuntimeMode`, `LiveRejectedMismatch`, `LiveRejectedStaleArtifact`, `LiveRejectedFailedEvidenceGate`, `LiveRejectedMissingEvidence`, `PromotionRejectedDeterminism` — with the source contract method, the plain-language meaning, the common causes (specific to discovery, walk-forward, forward-test, prop-firm, and live-sim), and the rebuild instructions;
- a recommended fix-order walkthrough: runtime safety → validation evidence → live-contract mismatches → determinism, so operators do not chase secondary errors before their root cause is resolved.

No code changes; the doc is self-contained and references the existing crates / helpers. The `live_execution_simulation` always-missing rejection is documented as structural (deferred until the simulator lands) rather than a bug, so operators are not confused by gates that cannot pass today.

Next follow-on targets: populate `live_sim_runtime_model_hash` once a live-execution simulator is in place, explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` (P0-9), UI exposure of scheduler/hardware plans (P2-1).


### 2026-05-10: Follow-on Phase 49 completed — promotion-evidence summary in DiscoveryRunProfile

Continued the Phase 48 producer-consumer wiring by surfacing the typed evidence-summary directly in the persisted `DiscoveryRunProfile` so operators see promotion-readiness signals in the profile JSON without instantiating a `LivePromotionGate`:

- made `DiscoveryPerKindEvidenceHashes` `Serialize` and added `all_present()` + `missing_kinds()` helpers — operators / UI layers can render "which artifact kinds are present, which are missing" without parsing `MissingValidationEvidence` strings;
- extended `DiscoveryRunProfile` with three new fields: `prop_firm_validation_artifacts_observed` (was missing despite Phase 29 producing the artifacts), `validation_evidence_hashes` (typed per-kind hashes), `validation_evidence_complete` (single boolean), and `validation_evidence_missing_kinds` (sorted list of missing-kind names);
- updated `build_discovery_profile` to compute the per-kind hashes via `discovery_per_kind_evidence_hashes` and propagate the typed summary; the live-execution simulation hash always lands as `None` until a simulator is wired, which surfaces in the profile as `validation_evidence_complete = false` and `validation_evidence_missing_kinds = ["live_execution_simulation"]` (plus any other missing kinds);
- added a discovery-test verifying that a partial result (canonical + forward-test + prop-firm but no walk-forward) produces a profile with the right `Some` / `None` per kind, `validation_evidence_complete = false`, and the missing-kind list containing both `walkforward` and `live_execution_simulation`.

The `*_profile.json` written next to every portfolio export now documents exactly which validation evidence hashes were produced and which are still missing for live promotion. UI / operator dashboards can read this single field to decide whether a portfolio is promotion-ready without re-deriving any hash logic.

Next follow-on targets: populate `live_sim_runtime_model_hash` once a live-execution simulator is in place, explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` (P0-9), UI exposure of scheduler/hardware plans (P2-1).


### 2026-05-10: Follow-on Phase 48 completed — DiscoveryResult → ValidationEvidenceManifest

Connected the producer-side artifact pipeline to Codex's promotion-contract surface (Phases 30-44) so the search bridge feeds the typed `ValidationEvidenceManifest` directly:

- added `discovery_validation_evidence_manifest(result)` in `forex-search::discovery` that hashes the four DiscoveryResult artifact vectors (canonical / walk-forward / forward-test / prop-firm) into a typed manifest. Live-execution simulation hash is intentionally left empty so `ValidationEvidenceManifest::validate` surfaces the typed `MissingValidationEvidence("live_execution_simulation_hash")` error rather than a silently-filled placeholder — the simulator is still deferred and the gate must reject until it lands;
- added `discovery_per_kind_evidence_hashes(result)` returning the typed `DiscoveryPerKindEvidenceHashes` struct (one `Option<String>` per kind) so operator/UI layers can render a diagnostic view ("forward-test present, live-sim missing") without forcing manifest validation;
- relaxed `forex_search::artifact_io::stable_json_hash` to accept `T: Serialize + ?Sized` so the helpers can hash `&[T]` slice arguments without intermediate copies;
- re-exported `DiscoveryPerKindEvidenceHashes`, `discovery_validation_evidence_manifest`, and `discovery_per_kind_evidence_hashes` from `forex-search::lib`;
- added discovery-tests covering: manifest rejects missing live-sim evidence (the always-failing gate), manifest rejects missing walk-forward evidence (an explicitly-empty kind), per-kind hashes return `Some` only for present kinds, and per-kind hashes return all-`None` for an empty result.

The integration handoff guardrail from Codex's Phase 40 doc — "wire producers into this contract, do not create another live-readiness schema in discovery or UI code" — is now satisfied on the discovery side. The forex-app discovery service can call `discovery_validation_evidence_manifest` after running the cycle to feed Codex's `LivePromotionGate` directly.

Next follow-on targets: persist a `PromotionReadinessReport` as part of `DiscoveryRunProfile` so operators see promotion verdicts without code, populate `live_sim_runtime_model_hash` once a live-execution simulator is in place, explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` (P0-9), UI exposure of scheduler/hardware plans (P2-1).


### 2026-05-10: Follow-on Phase 47 completed — operator-facing artifact safety reference

Closed P2-3 by writing the short, opinionated reference document the audit explicitly requires for every critical artifact type. The document lives at [`docs/operator/artifact_safety.md`](../operator/artifact_safety.md) and answers the audit's five operator questions for each of the five validation artifact kinds (`canonical_strategy_backtest`, `walkforward_validation`, `forward_test_validation`, `live_execution_simulation`, `prop_firm_risk_validation`):

- what it is and the one-line intent boundary that prevents kind-confusion;
- how it is produced, including the discovery / forex-app helper that builds it and the persistence directory layout;
- the required provenance fields (`artifact_kind`, schema version, scope hashes, temporal scope);
- whether it is live-safe — and which gate / aggregate flag the live bridge reads to decide acceptance vs rejection;
- how it is invalidated and how to rebuild it.

The document closes with two operator-facing sections that did not previously exist anywhere in the repo: a worked example showing how `LiveExecutionContract::validate_provenance` + `validate_evidence` chain together via the `live_validation_evidence_from_discovery` bridge, and a "trigger → affected artifacts" rebuild matrix that maps real operational events (dataset re-fetch, rule override, temporal-contract upgrade, gene mutation) to the artifacts that need rebuilding. The matrix surfaces the Phase 25 / 29 rule that `prop_firm_risk_validation` is bound to a specific `rules_hash` so different challenges produce coexisting files.

No code changes; the doc is self-contained and references the existing crates / helpers. Working tree clean apart from the new file and this audit-log entry.

Next follow-on targets: per-challenge prop-firm preset selector in the UI, populate `live_sim_runtime_model_hash` once a live-execution simulator is in place, explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` (P0-9), UI exposure of scheduler/hardware plans (P2-1).


### 2026-05-10: Follow-on Phase 46 completed — per-challenge PropFirmRiskRules on DiscoveryRequest

Removed the last hardcoded fallback in the prop-firm wiring so non-FTMO challenges drive the gate without code changes:

- added `prop_firm_rules: PropFirmRiskRules` to `forex_app::app_services::discovery::DiscoveryRequest`. The field is typed (no `Option`) so every caller is forced to make an explicit choice — defaults to `PropFirmRiskRules::default()` for backward compatibility, but the type makes "I'm using FTMO defaults" visible in the call site;
- updated the discovery service's prop-firm compute call to read `search_request.prop_firm_rules` instead of constructing a fresh `PropFirmRiskRules::default()` inline; the same OOS tail still drives forward-test and prop-firm validation, but the rule set is now caller-controlled;
- updated the three production `DiscoveryRequest` constructors (`forex_app::main::run_app` headless boot, `forex_app::main` UI request builder, and `forex_app::ui::discovery::render` UI form submission) to populate the new field with `PropFirmRiskRules::default()` until per-challenge UI controls are added;
- updated the discovery test fixture (`sample_request`) so existing test coverage stays compilable.

Per-challenge UI controls (e.g. an FTMO vs The Funded Trader vs MFF preset selector) belong in a UI follow-on phase that wires real challenge configurations onto the discovery form. The contract surface and the call-site plumbing are in place — the only remaining work is exposing the knobs to operators.

Next follow-on targets: per-challenge prop-firm preset selector in the UI, populate `live_sim_runtime_model_hash` once a live-execution simulator is in place, explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` (P0-9), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-10: Follow-on Phase 45 completed — prop-firm wired into forex-app discovery service

Closed the final P1-4 wiring slice so the forex-app UI discovery flow now produces prop-firm evidence end-to-end without any caller intervention:

- imported `PropFirmRiskRules`, `compute_discovery_prop_firm_artifacts`, and `save_prop_firm_validation_artifacts` into `forex-app::app_services::discovery`;
- ran `compute_discovery_prop_firm_artifacts` immediately after the existing forward-test computation, reusing the same OOS tail (`tail_features` + `tail_ohlcv` built from rows past `wfv_bound`) so both gates evaluate the same held-out window. The default `PropFirmRiskRules` (FTMO-style) drive the validation; per-challenge rule overrides belong on the discovery request and are tracked as a follow-on;
- prop-firm computation failures are logged at `warn` and do not block portfolio export, mirroring the forward-test fallback so existing `walkforward_passed` / `cpcv_passed` gates remain the production-blocking signals while prop-firm evidence becomes diagnostic-or-blocking depending on the live contract;
- persisted artifacts now write to a sibling `*.prop_firm_validations/` directory next to `*.forward_tests/` / `*.canonical_backtests/` / `*.walkforward_validations/`, so a downstream live bridge can discover all four validation kinds from one portfolio export root.

After Phase 30, the discovery flow produces every validation kind required for `LiveExecutionContract::validate_evidence` to accept a portfolio without external preprocessing: walkforward + CPCV from the discovery cycle itself, forward-test from the OOS tail (Phase 24), and prop-firm from the same OOS tail (this phase). The Phase 28 evidence bridge already aggregates all four into `LiveValidationEvidence`, so the live bridge's gate work is purely a contract-load operation now.

Next follow-on targets: per-challenge `PropFirmRiskRules` on `DiscoveryRequest` so the forex-app UI can drive non-FTMO challenges, populate `live_sim_runtime_model_hash` once a live-execution simulator is in place, explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` (P0-9), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-10: Follow-on Phase 29 completed — discovery prop-firm validation artifacts

Closed the third slice of P1-4 by giving the discovery pipeline a prop-firm validator that mirrors the Phase 24 forward-test wiring, so a portfolio's persisted prop-firm artifacts feed the live-bridge gate at acceptance time:

- added `prop_firm_validation_artifacts: Vec<PropFirmRiskValidationArtifactFile>` to `DiscoveryResult` (defaults empty) and updated every internal/test constructor;
- added `compute_discovery_prop_firm_artifacts(portfolio, effective_feature_names, tail_features, tail_ohlcv, config, rules)` that aligns the tail's columns to the post-prefilter set, replays each gene through `signals_for_gene_full` + `simulate_trades_core`, and aggregates the trades through `compute_prop_firm_risk_summary` to build one `PropFirmRiskValidationArtifactFile` per strategy. The signature matches `compute_discovery_forward_test_artifacts` so the forex-app discovery service can reuse the same tail it already builds for forward-test;
- added `save_prop_firm_validation_artifacts(dir, result)` for atomic per-strategy persistence using the same content-addressable filename scheme as the canonical / walk-forward / forward-test helpers;
- extended `live_validation_evidence_from_discovery` to populate `prop_firm_passed` from the persisted artifacts: `Some(true)` when every artifact reports `all_rules_passed`, `Some(false)` when at least one fails, `None` when no artifact was produced (the live bridge then treats `None` as missing evidence whenever the gate is required);
- re-exported the new compute / save helpers from `forex-search::lib`;
- added `discovery::tests` covering: empty-portfolio short-circuit, missing-feature rejection, one-artifact-per-strategy emission, atomic save round-trip, and the evidence-bridge `prop_firm_passed` aggregation across passing / failing artifact mixes.

The end-to-end search → live pipeline now reaches every gate the audit's P0-10 separation list requires: walkforward, CPCV, forward-test, and prop-firm evidence is built, persisted, aggregated into typed `LiveValidationEvidence`, and consumed by `LiveExecutionContract::validate_evidence` at the live boundary. The forex-app discovery service still needs to actively call `compute_discovery_prop_firm_artifacts` (currently only forward-test is wired) — that wiring is the next slice along with the live-sim runtime-model hash.

Next follow-on targets: wire the prop-firm helper into the forex-app discovery service so persisted artifacts appear automatically in the OOS path, populate `live_sim_runtime_model_hash` once a live-execution simulator is in place, explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` (P0-9), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-10: Follow-on Phase 28 completed — DiscoveryResult → LiveValidationEvidence bridge

Closed the second slice of P1-4 by giving downstream callers a single function that turns a `DiscoveryResult` into the typed `LiveValidationEvidence` record introduced in Phase 27, so a live bridge can call `LiveExecutionContract::validate_evidence` without re-deriving any pass/fail logic itself:

- added `live_validation_evidence_from_discovery(result)` in `forex-search::discovery` mapping `validation_gates.walkforward_passed` / `cpcv_passed` directly, and deriving `forward_test_passed` from the persisted artifacts: `Some(true)` only when at least one artifact exists AND every artifact reports `trade_count > 0` AND `net_profit > 0`, `Some(false)` when artifacts exist but at least one fails the rule, `None` when no artifact was produced (the live bridge then treats `None` as missing evidence whenever the gate is required);
- `prop_firm_passed` and `live_sim_runtime_model_hash` are intentionally `None` in this slice — Phase 25 added the contract surface for both, but the actual computation is opt-in per challenge / per simulator and belongs to its own phase;
- re-exported the helper from `forex-search::lib`;
- added `discovery::tests` covering the empty-forward-test passthrough, the all-profitable → `Some(true)` aggregation, the any-unprofitable → `Some(false)` aggregation, the zero-trades → `Some(false)` rejection, and the failed-walkforward / failed-cpcv propagation.

The pipeline now reaches end-to-end on the search side: `run_discovery_cycle_with_progress` produces the result, `compute_discovery_forward_test_artifacts` populates the held-out tail evidence, `live_validation_evidence_from_discovery` translates that into the typed record, and `LiveExecutionContract::validate_evidence` is the single point of acceptance / rejection at the live boundary.

Next follow-on targets: extend the bridge to populate `prop_firm_passed` from a discovery-driven prop-firm validator (third slice of P1-4), explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` (P0-9), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-09: Follow-on Phase 27 completed — live bridge validation evidence gate

Started P1-4 by extending `forex_core::contracts::LiveExecutionContract` so the live bridge can demand validation evidence from the new walk-forward / CPCV / forward-test / prop-firm / live-sim artifact kinds — Phase 5's contract validated provenance and runtime mode but not the *evidence* of validation:

- added `LiveValidationEvidence` (typed `walkforward_passed` / `cpcv_passed` booleans + `Option<bool>` flags for `forward_test_passed` / `prop_firm_passed` so callers can record "no test was run" vs "test failed", plus an `Option<String>` for the live-sim `runtime_model_hash`); a `passed_all()` constructor for the canonical green-path fixture and a `Default` impl that surfaces no evidence;
- extended `LiveExecutionContract` with five new fields (`require_walkforward_pass`, `require_cpcv_pass`, `require_forward_test_pass`, `require_prop_firm_pass`, `required_live_sim_runtime_model_hash`) plus matching builder methods (`with_required_*`); the existing `new` keeps every gate disabled, so upgrading a contract instance never silently tightens the bar;
- added `LiveExecutionContract::validate_evidence(&LiveValidationEvidence)` — production callers chain it after `validate_provenance` so the live bridge refuses artifacts that survived the provenance / temporal checks but lack the audited validation outcomes;
- added two new error variants to `ArtifactContractError`: `LiveRejectedFailedEvidenceGate { gate: &'static str }` for "evidence present but did not pass" and `LiveRejectedMissingEvidence { gate: &'static str }` for "the contract requires a gate the evidence record never carried"; both are surfaced through a new branch of the `Display` impl;
- re-exported `LiveValidationEvidence` from `forex_core::lib` so downstream crates (forex-search, forex-app, forex-cli) can build evidence records without depending on the contracts submodule directly;
- added contracts-tests covering: default vs `passed_all` evidence, `validate_evidence` accepting when no gate is required, walkforward / cpcv failed-gate rejection, forward-test missing-vs-failed differentiation, prop-firm missing-vs-failed differentiation, and live-sim runtime-model hash missing / mismatch / accept paths.

The contract surface for P1-4 is in place. Wiring real `LiveValidationEvidence` records from the discovery / forex-app pipeline (so a portfolio's persisted forward-test + prop-firm artifacts feed the gate at live-bridge load time) is the next concrete slice.

Next follow-on targets: build a `LiveValidationEvidence` from `DiscoveryResult` + persisted prop-firm / live-sim artifacts (P1-4 second slice), explicit degraded-mode metadata propagation through runtime layers (P1-3), DeterminismPolicy rollout into `forex-models` RL / exit-agent / genetic wrapper (P0-9), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-09: Follow-on Phase 26 completed — DeterminismPolicy through search

Started P0-9 by routing the genetic-search RNG seed through the canonical [`forex_core::contracts::DeterminismPolicy`] enum that was added in Phase 1 but until now was used only by the artifact provenance struct:

- added `GeneticSearchRuntimeOverrides::determinism_policy()` returning the typed enum: `Some(seed)` maps to `Deterministic { seed }`, `None` maps to `NonDeterministicAllowed` (preserving the existing "seed from OS RNG" behavior that fell out of the legacy `seed: Option<u64>` field);
- rewrote `build_search_rng` in `genetic/search_engine.rs` to consume the typed policy via an explicit `match`, so adding new modes (e.g. `BestEffort`) only requires a new arm rather than an `Option` re-interpretation;
- added a public `current_determinism_policy()` accessor at the genetic-runtime-overrides module + lib boundary so artifact-emitting paths can record the determinism mode in `ArtifactProvenance` without re-deriving it from the seed field;
- re-exported `DeterminismPolicy` itself from `forex-search::lib` so downstream crates do not need to depend on `forex_core::contracts` just to inspect / construct the enum;
- added unit tests covering the seed-Some → Deterministic, seed-None → NonDeterministicAllowed mapping; the `DeterminismPolicy::seed()` round-trip for all three variants; and the legality of the lib-level `current_determinism_policy()` accessor under any installed override.

The change is intentionally a thin typed wrapper — it does not change runtime behavior, but it gives every downstream caller (artifact provenance, RL, exit-agent exploration, sampling) a typed boundary to consume instead of `Option<u64>`. Subsequent P0-9 phases can migrate those callers (forex-models RL, forex-models exit-agent, evolution_math sampling) onto the same enum without further public-API churn.

Next follow-on targets: extend `DeterminismPolicy` consumption into `forex-models` (RL exploration, exit-agent search, genetic-discovery wrapper), wire the new validation artifacts into the search → portfolio → live bridge gate (P1-4), explicit degraded-mode metadata propagation through runtime layers (P1-3), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-09: Follow-on Phase 25 completed — live-sim and prop-firm artifact contracts

Closed out the P0-10 separation list by adding the last two missing typed contracts so each evaluation kind has an explicit `artifact_kind` boundary:

- added `LiveExecutionRuntimeModel` (slippage / latency / spread / commission / partial-fill rate / kill-zone toggle / backend kind) — recorded inside the live-sim artifact so a downstream live bridge can reject artifacts whose execution semantics do not match its current configuration;
- added `LiveExecutionSimulationSummary` (bars / trades / blocked-by-killzone / partial-fill counters + canonical `BacktestMetrics`) and the supporting `LiveExecutionSimulationScope` that hashes dataset + evaluation-config + strategy + runtime model into the shared `TemporalScopeHashes` boundary;
- added `LiveExecutionSimulationArtifactFile` with `LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND` / v1 schema, `validate_for_temporal_contract` rejecting wrong artifact kinds, unsupported schema versions, and temporal-contract drift, plus atomic `write_*` / validating `read_*` helpers reusing `forex-search::artifact_io`;
- added `PropFirmRiskRules` (loss / overall-DD / consistency / min-days / max-trades-per-day / profit-target knobs with FTMO-style defaults and per-field opt-out via `<= 0.0`), `PropFirmRiskValidationSummary` (per-rule pass/fail flags + worst-observed values + net return), and `PropFirmRiskValidationScope` hashing dataset + evaluation-config + strategy + rules + temporal scope;
- added `PropFirmRiskValidationArtifactFile` mirroring the canonical / walk-forward / forward-test pattern, plus atomic `write_*` / `read_*` helpers that reject wrong kind / unsupported schema / temporal drift on load;
- added `PropFirmRiskInput` and `compute_prop_firm_risk_summary` so callers feed observed trades from any source (canonical backtest, walk-forward, forward-test, live-sim) and the helper aggregates daily PnL / overall drawdown / consistency / trade limits / profit target into explicit pass/fail flags — entirely deterministic, no simulation embedded;
- re-exported every new type, IO helper, kind constant, schema-version constant, and compute helper from `forex-search::lib`;
- added validation-module unit tests covering the live-sim artifact's runtime-model + temporal-scope binding, atomic round-trip, wrong-kind / unsupported-schema rejection, plus the prop-firm helper's pass / daily-loss-breach paths and the prop-firm artifact's atomic round-trip and load-time validation.

P0-10 is now complete on the contract surface (`canonical_strategy_backtest`, `walk_forward_validation`, `forward_test_validation`, `live_execution_simulation`, `prop_firm_risk_validation` — five distinct kinds, each with a shared temporal scope + atomic-IO + load-time validation). Wiring the live-sim / prop-firm artifacts into the discovery / forex-app flow is intentionally deferred since it requires picking a slippage/latency model and a default rule set per challenge — both decisions belong in their own phases.

Next follow-on targets: P0-9 `DeterminismPolicy` rollout across genetic / RL / exit-agent paths, P1-3 explicit degraded-mode metadata propagation through the runtime layers, P1-4 wiring the new validation artifacts into the search → portfolio → live bridge gate (Phase 5 already enforces typed live-readiness; the new artifacts let live execution accept / reject based on prop-firm and live-sim evidence), P2-1 UI exposure of scheduler/hardware plans, and P2-3 operator-facing artifact safety documentation.


### 2026-05-09: Follow-on Phase 24 completed — forward-test wired into discovery

Continued P0-10 by wiring the Phase 23 forward-test contract into the discovery / forex-app pipeline so the held-out 20% tail produces persisted forward-test evidence:

- added `forward_test_validation_artifacts: Vec<ForwardTestValidationArtifactFile>` to `DiscoveryResult` and updated every internal/test constructor (and the discovery cycle return path) to populate it (defaults to `Vec::new()`);
- added `compute_discovery_forward_test_artifacts(portfolio, effective_feature_names, tail_features, tail_ohlcv, config)` in `forex-search::discovery` that aligns the tail's columns to the post-prefilter set, replays each gene via `signals_for_gene_full` + `compute_forward_test_summary`, and returns one `ForwardTestValidationArtifactFile` per strategy — the helper bails when the tail is missing any effective feature so callers cannot mismatch feature pipelines;
- added `save_forward_test_validation_artifacts(dir, result)` for atomic per-strategy persistence using the same content-addressable filename scheme as the canonical backtest / walk-forward helpers;
- extended `DiscoveryRunProfile` with `forward_test_validation_artifacts_observed` so the persisted profile JSON now reports forward-test coverage alongside canonical/walk-forward counts;
- re-exported the new compute / save helpers from `forex-search::lib`;
- wired the helper into the `forex-app` UI discovery service: after `run_discovery_cycle_with_progress` returns and the portfolio passes the existing non-empty check, the service builds a tail `Ohlcv` / `FeatureFrame` from rows past `wfv_bound`, calls `compute_discovery_forward_test_artifacts` with the resolved config, stores the result on `DiscoveryResult`, and persists it to a sibling `*.forward_tests/` directory next to the existing portfolio / profile / canonical / walk-forward outputs. Forward-test failures are logged at `warn` and do not block portfolio export, since the existing `walkforward_passed` / `cpcv_passed` gates remain the production-blocking signals;
- added `discovery` unit tests covering the empty-portfolio short-circuit, missing-feature rejection, one-artifact-per-strategy emission, atomic save, and the run-profile field exposure.

The forex-cli / orchestrator paths intentionally remain untouched in this phase — they currently process the full dataset without an explicit OOS partition, so wiring forward-test there would require deciding how to partition CLI-driven runs (a separate decision that does not belong inside the contract-completion slice).

Next follow-on targets: `LiveExecutionSimulationArtifactFile` and `PropFirmRiskValidationArtifactFile` typed contracts (P0-10), `DeterminismPolicy` rollout across genetic / RL / exit-agent paths (P0-9), explicit degraded-mode metadata propagation through runtime layers (P1-3), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-09: Follow-on Phase 23 completed — forward-test validation artifact contract

Started P0-10 by adding the missing `forward_test_validation` artifact type next to the existing canonical backtest (Phase 14) and walk-forward (Phase 10) contracts:

- added `FORWARD_TEST_VALIDATION_ARTIFACT_KIND` and a v1 schema constant alongside the existing canonical/walk-forward kind constants so the load-time validation paths cannot confuse the three;
- added `ForwardTestSummary` (intentionally flat — `bars`, canonical `BacktestMetrics`, `span_days` — because forward testing produces one unbiased OOS estimate, not a folded distribution);
- added `ForwardTestValidationScope` binding dataset / evaluation-config / strategy hashes plus the shared `TemporalScopeHashes`, with `new`, `from_parts`, and `validate_temporal_contract` mirroring the canonical/walk-forward scopes;
- added `ForwardTestValidationArtifactFile` with explicit kind/schema metadata and `validate_for_temporal_contract` rejecting wrong artifact kinds, unsupported schema versions, and temporal-contract drift;
- added atomic `write_forward_test_validation_artifact_atomic` and validating `read_forward_test_validation_artifact` IO helpers reusing `forex-search::artifact_io`;
- added `ForwardTestInput` and `compute_forward_test_summary` so callers can produce a forward-test summary from a pre-sliced held-out tail using the same evaluation core as canonical backtests, with input-length validation that refuses empty tails or mismatched array lengths;
- re-exported every new type / IO helper / kind constant from `forex-search`;
- added `validation` unit tests covering temporal-contract acceptance, temporal drift rejection, wrong-kind / unsupported-schema rejection, atomic round-trip, span-days computation across day boundaries, and the input-length / empty-tail rejection paths.

The artifact contract is in place for the next slice of P0-10 work; wiring the forward-test gate into the discovery / forex-app pipelines is intentionally deferred to a separate phase so this change stays scoped to the contract surface.

Next follow-on targets: wire the forward-test artifact into the discovery / forex-app held-out tail (Phase 24), `LiveExecutionSimulationArtifactFile` and `PropFirmRiskValidationArtifactFile` typed contracts, P0-9 `DeterminismPolicy` rollout across genetic / RL / exit-agent paths, P1-3 explicit degraded-mode metadata propagation through the runtime layers, P2-1 UI exposure of scheduler/hardware plans, and P2-3 operator-facing artifact safety documentation.


### 2026-05-09: Follow-on Phase 22 completed — install-once for remaining genetic env-readers

Closed out the genetic-module env-reader cleanup so `forex-search/src/genetic` no longer touches `std::env` per call:

- retired `RegimeLabelPolicy::from_env` and its private `env_i64` / `env_usize` / `env_f64` helpers (no callers in this crate or any dependent), eliminating 11 `FOREX_BOT_REGIME_LABEL_*` env reads from the regime-labels surface — the typed `RegimeLabelPolicy` struct fields remain the canonical configuration boundary;
- converted `SmcSearchConfig::from_env` into a lazy `OnceLock`-cached reader: the `FOREX_BOT_PROP_SMC_*` env vars are now read at most once per process, the existing `SmcSearchConfig::from_env()` call sites in `evolve_search_with_progress_and_limits` (×2) and `forex-models::genetic::train_with_discovery` keep their public API, and a new `install_smc_search_config_from_env` lets binaries force the cache to populate at startup;
- added `Default` for `SmcSearchConfig` (with audit-aligned probability defaults) so future tests / callers can build a deterministic config without touching either the env or the cache;
- introduced `SeenSignatureMemoryRuntimeOverrides` (typed `flush_every`, `load_max`, `max_entries`, `file_path`) plus the matching `OnceLock` install path (`install_seen_signature_memory_runtime_overrides`, `install_seen_signature_memory_runtime_overrides_from_env`, `current_seen_signature_memory_runtime_overrides`); rewrote `SeenSignatureMemory::from_env` to consult the cached overrides instead of inlining four `std::env::var` calls per construction (the file is still loaded each call because the in-memory state is per-search, but the env surface is read at most once);
- exported the new types and install helpers through `genetic::mod` and `forex_search::lib`, and extended `forex_search::install_search_runtime_overrides_from_env` so the unified entry point already wired into both binary `main()` paths now also primes the SMC and seen-signature caches;
- added module-local override unit tests for the new surfaces (SMC search-config defaults / legality, seen-signature-memory override defaults, current-accessor legality).

After Phase 22, `forex-search/src/genetic/*` has zero per-call `std::env::var` invocations: every env-driven knob is read at process startup via `install_search_runtime_overrides_from_env` (or lazily once on first use). The remaining env reads in `forex-search` live entirely outside the genetic submodule (CUDA/WGPU diagnostics in `cubecl_*.rs`, `discovery_gpu.rs`, and the `FOREX_BOT_REQUIRE_GPU` safety gate in `lib.rs`), and they are explicitly diagnostic rather than production-semantic.

Next follow-on targets: P0-9 `DeterminismPolicy` rollout across genetic / RL / exit-agent paths, P0-10 forward-test / live-execution / prop-firm validation contracts, P1-3 explicit degraded-mode metadata propagation through the runtime layers, P2-1 UI exposure of scheduler/hardware plans, and P2-3 operator-facing artifact safety documentation.


### 2026-05-09: Follow-on Phase 21 completed — typed cost profile and SMC weight overrides

Continued P0-8 cleanup into the most production-critical evaluation surface — symbol identity, account currency, pip value, spread/commission cost model, and SMC weighting — which previously read `FOREX_BOT_PROP_*` env vars on every `EvaluationConfig::default` call:

- added `CostProfileRuntimeOverrides` (typed `Option`s for `symbol`, `account_currency`, `pip_value`, `quote_to_account_rate`, `pip_value_per_lot`, `spread_pips`, `commission_per_trade`) so production callers that pass explicit values continue to bypass any fallback while env-driven fallbacks resolve through the typed boundary;
- added `SmcWeightRuntimeOverrides` covering the `FOREX_BOT_PROP_SMC_GATE` threshold and all 11 `FOREX_BOT_PROP_SMC_W_*` weights with documented unit-weight defaults;
- combined the two into `StrategyEvaluationRuntimeOverrides` with a single `from_env` reader and a `OnceLock` install path (`install_strategy_evaluation_runtime_overrides`, `install_strategy_evaluation_runtime_overrides_from_env`, `current_strategy_evaluation_runtime_overrides`);
- replaced the inline env reads in `infer_market_cost_profile` and the env-driven branches in `EvaluationConfig::default`; both now consult the typed overrides exactly once, eliminating the previous double-read of cost env vars (once via `infer_market_cost_profile`, once via `EvaluationConfig::default`);
- exported every new override type through `genetic::mod` and `forex_search::lib`, and extended `install_search_runtime_overrides_from_env` so the unified entry point already wired into the binary `main()` paths now also resolves the strategy-evaluation overrides — production binaries continue to honour the legacy env vars with one explicit call at startup;
- added override-module unit tests covering documented defaults for `CostProfileRuntimeOverrides`, the unit-weight default invariant for `SmcWeightRuntimeOverrides`, the neutral-state composition of `StrategyEvaluationRuntimeOverrides`, and the legal-values guarantee on the runtime accessor when no install has happened.

After Phase 21, `forex-search/src/genetic/strategy_gene.rs` no longer touches `std::env` at all. The remaining env readers in the crate live inside typed `from_env` constructors (`SmcSearchConfig`, `SeenSignatureMemory`, `RegimeLabelPolicy`) that are already audit-aligned struct boundaries used at construction time rather than per-call.

Next follow-on targets are `genetic/regime_labels::RegimeLabelPolicy::from_env` and `genetic/smc_indicators::SmcSearchConfig::from_env` (already typed, but read on every search invocation), `evolution_math::SeenSignatureMemory::from_env` (file path + flush thresholds), `DeterminismPolicy` rollout (P0-9), additional canonical evaluation contracts (P0-10), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-09: Follow-on Phase 20 completed — finished search-engine env-flag cleanup

Closed out the Phase 19 follow-on by extending `GeneticSearchRuntimeOverrides` to cover every remaining `FOREX_BOT_PROP_*` knob that the genetic search engine still read inline at the top of `evolve_search_with_progress_and_limits`:

- added three nested override sub-structs — `SmcGateOverrides` (start/end/curve/stagnation_step), `ArchiveScoringOverrides` (mode/min_net/min_pf/min_sharpe), `SelectionPolicyOverrides` (parent/survivor policies, immigrant ratio, survivor fraction, temperature) — plus a top-level `seen_retry_attempts` field, all with audit-aligned defaults that match the legacy env-var fallbacks;
- consolidated the legacy clamp logic (curve ≥ 0.1, stagnation_step ≥ 0, immigrant ratio / survivor fraction in `[0, 0.95]`, temperature ≥ 1e-3, retry attempts ≥ 1) into typed `resolved_*` and `effective_*` accessor methods so the search engine no longer duplicates min/max gymnastics at the call site;
- extended `GeneticSearchRuntimeOverrides::from_env` to consume the full `FOREX_BOT_PROP_SMC_GATE_*`, `FOREX_BOT_PROP_ARCHIVE_*`, `FOREX_BOT_PROP_RANDOM_IMMIGRANTS`, `FOREX_BOT_PROP_SURVIVOR_FRACTION` (or legacy `FOREX_BOT_PROP_ELITE_FRACTION`), `FOREX_BOT_PROP_PARENT_SELECTION` / `SURVIVOR_SELECTION` / `SELECTION_TEMPERATURE`, and `FOREX_BOT_PROP_SEEN_RETRY` env surface in one place;
- removed the inline `env_f32` / `env_f64` / `env_str` closures plus every `std::env::var` call from `evolve_search_with_progress_and_limits`; the body now reads only `current_genetic_search_runtime_overrides()` and the resolved sub-structs;
- pruned the now-unused `ParentSelectionPolicy` import from `search_engine.rs` since the typed override boundary owns parent-policy parsing;
- added `runtime_overrides` unit tests covering documented defaults across every sub-struct, the SMC-gate clamp behavior on invalid curve/stagnation values, the selection-policy clamp behavior on out-of-range immigrant ratio / survivor fraction / temperature, and the seen-retry minimum-of-1 floor.

The search engine no longer touches the environment at all during a run; the legacy `SmcSearchConfig::from_env` and `SeenSignatureMemory::from_env` constructors are the only remaining env-readers in `forex-search/src/genetic`, and both are already typed boundaries used at struct-construction time rather than per-call.

Next follow-on targets are the remaining production-affecting env vars in `genetic/strategy_gene` (SMC weight knobs and prop-firm cost profile fields), `genetic/regime_labels` and `genetic/smc_indicators` (still typed `from_env` constructors, but they read on every call), `DeterminismPolicy` rollout (P0-9), additional canonical evaluation contracts (P0-10), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-09: Follow-on Phase 19 completed — typed genetic search runtime overrides

Continued P0-8 cleanup into the genetic search engine, where the most production-affecting `FOREX_BOT_*` knobs (RNG seed, novelty weighting, tournament size, stagnation patience, archive cap) were previously read with inline `std::env::var` calls at the top of every `evolve_search` invocation:

- added a new `forex_search::genetic::runtime_overrides` module with `GeneticSearchRuntimeOverrides` (`seed`, `novelty_weight`, `stagnation_patience`, `tournament_size_override`, `archive_cap_override`) plus a one-shot `from_env` reader and the `OnceLock` install path (`install_genetic_search_runtime_overrides`, `install_genetic_search_runtime_overrides_from_env`, `current_genetic_search_runtime_overrides`);
- consolidated the legacy formula for tournament size, archive cap, and stagnation patience into typed methods (`effective_tournament_size`, `effective_archive_cap`, `effective_stagnation_patience`) so `search_engine.rs` no longer duplicates min/max/clamp logic at the call site;
- replaced `build_search_rng` and the inline `FOREX_BOT_SEARCH_SEED` / `FOREX_BOT_NOVELTY_WEIGHT` / `FOREX_BOT_PROP_STAGNATION_GENS` / `FOREX_BOT_PROP_TOURNAMENT_SIZE` / `FOREX_BOT_PROP_ARCHIVE_CAP` reads in `evolve_search_with_progress_and_limits` with a single resolved `current_genetic_search_runtime_overrides()` call;
- retired the dead `DiversityArchiveConfig::from_env` helper (no callers in this crate or any dependent), eliminating the last `FOREX_BOT_PROP_DIVERSE_*` env reads from the diversity archive surface and leaving an explanatory comment that future env-driven defaults must come through a typed `*RuntimeOverrides` boundary;
- re-exported the new genetic-search override types from `forex-search`, and extended `install_search_runtime_overrides_from_env` so the unified entry point already wired into `forex-cli::main` and `forex-app::main` now also resolves the genetic-search overrides — production binaries continue to pick up the legacy env vars with one explicit call at startup;
- added override-module unit tests covering documented defaults, the legacy population-derived tournament-size formula (including the algorithmic minimum-of-2 clamp), the archive cap floor/ceiling around `population` and `200_000`, and the stagnation patience minimum-of-1.

Next follow-on targets are the remaining `FOREX_BOT_PROP_*` knobs in `evolve_search_with_progress_and_limits` (SMC gate curves, archive thresholds, immigrant/elite/parent/survivor selection knobs), `DeterminismPolicy` rollout (P0-9), additional canonical evaluation contracts (P0-10), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-09: Follow-on Phase 18 completed — typed backtest and quality runtime overrides

Continued the P0-8 cleanup beyond discovery into the canonical backtest and strategy-quality math, both of which previously read `FOREX_BOT_*` env vars on every metric call:

- added `BacktestRuntimeOverrides` (`initial_equity`, `month_capacity`) plus a one-shot `from_env` reader and a process-wide `OnceLock` install path (`install_backtest_runtime_overrides`, `install_backtest_runtime_overrides_from_env`, `current_backtest_runtime_overrides`) in `forex-search::eval`;
- replaced the env-driven `BacktestSettings::initial_equity` / `month_capacity` accessors so canonical backtest math now resolves through the typed overrides instead of `std::env::var` per call, while keeping the same accessor signatures for compatibility with all existing struct-literal call sites;
- added `QualityRuntimeOverrides` (`min_trades_per_month`, `trading_days_per_month`) plus the matching install/current accessors in `forex-search::quality`, and removed the inline env reads from monthly metric aggregation and the trade-frequency annualization helper;
- exposed both override types and a unified `forex_search::install_search_runtime_overrides_from_env` convenience entry point so binaries can resolve every legacy `FOREX_BOT_BACKTEST_*` / `FOREX_BOT_PROP_*` / `FOREX_BOT_TRADING_DAYS_PER_MONTH` knob in one explicit call;
- wired `install_search_runtime_overrides_from_env` into `forex-cli::main` and `forex-app::main` so the existing env-driven behavior is preserved end-to-end while `forex-search` itself stops touching the environment during a run;
- added `eval` and `quality` override unit tests covering documented defaults, clamp behavior on bad inputs, and the deterministic fallback path of the runtime accessors when no install has happened.

Next follow-on targets are deeper P0-8 cleanup of the remaining `FOREX_BOT_PROP_*` knobs in genetic search/diversity/regime modules, `DeterminismPolicy` rollout (P0-9), additional canonical evaluation contracts (P0-10), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-09: Follow-on Phase 17 completed — typed discovery runtime overrides

Implemented the next P0-8 slice by removing production-affecting env-var reads from `forex-search::discovery` and replacing them with a typed runtime-overrides boundary:

- added `DiscoveryRuntimeOverrides` (typed `prefilter_top_k`, `prefilter_insample_frac`, `funnel_stage1_pct`) plus a single explicit `DiscoveryRuntimeOverrides::from_env` reader for the legacy `FOREX_BOT_PREFILTER_TOP_K`, `FOREX_BOT_PREFILTER_INSAMPLE`, and `FOREX_BOT_FUNNEL_STAGE1_PCT` env vars;
- added `runtime_overrides` to `DiscoveryConfig` with deterministic defaults and a `with_env_runtime_overrides` opt-in helper, so `DiscoveryConfig::default()` and `DiscoveryConfig::from_settings` no longer pick up ambient env state;
- removed the env reads inside `run_discovery_cycle_with_progress` and `prefilter_features` and made `prefilter_features` accept the in-sample fraction as an explicit argument resolved from typed config;
- exposed the resolved `prefilter_top_k`, `prefilter_insample_frac`, and `funnel_stage1_pct` in `DiscoveryRunProfile` so persisted profile JSONs document exactly which knobs were used;
- wired the `with_env_runtime_overrides` opt-in into `forex-cli` `discover`, `forex-search::DiscoveryOrchestrator::run_batch`, the `forex-app` discovery service (also using the resolved config to write the profile), and `forex-models::genetic` so existing call sites keep their previous env-driven behavior explicitly;
- added discovery tests proving default determinism, clamp behavior on bad inputs, that `DiscoveryConfig::default` ignores the legacy env vars, and that the run profile reports the resolved overrides.

Next follow-on targets are wider P0-8 cleanup of the remaining `FOREX_BOT_*` knobs in eval/genetic search/regime modules, `DeterminismPolicy` rollout (P0-9), additional canonical evaluation contracts (P0-10), UI exposure of scheduler/hardware plans (P2-1), and operator-facing artifact safety documentation (P2-3).


### 2026-05-09: Follow-on Phase 16 completed — discovery validation artifact persistence

Continued the discovery validation wiring from Phase 15 by persisting the canonical backtest and walk-forward validation artifacts that Phase 15 only computed in memory:

- added `save_canonical_backtest_artifacts` and `save_walkforward_validation_artifacts` in `forex-search::discovery`, each writing one atomic per-strategy JSON file under a caller-supplied directory using the shared validation artifact IO helpers and content-addressable filenames derived from the strategy hash;
- re-exported the new save helpers from `forex-search` so CLI/UI/orchestration callers can persist validation evidence without reaching into the validation submodule;
- wired the helpers into `DiscoveryOrchestrator::run_batch` so each `{symbol}_{tf}` work unit produces sibling `{symbol}_{tf}_canonical_backtests/` and `{symbol}_{tf}_walkforward_validations/` directories whenever the discovery result carries those artifacts;
- wired the same helpers into the `forex-cli discover` and `forex-app` discovery service so single-symbol and UI-driven runs produce matching `.canonical_backtests/` and `.walkforward_validations/` directories alongside the existing portfolio/profile/quality/trade-log outputs;
- added discovery tests covering per-strategy file emission, the empty-result no-op path, and filename sanitization for the `fnv64:` strategy hash format.

Next follow-on targets are UI exposure of scheduler-owned hardware plans and operator-facing artifact safety documentation.


### 2026-05-08: Follow-on Phase 15 completed — discovery validation gates and profile exports

Continued the discovery/backtest validation wiring from P0-10/P1-4/H39 by making portfolio export depend on explicit validation gate state instead of implicit discovery success:

- added `DiscoveryValidationGates` to `DiscoveryResult`, including mandatory `walkforward_passed` and `cpcv_passed` booleans, canonical/walk-forward artifact counts, CPCV fold count/ratio, and the temporal-contract hash used for validation;
- wired final portfolio candidates into canonical backtest artifact construction and per-strategy walk-forward validation artifacts using the shared `TemporalFeatureContract`;
- added a CPCV gate based on purged combinatorial test folds and the configured minimum profitable-fold ratio;
- changed `save_portfolio_json` to accept the full `DiscoveryResult`, require both validation gates before writing, and continue exporting indicator names from the effective post-prefilter feature list;
- extended `DiscoveryRunProfile` with validation gate status, artifact counts, CPCV diagnostics, and the validation temporal-contract hash.

Next follow-on targets are persisting the generated validation artifacts alongside portfolio/profile outputs, UI exposure of scheduler-owned hardware plans, and operator-facing artifact safety documentation.


### 2026-05-08: Follow-on Phase 14 completed — canonical backtest artifact persistence

Continued the backtest/validation artifact persistence wiring after shared search artifact IO was established:

- added a typed `BacktestMetrics` wrapper for the canonical evaluator metric layout, preserving compatibility with the existing `[f64; 11]` array while making artifact boundaries explicit;
- added `CanonicalBacktestScope` to bind dataset, evaluation-config, strategy, and shared temporal-scope hashes;
- added `CanonicalBacktestArtifactFile` with explicit artifact kind/schema metadata, atomic JSON write/read helpers, and temporal-drift rejection on load;
- re-exported the canonical backtest artifact types and IO helpers from `forex-search`.

Next follow-on targets are wiring canonical backtest and walk-forward artifacts into discovery/OOS gating and run-profile exports, UI exposure of scheduler-owned hardware plans, and operator-facing artifact safety documentation.


### 2026-05-08: Follow-on Phase 13 completed — remove legacy search JSON writers

Continued the add/remove cleanup across older search files now that shared artifact IO exists:

- moved discovery portfolio/profile/quality/trade-log JSON exports onto `forex-search::artifact_io::write_json_atomic`;
- moved GPU genome exports in both GPU and non-GPU build paths onto the same shared atomic JSON writer;
- moved strategy-quality ranking exports onto the shared writer and removed the local serialization fallback that could silently emit an empty JSON array;
- removed direct `serde_json::to_string_pretty` + `fs::write` JSON persistence from `forex-search` outside the shared artifact IO module.

Next follow-on targets remain backtest/validation artifact persistence wiring, UI exposure of scheduler-owned hardware plans, and operator-facing artifact safety documentation.


### 2026-05-08: Follow-on Phase 12 completed — shared search artifact IO cleanup

Continued the staged add/remove cleanup requested by the audit after checkpoint and walk-forward validation artifacts both needed durable JSON persistence:

- added a focused `forex-search::artifact_io` helper for stable JSON hashing, atomic JSON writes, typed JSON reads, temporary artifact names, and shared FNV-1a hashing;
- removed the private atomic write/read/hash implementation from `checkpoint.rs` and kept the checkpoint API backed by the shared helper;
- removed the validation module's dependency on checkpoint internals for `stable_json_hash`;
- added walk-forward validation artifact write/read entry points that reuse the shared atomic artifact IO path and validate temporal contracts on load;
- preserved the public `checkpoint::stable_json_hash` compatibility re-export while moving ownership to the shared module.

Next follow-on targets remain discovery/backtest persistence wiring for validation artifacts, UI exposure of resolved scheduler/hardware plans, and operator-facing artifact safety documentation.


### 2026-05-08: Follow-on Phase 11 completed — remove duplicate temporal scope fields

Implemented the requested cleanup/removal work after both checkpoint and walk-forward validation artifacts carried temporal semantics:

- added a shared `TemporalScopeHashes` contract in `forex-core::contracts` for aggregate temporal, timestamp, feature-availability, and label-policy hashes;
- removed duplicated per-artifact temporal hash fields from search checkpoint and walk-forward validation scopes;
- removed the walk-forward validation module's local temporal hash comparison helper and delegated drift checks to the shared contract type;
- preserved checkpoint resume and walk-forward validation drift tests while adding contract coverage for shared temporal-scope hash validation.

Next follow-on targets are wiring these validation artifacts into discovery/backtest persistence, exposing scheduler/hardware plans in the UI, and documenting operator-facing artifact safety rules.

### 2026-05-08: Follow-on Phase 10 completed — walk-forward validation temporal artifact

Implemented the next audited slice from P0-7/P0-10/P1-4 by extending temporal provenance beyond search checkpoints into canonical validation output:

- added a `WalkforwardValidationScope` that binds dataset, evaluation config, temporal-contract, timestamp-policy, feature-availability-policy, and label-policy hashes;
- added a `WalkforwardValidationArtifactFile` wrapper with explicit artifact kind/schema metadata and validation against the active `TemporalFeatureContract`;
- made walk-forward summaries serializable/deserializable so validation artifacts can be persisted and reloaded without losing typed semantics;
- re-exported the validation artifact types from `forex-search`;
- added tests proving validation artifacts accept matching temporal semantics and reject temporal drift or wrong artifact kinds.

Next follow-on targets are wiring these validation artifacts into discovery/backtest persistence, exposing scheduler/hardware plans in the UI, and documenting operator-facing artifact safety rules.

### 2026-05-07: Follow-on Phase 9 completed — remove legacy non-temporal checkpoint scopes

Implemented the requested staged removal work from P1-5 after the temporal checkpoint contract existed and tests covered the new path:

- removed the legacy checkpoint-scope construction path that allowed search checkpoints to be created without a `TemporalFeatureContract`;
- removed serde defaults for temporal scope fields so v2 checkpoint payloads must carry explicit temporal, timestamp, feature-availability, and label-policy hashes;
- collapsed checkpoint scope builders to require the temporal contract at construction time rather than adding it as an optional follow-up mutation;
- updated checkpoint tests to use explicit temporal contracts for all resume and portfolio fixtures, preserving the existing atomic/resume coverage while enforcing the new invariant.

Next follow-on targets are wiring the temporal contract into canonical backtest/forward artifacts, exposing scheduler/hardware plans in the UI, and documenting operator-facing artifact safety rules.

### 2026-05-07: Follow-on Phase 8 completed — temporal search checkpoint scope

Implemented the next audited slice from P0-6/P0-7 search resume and temporal causality hardening:

- extended `SearchCheckpointScope` with timestamp-policy, feature-availability-policy, label-policy, and aggregate temporal-contract hashes;
- added builders that derive search checkpoint scope hashes directly from `TemporalFeatureContract`;
- made checkpoint resume reject temporal contract drift in addition to config, dataset, and search-space drift;
- bumped search checkpoint and portfolio artifact schema versions so old semantics are invalidated explicitly;
- copied the source checkpoint scope into portfolio-selection artifacts so selected portfolios remain bound to the temporal semantics that produced them;
- added checkpoint tests for temporal-policy resume rejection and portfolio source-scope preservation.

Follow-on removal of the legacy non-temporal checkpoint scope is now complete. Next follow-on targets are wiring the temporal contract into canonical backtest/forward artifacts, exposing scheduler/hardware plans in the UI, and documenting operator-facing artifact safety rules.

### 2026-05-07: Follow-on Phase 7 completed — shared temporal feature contract

Implemented the next audited slice from P0-7 timestamp, MTF, and causality policy hardening:

- added a canonical `TemporalFeatureContract` in `forex-core::contracts` that binds timestamp policy, MTF/feature availability policy, label policy, walk-forward policy, and live-readiness policy into one typed boundary;
- added stable policy hashes for timestamp and feature-availability policies so artifact provenance can be validated against the concrete temporal contract rather than local subsystem assumptions;
- made canonical trading contracts reject lookahead and partial higher-timeframe availability before they can be treated as live-capable semantics;
- added provenance validation for temporal hashes so search/training/backtest/live paths can reject artifacts whose timestamp, feature availability, or label policies drift from the active contract;
- added unit tests for strict live temporal hashing, lookahead/partial-MTF rejection, and provenance mismatch rejection.

Follow-on temporal checkpoint wiring is now complete. Next follow-on targets are wiring the same temporal contract into backtest/forward artifact creation, exposing scheduler/hardware plans in the UI, and documenting operator-facing artifact safety rules.

### 2026-05-06: Phase 1 completed — canonical artifact contracts

Implemented and removed the completed Phase 1 contract items from the active implementation sequence:

- added the shared Rust contract surface in `forex-core` for artifact kinds, provenance, timestamp policy, feature availability policy, determinism policy, backend/runtime mode metadata, degraded runtime reasons, and device assignments;
- added separate typed training-model, search-checkpoint, portfolio-selection, model-runtime, and live-ready strategy artifact contracts;
- added generic and typed artifact envelopes that validate required provenance fields, schema version, backend/device assignment consistency, artifact-kind intent, and live-readiness;
- added contract tests covering missing provenance rejection, wrong artifact-kind rejection, typed artifact-kind separation, degraded live artifact rejection, and acceptance of a canonical live-ready artifact.

Next active execution target is Phase 2 scheduler ownership, starting with canonical `HardwareProfile`/`WorkloadExecutionPlan` usage and moving runtime decisions out of local env probes.

### 2026-05-06: Phase 2 started — scheduler-owned workload assignments

Implemented and removed the completed first scheduler-ownership items from the active implementation sequence:

- added stable `HardwareProfile` identity so artifacts and resolved assignments can reference the canonical detected profile without depending on probe timestamp;
- added scheduler-owned CPU budget, GPU budget, precision policy, and resolved workload assignment types derived from `WorkloadExecutionPlan`;
- added conversion from planner backends into artifact `DeviceAssignment`/`BackendKind` metadata so downstream modules can consume resolved assignments instead of re-probing device state;
- added scheduler tests covering profile identity stability, CUDA search assignment budgets/device metadata, and explicit degraded metadata when a GPU-oriented search path falls back to CPU.

Next active execution targets are moving env parsing into typed config resolution and making models/search receive these assignments instead of deciding device locally.

### 2026-05-06: Phase 2 continued — typed runtime override resolution

Implemented and removed the env-parsing scheduler item from the active implementation sequence:

- added `HardwareRuntimeOverrides` as the typed boundary for diagnostic environment overrides such as CPU budget, requested training precision, backend precision hints, and WGPU hint devices;
- added explicit `HardwareExecutionPlan::from_settings_profile_and_overrides` so production callers can pass resolved config instead of allowing env vars to silently rewrite the plan;
- moved hardware-probe WGPU hint devices and backend precision hints behind `HardwareRuntimeOverrides`;
- added scheduler tests proving explicit runtime overrides control CPU budget/precision without env mutation and that WGPU hint devices are consumed from typed overrides.

Next active execution target is making models/search receive scheduler assignments instead of deciding device locally.

### 2026-05-06: Phase 2 completed — model/search assignment consumers

Implemented and removed the final scheduler-ownership item from the active implementation sequence:

- added `GpuDiscoveryConfig::apply_scheduler_assignment` / `with_scheduler_assignment` so search discovery can consume `ResolvedWorkloadAssignment` for backend, devices, precision, and chunk size instead of deciding those locally;
- added model hardware `select_device_from_assignment` so model code can consume scheduler-owned `DeviceAssignment` metadata directly;
- added tests proving search assignment consumption, non-search assignment rejection, CUDA model assignment conversion, and degraded CPU assignment conversion.

Phase 2 is now complete and has been removed from the active plan. Next active execution target is Phase 3 search/checkpoint hardening, starting with atomic checkpoint writes.

### 2026-05-07: Phase 3 completed — search/checkpoint hardening

Implemented and removed the Phase 3 search/checkpoint-hardening items from the active implementation sequence:

- added a search checkpoint artifact file boundary in `forex-search` with schema/kind metadata, hash-bound resume scope, completed-generation state, deterministic seed-chain metadata, evaluated-candidate ledger, genes, and metrics;
- added durable atomic JSON artifact writes using a same-directory temporary file, `fsync`, and atomic rename for checkpoints and derived portfolio artifacts;
- added resume validation that rejects changed config, dataset, search-space hashes, unsupported schema versions, and non-checkpoint artifact kinds;
- added deterministic seed-chain and evaluated-candidate ledger helpers so resumed searches can avoid silently repeating candidates and can reproduce candidate seeds;
- added a separate portfolio-selection artifact file type whose kind and schema are distinct from search checkpoints and whose source checkpoint hash binds it back to the checkpoint it was selected from;
- added unit tests covering atomic checkpoint write/read, hash mismatch rejection, wrong-kind rejection, deterministic seed derivation, duplicate ledger suppression, and checkpoint-vs-portfolio artifact separation.

Phase 3 is now complete and has been removed from the active plan. Next active execution target is Phase 4 CPU/GPU parity tests.

### 2026-05-07: Phase 4 completed — CPU/GPU parity tests

Implemented and removed the Phase 4 CPU/GPU-parity items from the active implementation sequence:

- added a `forex-search::parity` module with canonical, approximate, and degraded execution semantics for parity comparisons;
- added metric-matrix parity reporting with tolerance tracking, maximum absolute delta, and per-cell mismatch records;
- made canonical comparisons reject silent mismatches while approximate and degraded comparisons remain explicit reports;
- added a small deterministic search/evaluation fixture comparing scalar CPU reference metrics against the population evaluator used by GPU-enabled evaluation paths;
- added tests covering exact canonical parity, silent mismatch rejection, and explicit approximate/degraded mismatch reporting.

Phase 4 is now complete and has been removed from the active plan. Next active execution target is Phase 5 live bridge gate.

### 2026-05-07: Phase 5 completed — live bridge gate

Implemented and removed the Phase 5 live-bridge-gate items from the active implementation sequence:

- added `LiveExecutionContract` as the live/runtime boundary for expected feature schema, timestamp policy, feature availability policy, symbol universe, runtime/cost configuration, risk configuration, required backend, and maximum artifact age;
- added live contract validation for typed and untyped artifact envelopes so live callers can validate against their current execution context before accepting an artifact;
- added explicit live rejection errors for mismatched contract fields and stale artifacts in addition to existing wrong-kind and unsafe-runtime rejection;
- added tests proving live accepts a matching live-ready strategy and rejects search checkpoints, diagnostic-only artifacts, stale artifacts, backend mismatches, feature/timestamp/runtime/risk mismatches, and degraded artifacts.

Phase 5 is now complete and has been removed from the active plan. Next active execution target is Phase 6 modular cleanup.


### 2026-05-07: Phase 6 started — contract module split

Implemented and removed the first modular-cleanup item from the active implementation sequence:

- split the large `forex-core` contract surface into focused modules for primitives/provenance, artifact envelopes, live execution gates, contract errors, and contract tests;
- kept `forex_core::contracts` as the public compatibility module by re-exporting the split modules from `contracts.rs`;
- preserved existing contract behavior and test coverage while reducing the previous contract god-file into smaller ownership boundaries.

Phase 6 remains active. Next active execution targets are moving backend logic behind adapters, removing duplicate local helpers, and deleting stale code only after tests pass.

### 2026-05-07: Phase 6 continued — scheduler backend adapter split

Implemented and removed the backend-adapter modular-cleanup item from the active implementation sequence:

- moved scheduler accelerator backend ownership into `forex-core::system::backends`;
- kept `AcceleratorBackend` publicly re-exported through `forex_core::system` and `forex_core` so existing callers remain source-compatible;
- isolated backend normalization and primary-backend selection from the hardware planner body while preserving scheduler tests.

Phase 6 remains active. Next active execution targets are removing duplicate local helpers and deleting stale code only after tests pass.

### 2026-05-07: Phase 6 continued — search scheduler-assignment helper deduplication

Implemented and removed the duplicate-local-helper modular-cleanup item from the active implementation sequence:

- added a shared `forex-search::scheduler_assignment` helper for converting scheduler `ResolvedWorkloadAssignment` metadata into local discovery backends;
- removed duplicated `accelerator_backend_from_assignment` implementations from the GPU and non-GPU discovery paths;
- preserved both discovery assignment tests while keeping scheduler-assignment conversion in one owning module.

Phase 6 remains active. Next active execution target is deleting stale code only after tests pass.

### 2026-05-07: Phase 6 completed — stale legacy bootstrap cleanup

Implemented and removed the final stale-code cleanup item from the active implementation sequence:

- removed the unused legacy cTrader bootstrap batch implementation from `forex-app` after the context-driven bootstrap path and job reporting flow had already replaced it;
- kept the active bootstrap path, cTrader account runtime integration, and job snapshot reporting intact;
- ran targeted/full checks where possible and documented the app test environment limitation caused by `catboost-rust` attempting to download headers without network reachability.

Phase 6 is now complete and has been removed from the active plan.
