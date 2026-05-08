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
