# Audit coverage matrix — 2026-05-10

Cross-references every source audit under `docs/audits/` against the
Phase 1-70 follow-on slice. Built after Phase 70 closure to identify
what still has actionable items the consolidated plan
(`ALL_AUDITS_CONSOLIDATED_2026-05-06.md`) marked but did not land.

Postscript: Phases 72-75 landed the quick-win bucket this matrix originally
recommended: Python/PyO3 guardrail, `allow(dead_code)` audit, indicator
registry metadata, and registry validation surface.

Postscript 2: Phases 76-79 landed the typed model runtime propagation slice
for `PredictionMetadata`, NEAT, CRFMNES / neuro-evolution, statistical linear
models, and swarm forecasting results. Legacy string fields remain for
backward compatibility, but new writes now carry typed `BackendKind`,
`RuntimeMode`, and degraded-reason metadata where this slice touched artifacts.

Postscript 3: Phases 80-85 cleaned up the stale `forex-models` contract test
failures that remained after typed propagation. The slice hardened Burn/deep
runtime provenance, exit-agent trained-artifact reports, streaming Hoeffding
runtime details, and RL fallback/runtime contracts; the forex-models lib suite
is green at 335 tests after this cleanup.

Postscript 4: Phase 86 landed the training-model artifact producer. Every
`training_orchestrator` model save now writes a typed
`TrainingModelArtifactContract` envelope beside the runtime profile, with
artifact provenance, dataset fingerprint, feature/label/runtime hashes, backend
kind, runtime mode, device assignment, hardware profile id, and source commit.

Postscript 5: Phase 87 deduplicated high-confidence repo JSON artifact IO.
`forex_core::storage::json` now owns atomic writes, backup writes, typed reads,
temporary artifact paths, and stable hashes; `forex-search::artifact_io` became
a compatibility re-export and common `forex-models` metadata/artifact writers
now use the shared helper. The same slice also moved tree-model JSON sidecars
and swarm forecaster JSON save/load onto the core-backed helpers, leaving local
file writers only for binary/raw payloads and test corruption fixtures.

`✅` = addressed by the listed phase(s); `🟡` = partially addressed;
`🔴` = not addressed yet (actionable gap).

| # | Source audit | Status | Phases | Open items |
|---|---|---|---|---|
| 1 | `architecture_unification_duplicate_code_cleanup` | ✅ | 6, 12, 13, 61-70, 87 | — |
| 2 | `artifact_intent_clarification_training_vs_search_resume` | ✅ | 1, 3, 8 | — |
| 3 | `core_config_domain_modularization` | 🟡 | 6, 17-22 | training/search large-file split deeper than Phase 6 |
| 4 | `cpu_gpu_semantic_parity_requirement` | ✅ | 4 | — |
| 5 | `custom_cuda_kernel_preservation` | ✅ | preserved (no deletion) | — |
| 6 | `dead_code_and_stale_artifacts` | 🟡 | 6, 9, 12, 13, 61-70, 73 | vendor-patches review; CI feature-matrix |
| 7 | `deep_duplicate_logic_and_unified_scheduler` | ✅ | 2, 6 | — |
| 8 | `deep_search_engine_state_audit_pass2` | ✅ | 3, 8, 9 | — |
| 9 | `evaluation_contract_deep_audit_pass3` | ✅ | 10-16, 23-31 | — |
| 10 | `evolution_neat_crfmnes_gpu_first` | 🟡 | preserved kernels, 77-78 | runtime parity tests for evolution kernels |
| 11 | `feature_timestamp_mtf_causality_deep_audit_pass5` | ✅ | 7 | — |
| 12 | `forex_data_functional` | 🟡 | 7, 68, 74-75 | explicit candle-timestamp-policy threading inside resample / hpc_ta / quant_features / smc / parquet_migration; volume-validation surface |
| 13 | `forex_models_functional` | 🟡 | 26, 59, 67, 76, 79, 80-85 | ONNX legacy boundary |
| 14 | `forex_search_functional` | ✅ | 16-32, 45-51 | — |
| 15 | `generic_scheduler_small_files_refactor_note` | ✅ | 6 | — |
| 16 | `gpu_cuda_hpc_parity_deep_audit_pass4` | 🟡 | 4 | parity tests beyond strategy search (statistical / NEAT / CRFMNES backends) |
| 17 | `gpu_first_kernel_everywhere_report` | ✅ | preserved, 76-79 | — |
| 18 | `hardware_autodetect_config_ui_architecture` | 🟡 | 2 | UI hardware/runtime panel exposing scheduler-owned plans (P2-1) |
| 19 | `model_runtime_backend_fragmentation` | 🟡 | 2, 76-79, 80-85 | model runtime artifact contract wired in forex-models bridge |
| 20 | `modularization_maintainability_refactor_principle` | ✅ | 6 + 61-70, 87 | — |
| 21 | `python_pyo3_legacy` | ✅ | confirmed clean, 72 | — |
| 22 | `quality_challenge_validation_refactor` | ✅ | 25, 29-31 | — |
| 23 | `rust_env_flags_config_debt` | ✅ | 17-22 | — |
| 24 | `search_backtest_forward_cpu_gpu` | ✅ | 14-16, 23-31 | — |
| 25 | `search_checkpoint_resume_requirement` | ✅ | 3, 8, 9 | — |
| 26 | `search_discovery_pipeline` | ✅ | 14-31 | — |
| 27 | `search_gpu_discovery_scheduler` | 🟡 | 2 | CUDA discovery scheduler resource budgeting wired through `GpuDiscoveryConfig` (chunk size already typed; VRAM budget ledger missing) |
| 28 | `search_orchestration_refactor` | ✅ | 6 | — |
| 29 | `search_portfolio_artifact_contract` | ✅ | 1, 8, 14-16 | — |
| 30 | `search_to_live_bridge` | ✅ | 5, 27, 28, 30, 48 | — |
| 31 | `training_model_artifact_contract` | ✅ | 1, 86 | — |
| 32 | `unified_module_logic_architecture` | ✅ | 6 + 61-70 | — |
| 33 | `universal_hardware_parity_requirement` | ✅ | 2, 4 | — |

## Actionable gaps grouped by impact

### High-leverage, low-risk (landed in Phases 72-75)

- **CI guardrail for Python/PyO3 reintroduction** (#21): landed in
  Phase 72 via `scripts/check_no_python_legacy.sh` and CI wiring.
- **`allow(dead_code)` audit** (#6): landed in Phase 73 via
  `dead_code_allowlist_2026-05-10.md`; the stale `SessionAccum`
  suppression was removed.
- **Indicator registry metadata** (#12): landed in Phases 74-75 via
  `forex-data::core::feature_registry` and `FeatureFrame` registry
  validation helpers.

### Medium-leverage, medium-risk

- **Typed model runtime propagation** (#10, #17, #19): landed in
  Phases 76-79 for `PredictionMetadata`, NEAT, CRFMNES /
  neuro-evolution, statistical linear artifacts, CUDA linear fit
  results, and swarm forecast results.
- **Training-model artifact producer** (#31): landed in Phase 86 through
  the staged `training_orchestrator` persistence path.
- **Streaming / RL runtime metadata** (#13). The remaining model gaps are
  RL exit-agent runtime/device routing, streaming/adaptive runtime
  metadata, and the ONNX legacy boundary.

### Larger / deferred

- **UI hardware/runtime panel** (#18, P2-1). Out-of-scope for the
  contract layer; needs the egui side to consume the scheduler-owned
  plans.
- **Indicator-level CPU/GPU parity tests** (#16). Phase 4 covered the
  population evaluator; statistical / evolution backends need their
  own parity fixtures.
- **Large-file split deeper than Phase 6** (#3). Training orchestrator,
  ensemble.rs, exit_agent.rs are still large. Splitting risks behavior
  drift; defer until contract tests cover the affected surfaces.

## Recommendation

The contract / validation / extraction work that occupied Phases 1-70
is complete. The gaps above split into three buckets:

1. **Quick wins (Phases 72-75)**: landed. CI guardrail,
   `allow(dead_code)` audit, indicator registry metadata, and registry
   validation surface are now in the follow-on log.
2. **Typed propagation (Phases 76-79)**: landed for the model result
   surfaces listed above. Remaining model work should now focus on RL /
   streaming routing and artifact producers rather than another broad
   string-to-enum sweep.
3. **Deferred infrastructure (no phase)**: UI exposure, parity tests
   for statistical/evolution backends, model-runtime bridge wiring,
   ONNX legacy boundary, and large-file splits. Each needs its own design
   pass before landing — out of scope for a routine follow-on slice.

The next concrete slice should be either the model-runtime artifact bridge
or the ONNX legacy-boundary cleanup, unless the larger deferred infrastructure
gets its own design pass first.
