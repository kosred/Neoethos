# Follow-on Phases 76-79 - Typed model runtime metadata

Date: 2026-05-10

Scope: close the typed runtime/backend propagation bucket from
`audit_coverage_matrix_2026-05-10.md` for the model surfaces that still carried
only string backend labels.

## Phase map

- **Phase 76 - shared typed inference helpers**: added model-runtime helpers
  that derive `BackendKind`, `RuntimeMode`, and `RuntimeDegradedReason` from
  existing backend/degraded strings, then attached those typed fields to
  `PredictionMetadata`.
- **Phase 77 - NEAT runtime metadata**: persisted typed `BackendKind` in NEAT
  artifacts and runtime predictions while preserving the legacy
  `runtime_backend` string.
- **Phase 78 - CRFMNES / neuro-evolution runtime metadata**: persisted typed
  `BackendKind` for neuro-evolution artifacts and propagated degraded local
  surrogate runtime metadata into runtime predictions.
- **Phase 79 - statistical and swarm metadata**: carried typed backend/runtime
  metadata through statistical linear artifacts, CUDA linear fit results, and
  swarm forecast results.

## Compatibility notes

Legacy string fields remain in place for artifact compatibility and operator
readability. New typed fields use serde defaults / skip-empty behavior so older
artifacts can still deserialize, while new writes include the typed contract
fields for downstream validation.

## Verification

- `cargo test -p forex-models --lib prediction_metadata_attaches_typed_runtime_contract`
- `cargo test -p forex-models --lib neat_save_records_train_val_rows_and_runtime_backend`
- `cargo test -p forex-models --lib neuro_evo_save_records_training_rows`
- `cargo test -p forex-models --lib runtime_backend_kind_from_label_maps_known_backend_families`
- `cargo test -p forex-models --lib runtime_mode_and_degraded_reason_are_typed_from_legacy_details`
- `cargo test -p forex-models --lib logistic_expert_trains_and_persists_runtime_metadata`
- `cargo test -p forex-models --lib select_external_or_fallback_result_marks_local_fallback_when_external_result_is_invalid`

Full-suite note: `cargo test -p forex-models --lib -- --test-threads=1` was
also attempted after this slice. It is not green yet: 297 passed and 38 failed,
with the remaining failures concentrated in older model-contract fixtures,
sidecar-drift expectation strings, RL / streaming routing, and broader training
metadata tests outside this typed propagation slice.

Known pre-existing warning during these tests:
`crates/forex-models/src/deep_models.rs:845` reports unused
`train_runtime_model`.
