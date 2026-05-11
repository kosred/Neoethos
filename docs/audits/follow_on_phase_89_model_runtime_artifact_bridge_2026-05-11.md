# Follow-on Phase 89 - model-runtime artifact bridge

Date: 2026-05-11

## Source gap

The `model_runtime_backend_fragmentation` audit still had one open contract
gap after Phases 76-85: `forex-models` propagated typed runtime metadata into
model results, but the training persistence bridge did not emit a typed
`ModelRuntimeArtifactContract` envelope.

## Changes

- Added a regression test proving `persist_training_artifacts` writes
  `model_runtime_artifact.json` beside the training runtime profile.
- Added `MODEL_RUNTIME_ARTIFACT_FILE_NAME` and a shared
  `write_model_runtime_artifact` helper on the existing runtime profile JSON
  writer surface.
- Added `write_model_runtime_artifact_contract_sidecar` so training model saves
  now emit a typed `ModelRuntimeArtifact<TrainingRuntimeProfile>` envelope.
- Deduplicated training/model-runtime artifact provenance construction through
  one shared builder that varies only the `ArtifactKind` and envelope type.

## Verification

- RED: `cargo test -p forex-models --lib persist_training_artifacts_writes_model_runtime_artifact_contract -- --nocapture`
- GREEN: `cargo test -p forex-models --lib persist_training_artifacts_writes_model_runtime_artifact_contract -- --nocapture`
- `cargo test -p forex-models --lib -- --test-threads=1`
- `cargo fmt --check`
- `git diff --check`
