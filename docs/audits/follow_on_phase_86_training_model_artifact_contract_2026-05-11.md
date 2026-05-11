# Follow-on Phase 86 - training-model artifact contract

Date: 2026-05-11

## Source gap

The Phase 71 coverage matrix left `training_model_artifact_contract` open:
`forex-core` defined `TrainingModelArtifactContract`, but no `forex-models`
producer emitted a typed `TrainingModelArtifact` envelope when a trained model
was saved.

## Changes

- Added `training_model_artifact.json` as the training-model contract sidecar
  beside `training_profile.json` inside the staged training artifact directory.
- Isolated the provenance/envelope builder in `forex-models::runtime::training_artifact`
  so the large training orchestrator only owns persistence ordering.
- Wired `training_orchestrator::persist_training_artifacts` so every successful
  model save writes the typed contract envelope before the staged directory is
  promoted.
- Reused the existing `TrainingRuntimeProfile` as the envelope payload, keeping
  the contract payload aligned with the runtime profile already produced for
  operators.
- Populated `ArtifactProvenance` with deterministic FNV-1a hashes for feature
  schema, dataset fingerprint, symbol/timeframe scope, timestamp policy,
  feature availability, label policy, training config, runtime config, search
  not-applicable scope, and risk config.
- Mapped training runtime hints into typed `BackendKind`, `RuntimeMode`,
  `DeviceAssignment`, degraded-reason metadata where needed, hardware profile
  identity, and source commit.
- Added regression coverage proving the staged training persistence path writes
  a deserializable `TrainingModelArtifact<TrainingRuntimeProfile>` with
  `ArtifactKind::TrainingModel` provenance.

## Verification

- `cargo test -p forex-models --lib persist_training_artifacts_writes_training_model_artifact_contract -- --nocapture`
- `cargo test -p forex-models --lib -- --test-threads=1` (`336 passed; 0 failed`)
