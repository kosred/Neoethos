# Follow-on Phases 80-85 - Model contract cleanup

Date: 2026-05-10

Scope: close the stale `forex-models` contract failures left after the typed
runtime metadata slice, without weakening the stricter artifact contracts added
in earlier phases.

## Phase map

- **Phase 80 - invalid metadata fixtures and sidecar context**: added a
  test-only raw `TrainingSummaryMetadata` constructor for deliberately corrupt
  validation fixtures, and wrapped sidecar-validation failures so drift tests
  report the sidecar mismatch boundary instead of a lower-level field error.
- **Phase 81 - stale registry/export/pruning expectations**: aligned capability
  assertions with the verified model registry, fixed the ONNX test request flag,
  and changed swarm pruning to prefer non-duplicate candidates before using
  duplicates as a minimum-count backfill.
- **Phase 82 - Burn/deep runtime contract**: allowed `external_device` as
  externally supplied Burn provenance, made Burn reports account for the actual
  embargo rows used by the split, derived deep training summaries without
  violating `train_rows + val_rows == dataset_rows`, and required the persisted
  runtime triplet before deep inference/persistence.
- **Phase 83 - exit-agent artifact contract**: updated trained fixtures to
  persist `training_report`, validated partial runtime identity before report
  cross-checks, and preserved the missing-report inference guard for trained
  runtime state.
- **Phase 84 - streaming runtime detail truthfulness**: made Hoeffding runtime
  details distinguish fallback-only-by-weight from unavailable live committees,
  and aligned tests with the existing rule that persisted committee JSON plus a
  readiness flag is not enough to claim live committee inference.
- **Phase 85 - RL fallback/runtime contract**: avoided `.tmp` self-collision
  with the shared JSON writer by using `.staged` files, treated normalized
  `gpu:<idx>` policies as CUDA-capable for RL precision resolution, and cleaned
  RL fallback/report fixtures so each test reaches the contract it is asserting.

## Verification

- `cargo test -p forex-models --lib try_build_runtime_artifact_metadata`
- `cargo test -p forex-models --lib sidecar_drift`
- `cargo test -p forex-models --lib burn_models::tests:: -- --test-threads=1`
- `cargo test -p forex-models --lib deep_models::tests:: -- --test-threads=1`
- `cargo test -p forex-models --lib exit_agent::tests:: -- --test-threads=1`
- `cargo test -p forex-models --lib streaming::adaptive_impl::tests::online_hoeffding_runtime_details -- --test-threads=1`
- `cargo test -p forex-models --lib rl::dqn_impl::tests:: -- --test-threads=1`
- `cargo test -p forex-models --lib -- --test-threads=1`

Full-suite result after Phase 85: `335 passed; 0 failed`.
