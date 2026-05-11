# Follow-on Phase 87 - repo JSON artifact IO dedup

Date: 2026-05-11

## Source gap

Phase 86 added another JSON artifact writer in `forex-models`, while older
modules still carried local copies of the same temp/backup/read/hash patterns.
The user requested that the cleanup apply repo-wide, not only to
`forex-models`.

## Changes

- Added `forex_core::storage::json`, the shared owner for:
  - `write_json_atomic`
  - `write_json_with_backup`
  - `read_json`
  - `temporary_path`
  - `stable_json_hash`
- Replaced `forex-search::artifact_io` with a thin compatibility re-export
  over the shared core helper.
- Moved `forex-models::runtime` JSON writers for training profiles,
  training-model contract envelopes, ONNX export status, and optimization
  reports onto `forex_core::storage::json::write_json_with_backup`.
- Moved common model metadata/artifact helpers onto the shared writer/reader:
  statistical models, deep models, tree runtime metadata, meta-model artifacts,
  and genetic artifacts/runtime metadata.
- Moved tree-model JSON artifacts onto common tree JSON helpers backed by the
  core writer/reader: XGBoost runtime/local fallback, LightGBM runtime/local
  fallback, CatBoost runtime/local fallback, and Sklears model/runtime JSON.
- Moved swarm forecaster JSON save/load onto the core writer/reader, removing
  its local temp/backup writer.
- Moved the Phase 86 training-model contract hash calls onto the shared
  `stable_json_hash`.

## Deliberate non-changes

- Binary/raw file writers remain local (`vortex` buffers, model binaries,
  build-script generated files, sectioned logs).
- Test-only corruption fixtures still use direct writes when the purpose is to
  create malformed artifacts.
- The tree module's byte-level `atomic_write` remains only for native/binary
  tree model artifacts, not JSON sidecars.

## Verification

- `cargo test -p forex-core storage::json -- --nocapture`
- `cargo test -p forex-core --lib` (`70 passed; 0 failed`)
- `cargo test -p forex-search --lib`
- `cargo test -p forex-models --lib -- --test-threads=1` (`336 passed; 0 failed`)
