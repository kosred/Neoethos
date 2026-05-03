# Model Runtime / Backend / Artifact Fragmentation Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: deep/tree/RL/statistical/evolution model runtime backend provenance, device policy duplication, artifact write/validation boilerplate, fallback runtime contracts.

## Summary

This pass continues the duplicate-logic audit beyond the search engine and custom CUDA kernels.

The model stack has several strong subsystem-specific implementations, but many concepts are repeated independently:

- requested device policy
- effective device policy
- execution backend
- runtime degraded reason
- training precision
- metadata sidecar validation
- fallback artifact
- atomic artifact writing
- feature column validation
- train/validation row validation

The issue is not that each subsystem is bad. The issue is that each subsystem has its own version of the same runtime/artifact concepts.

## Key findings

### 1. Deep/Burn models are closer to the desired pattern, but only locally

File:

- `crates/forex-models/src/deep_models.rs`

The deep model stack persists and validates:

- requested device policy
- effective device policy
- execution backend
- training precision
- Burn training report
- runtime metadata

This is good.

However, it is Burn/deep-specific. It does not unify with:

- custom CubeCL CUDA kernels
- statistical CUDA kernels
- NEAT CUDA kernels
- CRFMNES CUDA kernels
- RL/candle device policy
- core `HardwareExecutionPlan`

**Conclusion:** use the deep/Burn runtime provenance approach as inspiration, but move the concept into shared runtime types.

---

### 2. RL has its own backend/device/precision contract

File:

- `crates/forex-models/src/rl/dqn_impl.rs`

RL defines its own fields and helpers:

- `requested_backend`
- `requested_device_policy`
- `effective_backend`
- `effective_device_policy`
- `network_precision`
- `backend`
- `device_policy`
- precision resolution
- CUDA capability probing
- device policy normalization
- runtime metadata reconstruction
- fallback Q artifacts

This is a substantial parallel runtime contract.

**Risk:** RL can accept/persist backend and precision semantics differently from deep/statistical/evolution/search systems.

**Cleanup direction:** introduce shared `RuntimeBackendProvenance` and `PrecisionContract`.

---

### 3. Tree models have a separate fallback artifact ecosystem

Files:

- `crates/forex-models/src/tree_models/common.rs`
- `crates/forex-models/src/tree_models/lightgbm.rs`
- `crates/forex-models/src/tree_models/xgboost.rs`
- `crates/forex-models/src/tree_models/catboost.rs`
- `crates/forex-models/src/tree_models/sklears.rs`

Tree models have useful shared helpers in `tree_models/common.rs`, including:

- local fallback artifact
- probability normalization/calibration
- runtime backend details
- metadata read/write
- atomic write
- feature-column checks

This is good within tree models.

But it is not shared with statistical/evolution/deep/RL models.

**Cleanup direction:** keep tree-specific fallback math, but move general artifact/provenance/writer concepts to shared runtime layer.

---

### 4. Statistical models use another runtime backend fallback pattern

Files:

- `crates/forex-models/src/statistical/linear_impl.rs`
- `crates/forex-models/src/statistical/linear_gpu.rs`
- `crates/forex-models/src/statistical/common.rs`

Statistical models use:

- `runtime_backend_with_gpu_fallback`
- CUDA kernel enabled checks
- embedded runtime metadata
- metadata sidecar resolution
- runtime degraded reason

This is again similar to tree/deep/RL concepts, but not fully unified.

---

### 5. Evolution models duplicate artifact validation and runtime provenance

Files:

- `crates/forex-models/src/evolution/crfmnes_impl.rs`
- `crates/forex-models/src/evolution/neat_impl.rs`

Evolution models validate:

- model name
- family
- state
- feature columns
- label mapping
- training summary
- scaler dimensions
- fitted state
- backend/degraded reason

This is necessary, but much of the boilerplate should be shared.

---

### 6. Atomic artifact writing is not globally unified

Search results show `atomic_write` in tree-model common and per-tree usage. Other model families have their own staged/backup/rename helpers.

Examples:

- linear model artifact staging
- RL staged temp/backup files
- tree model atomic writes
- deep model record/config/metadata writes
- training profile atomic writes

**Risk:** some artifact families may be more crash-safe than others.

**Cleanup direction:** one shared `AtomicArtifactWriter` with directory/file modes.

---

### 7. Feature-column validation exists many times

Every family validates feature columns in its own way:

- tree `ensure_feature_columns_match`
- statistical common `ensure_feature_columns_match`
- deep model metadata checks
- RL metadata checks
- evolution artifact checks

**Risk:** strictness and error behavior differ across model families.

**Cleanup direction:** shared `FeatureSchemaContract` and `validate_feature_schema_compatibility`.

---

### 8. Runtime degraded reason is repeated but not standardized

Different subsystems store degraded reason differently:

- runtime degraded reason
- fallback reason
- backend unavailable reason
- precision degraded reason
- runtime details reason

**Risk:** UI/live/runtime cannot reliably interpret whether a model is degraded, approximate, fallback, CPU-only, or unsupported.

**Cleanup direction:** introduce:

```rust
pub enum RuntimeDegradationKind {
    BackendUnavailable,
    PrecisionUnsupported,
    FallbackModelUsed,
    MetadataMissing,
    FeatureSchemaMismatch,
    UnsupportedContract,
}
```

and a structured `RuntimeDegradation` object.

## Required shared types

### Runtime backend provenance

```rust
pub struct RuntimeBackendProvenance {
    pub requested_backend: Option<String>,
    pub requested_device_policy: String,
    pub effective_backend: String,
    pub effective_device_policy: String,
    pub device_ids: Vec<usize>,
    pub requested_precision: Option<String>,
    pub effective_precision: String,
    pub fallback_used: bool,
    pub degraded_reasons: Vec<RuntimeDegradation>,
}
```

### Precision contract

```rust
pub struct PrecisionContract {
    pub requested: TrainingPrecisionRequest,
    pub effective: TrainingPrecision,
    pub reason: Option<String>,
    pub supported_by_all_assigned_devices: bool,
}
```

### Artifact write contract

```rust
pub struct AtomicArtifactWriter {
    pub mode: ArtifactWriteMode,
    pub fsync: bool,
    pub backup_existing: bool,
    pub verify_after_write: bool,
}
```

### Feature schema contract

```rust
pub struct FeatureSchemaContract {
    pub feature_columns: Vec<String>,
    pub schema_hash: String,
    pub timestamp_unit: Option<String>,
    pub availability_hash: Option<String>,
}
```

## What to preserve

Do not delete subsystem-specific math or model logic:

- tree fallback probability logic
- RL reward/episode/Q fallback logic
- Burn device/runtime model init logic
- statistical CUDA kernels
- NEAT/CRFMNES CUDA kernels

These are valuable.

## What to extract

Extract generic repeated concepts:

- runtime backend provenance
- precision selection/validation
- artifact metadata validation
- feature schema validation
- atomic artifact writing
- degraded reason representation
- train/validation split contract

## Recommended implementation order

1. Add shared `RuntimeBackendProvenance` in `forex-models::runtime` or `forex-core`.
2. Add shared `RuntimeDegradation` enum/object.
3. Add shared `PrecisionContract`.
4. Add shared `AtomicArtifactWriter`.
5. Add shared `FeatureSchemaContract`.
6. Update deep/Burn model to emit shared provenance first because it already has similar fields.
7. Update RL artifact to use shared provenance while preserving old fields with serde defaults for compatibility.
8. Update statistical/evolution artifacts.
9. Update tree model fallback artifact metadata.
10. Only then remove old duplicate helpers.

## Bottom line

The model runtime layer has repeated versions of the same ideas. The right cleanup is not to delete model-specific implementations. The right cleanup is to move runtime provenance, precision policy, artifact validation, feature-schema validation, and atomic writing into shared contracts that every model family uses.
