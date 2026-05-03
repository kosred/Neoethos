# Training / Model Runtime / Artifact Contract Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: training orchestrator, model runtime profile, runtime artifact metadata, genetic model artifact, feature/label contract, determinism, saved model reproducibility.

## Summary

This audit moves beyond strategy search and inspects the next stage: model training and model artifacts.

The training stack is more structured than the raw discovery/export stack. It already has:

- `TrainingRuntimeProfile`
- `RuntimeArtifactMetadata`
- atomic profile writes
- capability family/state metadata
- feature column lists
- label mapping
- train/validation row summaries
- backend/device/precision planning
- HPO metadata
- embargo metadata

However, the model artifact contract is still not strong enough to guarantee exact reproducibility or safe runtime loading across feature schema changes, timestamp-unit changes, hardware backends, and genetic/search origin differences.

## Positive findings

### 1. Training runtime profile is real and validated

`TrainingRuntimeProfile` records many useful fields:

- model name
- capability family/state
- symbol
- base timeframe
- feature count
- dataset rows
- row budget
- label horizon
- effective label horizon
- triple barrier setting
- higher timeframes
- multi-resolution settings
- base signal filter
- L1 feature selection flag
- requested/planned backend/device/precision
- checkpoint path
- train/val years
- HPO backend/trials
- holdout percentage
- embargo minutes
- ONNX/RLlib/DDP/FSDP flags
- notes

It also has validation and atomic write logic.

**Assessment:** good foundation.

---

### 2. L1 feature selection appears train-split constrained

In `training_orchestrator.rs`, `apply_feature_selection` uses the first 80% of rows for feature selection before applying selected columns to the full frame.

This is a meaningful improvement over fitting feature selection on all rows.

**Assessment:** positive; keep and expand this contract.

---

### 3. Runtime prediction contract is stricter than search metrics

The runtime model side has a structured prediction contract:

- three-class probability layout
- confidence calculation
- abstain recommendation
- model metadata
- runtime details / degraded reason

**Assessment:** good direction.

## Findings / gaps

### 1. Runtime artifact metadata lacks feature schema hash

`RuntimeArtifactMetadata` stores `feature_columns`, but not:

- feature schema hash
- timestamp unit
- feature availability contract
- source timeframe per feature
- feature generator version
- feature ordering hash
- selected-feature mapping from raw schema to trained schema

**Risk:** a model can be loaded with same column names but different generation semantics, timestamp unit, or MTF availability.

**Severity:** Critical.

**Fix direction:** add `FeatureSchemaMetadata` to runtime artifacts.

---

### 2. Runtime artifact metadata lacks dataset fingerprint

The metadata stores row counts but not a dataset identity:

- dataset root/hash
- symbol dataset fingerprint
- timeframe list
- data range start/end
- data revision
- duplicate timestamp policy
- normalization/version info

**Risk:** two models trained on different data can look equivalent at runtime.

**Severity:** High.

**Fix direction:** add `DatasetFingerprint`.

---

### 3. Runtime artifact metadata lacks execution/training config hash

Training profile records many config values, but model artifact metadata does not contain a compact immutable hash of the effective training config.

**Risk:** model artifacts cannot be compared or rejected based on config drift.

**Severity:** High.

**Fix direction:** add `training_config_hash` and `runtime_contract_hash`.

---

### 4. `TrainingRuntimeProfile` records feature count but not actual feature names/schema

The training profile records `feature_count`, not the actual feature column list or schema hash.

**Risk:** profile alone cannot prove which exact features were used.

**Severity:** Medium-High.

**Fix direction:** profile should include at least `feature_schema_hash`, and optionally `feature_columns` or a pointer to a schema artifact.

---

### 5. Genetic `LabelSearch` still lacks first-class seed

`GeneticStrategyExpert::train_with_labels` wires RNG through helper functions, which is good. But the RNG itself is created from `rand::rng()`.

There is no first-class `genetic_seed`/`search_seed` in artifact metadata.

**Risk:** repeated training can produce different `genetic_portfolio.json` for same data/config.

**Severity:** High.

**Fix direction:** add typed seed to `GeneticArtifact` and model settings, and initialize `StdRng::seed_from_u64(seed)`.

---

### 6. Genetic artifact mixes score semantics

`GeneticArtifact` can represent:

- `DiscoveryBacked`
- `LabelSearch`

Both contain `Gene` objects, but their metrics do not mean the same thing. In label search, fields like `sharpe_ratio`, `win_rate`, `profit_factor`, `max_drawdown`, and `expectancy` are derived from label classification quality, not trade simulation.

**Risk:** downstream code may treat label-search genes as if they were backtest genes.

**Severity:** Critical.

**Fix direction:** add explicit score semantics:

```rust
pub enum GeneScoreSemantics {
    DiscoveryBacktest,
    LabelClassification,
    ApproxGpuPresearch,
    ImportedWarmStart,
}
```

Every artifact must carry this.

---

### 7. Model base still contains env-driven runtime behavior

Examples:

- `FOREX_BOT_EARLY_STOP_PATIENCE`
- `FOREX_BOT_EARLY_STOP_MIN_DELTA`
- `FOREX_BOT_DRIFT_THRESHOLD`

**Risk:** training behavior and drift behavior can change outside typed config.

**Severity:** Medium-High.

**Fix direction:** move these into typed model/training config and export them in profile/artifact metadata.

---

### 8. Feature time-order validation warns instead of failing when no time column exists

`validate_time_ordering` returns OK when no timestamp/time/date/datetime column exists, with only a warning.

**Risk:** models can train on data assumed sorted without a hard proof.

**Severity:** Medium-High.

**Fix direction:** for production training, require explicit timestamp vector or a `TimeOrderingAssumption::PreSortedVerified` contract.

---

### 9. Training profile checkpoint path is metadata, not proven resume system

`TrainingRuntimeProfile` has `checkpoint_path`, but the profile itself does not prove:

- checkpoint format
- save frequency
- resume validation
- optimizer state
- RNG state
- feature schema compatibility
- dataset/config hash compatibility

**Risk:** user may believe training resume exists when only checkpoint metadata exists.

**Severity:** Medium-High.

**Fix direction:** define `TrainingCheckpointManifest`.

---

## Required artifact contract

Add:

```rust
pub struct ModelRuntimeContract {
    pub artifact_version: u32,
    pub model_name: String,
    pub model_family: ModelFamily,
    pub symbol: String,
    pub base_timeframe: String,
    pub dataset_fingerprint: DatasetFingerprint,
    pub feature_schema: FeatureSchemaMetadata,
    pub label_contract: LabelContract,
    pub train_split: SplitContract,
    pub validation_split: SplitContract,
    pub training_config_hash: String,
    pub runtime_contract_hash: String,
    pub requested_backend: Option<String>,
    pub actual_backend: Option<String>,
    pub requested_precision: Option<String>,
    pub actual_precision: Option<String>,
    pub seed: Option<u64>,
    pub score_semantics: Option<GeneScoreSemantics>,
}
```

## Required `FeatureSchemaMetadata`

```rust
pub struct FeatureSchemaMetadata {
    pub feature_columns: Vec<String>,
    pub schema_hash: String,
    pub timestamp_unit: TimestampUnit,
    pub source_timeframes: Vec<String>,
    pub availability_contract_hash: String,
    pub generator_versions: Vec<String>,
}
```

## Required `DatasetFingerprint`

```rust
pub struct DatasetFingerprint {
    pub symbol: String,
    pub timeframes: Vec<String>,
    pub rows_by_timeframe: Vec<(String, usize)>,
    pub start_ts: Option<i64>,
    pub end_ts: Option<i64>,
    pub timestamp_unit: TimestampUnit,
    pub content_hash: Option<String>,
}
```

## Recommended implementation order

1. Extend `RuntimeArtifactMetadata` with optional `feature_schema_hash` and `dataset_fingerprint` fields using serde defaults.
2. Extend `TrainingRuntimeProfile` with `feature_schema_hash` and maybe `feature_columns`.
3. Add `GeneScoreSemantics` to `GeneticArtifact`.
4. Add `seed` to `GeneticArtifact` and initialize label-search RNG deterministically.
5. Move early-stop and drift-threshold env vars into typed settings.
6. Add strict time-ordering mode for training.
7. Add `TrainingCheckpointManifest` before claiming resumable training.
8. Add artifact load validation: feature schema hash, label mapping, model family, and runtime contract must match before prediction.

## Required tests

1. `runtime_artifact_rejects_missing_feature_schema_hash_in_strict_mode`
2. `training_profile_records_feature_schema_hash`
3. `genetic_label_search_same_seed_same_portfolio`
4. `label_search_gene_score_semantics_not_backtest_semantics`
5. `model_load_rejects_feature_schema_mismatch`
6. `training_time_ordering_requires_timestamp_in_strict_mode`
7. `training_checkpoint_manifest_rejects_config_hash_mismatch`

## Bottom line

The training/model side is in better shape than the raw search export layer, but it still needs stronger artifact identity. A trained model must carry enough metadata to prove: what data, what features, what labels, what split, what backend, what seed, and what score semantics produced it.
