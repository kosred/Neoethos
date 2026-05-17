use serde::{Deserialize, Serialize};

use super::capabilities::{CapabilityState, ModelFamily};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelMapping {
    pub raw_label: i32,
    pub class_index: usize,
}

impl LabelMapping {
    pub fn new(raw_label: i32, class_index: usize) -> Self {
        assert!(
            class_index < 3,
            "runtime label mapping class_index must stay inside the three-class contract"
        );
        Self {
            raw_label,
            class_index,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingSummaryMetadata {
    pub dataset_rows: usize,
    pub train_rows: usize,
    pub val_rows: usize,
}

impl TrainingSummaryMetadata {
    /// Constructs a strict training-summary record. Panics on invalid
    /// row counts to keep production callers honest. Callers that need
    /// to construct a *deliberately* invalid summary (e.g. drift-
    /// detection unit tests) must use [`Self::new_unchecked`].
    pub fn new(dataset_rows: usize, train_rows: usize, val_rows: usize) -> Self {
        assert!(
            dataset_rows > 0,
            "runtime training summary requires a non-zero dataset row count"
        );
        assert!(
            train_rows > 0,
            "runtime training summary requires a non-zero train row count"
        );
        assert!(
            train_rows + val_rows == dataset_rows,
            "runtime training summary requires train_rows + val_rows == dataset_rows"
        );
        Self {
            dataset_rows,
            train_rows,
            val_rows,
        }
    }

    /// Escape hatch for callers that need to construct invalid summaries
    /// without panicking. The runtime artifact validator
    /// (`validate_runtime_metadata` in each model crate) still rejects
    /// the result on load, so production code paths cannot smuggle
    /// these in — only tests asserting that rejection.
    pub fn new_unchecked(dataset_rows: usize, train_rows: usize, val_rows: usize) -> Self {
        Self {
            dataset_rows,
            train_rows,
            val_rows,
        }
    }

    /// Test-only alias for `new_unchecked` used by intra-crate tests.
    /// Kept for backwards compatibility with the legacy call sites in
    /// `base.rs`, `deep_models.rs`, `streaming/adaptive_impl.rs`,
    /// `tree_models/sklears.rs`, and `ensemble_tests.rs`.
    #[cfg(test)]
    pub(crate) fn raw_for_validation(
        dataset_rows: usize,
        train_rows: usize,
        val_rows: usize,
    ) -> Self {
        Self::new_unchecked(dataset_rows, train_rows, val_rows)
    }
}

/// Current schema version of every per-expert artifact metadata
/// JSON written by `crate::tree_models::*::save` /
/// `crate::deep_models::*::save` / etc. Per D4 versioning policy:
/// bump only on serialised-field BREAKING changes.
///
/// v1 (current): the pre-versioning shape. New optional fields
/// stay backward-compatible via `#[serde(default)]`.
pub const RUNTIME_ARTIFACT_METADATA_SCHEMA_VERSION:
    forex_core::SchemaVersion = forex_core::SchemaVersion::new(1);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeArtifactMetadata {
    /// On-disk schema version. Defaults to v1 (the pre-versioning
    /// shape) for artifacts written by older builds, so existing
    /// trained models load without breaking.
    #[serde(default = "forex_core::default_v1")]
    pub schema_version: forex_core::SchemaVersion,
    pub model_name: String,
    pub family: ModelFamily,
    pub state: CapabilityState,
    pub feature_columns: Vec<String>,
    pub label_mapping: Vec<LabelMapping>,
    pub training_summary: TrainingSummaryMetadata,
}

impl forex_core::HasSchemaVersion for RuntimeArtifactMetadata {
    const CURRENT: forex_core::SchemaVersion = RUNTIME_ARTIFACT_METADATA_SCHEMA_VERSION;
    fn schema_version(&self) -> forex_core::SchemaVersion {
        self.schema_version
    }
}

impl RuntimeArtifactMetadata {
    pub fn new(
        model_name: impl Into<String>,
        family: ModelFamily,
        state: CapabilityState,
        feature_columns: Vec<String>,
        label_mapping: Vec<LabelMapping>,
        training_summary: TrainingSummaryMetadata,
    ) -> Self {
        let model_name = model_name.into();
        assert!(
            !model_name.trim().is_empty(),
            "runtime artifact metadata requires a non-empty model_name"
        );
        assert!(
            !feature_columns.is_empty(),
            "runtime artifact metadata requires at least one feature column"
        );
        assert!(
            !label_mapping.is_empty(),
            "runtime artifact metadata requires a non-empty label mapping"
        );
        Self {
            schema_version: RUNTIME_ARTIFACT_METADATA_SCHEMA_VERSION,
            model_name,
            family,
            state,
            feature_columns,
            label_mapping,
            training_summary,
        }
    }
}

pub fn default_three_class_label_mapping() -> Vec<LabelMapping> {
    vec![
        LabelMapping::new(-1, 2),
        LabelMapping::new(0, 0),
        LabelMapping::new(1, 1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "runtime training summary requires a non-zero train row count")]
    fn training_summary_rejects_zero_train_rows() {
        let _ = TrainingSummaryMetadata::new(10, 0, 10);
    }

    #[test]
    #[should_panic(expected = "runtime artifact metadata requires at least one feature column")]
    fn runtime_metadata_rejects_empty_feature_columns() {
        let _ = RuntimeArtifactMetadata::new(
            "mlp",
            ModelFamily::Deep,
            CapabilityState::Implemented,
            Vec::new(),
            default_three_class_label_mapping(),
            TrainingSummaryMetadata::new(10, 8, 2),
        );
    }
}
