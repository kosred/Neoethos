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

    #[cfg(test)]
    pub(crate) fn raw_for_validation(
        dataset_rows: usize,
        train_rows: usize,
        val_rows: usize,
    ) -> Self {
        Self {
            dataset_rows,
            train_rows,
            val_rows,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeArtifactMetadata {
    pub model_name: String,
    pub family: ModelFamily,
    pub state: CapabilityState,
    pub feature_columns: Vec<String>,
    pub label_mapping: Vec<LabelMapping>,
    pub training_summary: TrainingSummaryMetadata,
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
