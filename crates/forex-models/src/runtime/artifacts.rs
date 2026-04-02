use serde::{Deserialize, Serialize};

use super::capabilities::{CapabilityState, ModelFamily};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelMapping {
    pub raw_label: i32,
    pub class_index: usize,
}

impl LabelMapping {
    pub fn new(raw_label: i32, class_index: usize) -> Self {
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
        Self {
            model_name: model_name.into(),
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
