use core::fmt;

use serde::{Deserialize, Serialize};

use crate::runtime::capabilities::{CapabilityState, ModelFamily};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionMetadata {
    pub model_name: String,
    pub family: ModelFamily,
    pub state: CapabilityState,
}

impl PredictionMetadata {
    pub fn new(model_name: impl Into<String>, family: ModelFamily, state: CapabilityState) -> Self {
        Self {
            model_name: model_name.into(),
            family,
            state,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuntimePredictionError {
    InvalidClassProbability { index: usize, value: f32 },
    InvalidConfidence { value: f32 },
}

impl fmt::Display for RuntimePredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidClassProbability { index, value } => {
                write!(f, "invalid class probability at index {index}: {value}")
            }
            Self::InvalidConfidence { value } => {
                write!(f, "invalid confidence: {value}")
            }
        }
    }
}

impl std::error::Error for RuntimePredictionError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimePrediction {
    class_probabilities: [f32; 3],
    confidence: Option<f32>,
    abstain_recommended: Option<bool>,
    metadata: PredictionMetadata,
}

impl RuntimePrediction {
    pub fn try_new(
        class_probabilities: [f32; 3],
        confidence: Option<f32>,
        abstain_recommended: Option<bool>,
        metadata: PredictionMetadata,
    ) -> Result<Self, RuntimePredictionError> {
        Self::validate_probabilities(&class_probabilities)?;
        Self::validate_optional_probability(confidence)?;

        Ok(Self {
            class_probabilities,
            confidence,
            abstain_recommended,
            metadata,
        })
    }

    pub fn class_probabilities(&self) -> [f32; 3] {
        self.class_probabilities
    }

    pub fn confidence(&self) -> Option<f32> {
        self.confidence
    }

    pub fn abstain_recommended(&self) -> Option<bool> {
        self.abstain_recommended
    }

    pub fn metadata(&self) -> &PredictionMetadata {
        &self.metadata
    }

    pub fn parts(&self) -> ([f32; 3], Option<f32>, Option<bool>, &PredictionMetadata) {
        (
            self.class_probabilities,
            self.confidence,
            self.abstain_recommended,
            &self.metadata,
        )
    }

    fn validate_probabilities(
        class_probabilities: &[f32; 3],
    ) -> Result<(), RuntimePredictionError> {
        for (index, value) in class_probabilities.iter().copied().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(RuntimePredictionError::InvalidClassProbability { index, value });
            }
        }

        Ok(())
    }

    fn validate_optional_probability(value: Option<f32>) -> Result<(), RuntimePredictionError> {
        if let Some(value) = value {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(RuntimePredictionError::InvalidConfidence { value });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_prediction_try_new_accepts_valid_three_class_probabilities() {
        let prediction = RuntimePrediction::try_new(
            [0.1, 0.7, 0.2],
            Some(0.7),
            Some(false),
            PredictionMetadata::new("lightgbm", ModelFamily::Tree, CapabilityState::Implemented),
        )
        .expect("valid prediction");

        assert_eq!(prediction.class_probabilities(), [0.1, 0.7, 0.2]);
        assert_eq!(prediction.confidence(), Some(0.7));
        assert_eq!(prediction.abstain_recommended(), Some(false));
    }

    #[test]
    fn runtime_prediction_try_new_rejects_nan_probabilities() {
        let error = RuntimePrediction::try_new(
            [0.1, f32::NAN, 0.2],
            Some(0.7),
            Some(false),
            PredictionMetadata::new("lightgbm", ModelFamily::Tree, CapabilityState::Implemented),
        )
        .expect_err("NaN probability should be rejected");

        match error {
            RuntimePredictionError::InvalidClassProbability { index, value } => {
                assert_eq!(index, 1);
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn runtime_prediction_try_new_rejects_probabilities_outside_unit_interval() {
        let error = RuntimePrediction::try_new(
            [0.1, 1.2, 0.2],
            Some(0.7),
            Some(false),
            PredictionMetadata::new("lightgbm", ModelFamily::Tree, CapabilityState::Implemented),
        )
        .expect_err("out of range probability should be rejected");

        match error {
            RuntimePredictionError::InvalidClassProbability { index, value } => {
                assert_eq!(index, 1);
                assert_eq!(value, 1.2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn runtime_prediction_serde_round_trips_with_valid_three_class_shape() {
        let prediction = RuntimePrediction::try_new(
            [0.1, 0.7, 0.2],
            Some(0.7),
            Some(false),
            PredictionMetadata::new("lightgbm", ModelFamily::Tree, CapabilityState::Implemented),
        )
        .expect("valid prediction");

        let json = serde_json::to_string(&prediction).expect("serialize prediction");
        let decoded: RuntimePrediction =
            serde_json::from_str(&json).expect("deserialize prediction");

        assert_eq!(decoded, prediction);
    }

    #[test]
    fn runtime_prediction_parts_expose_contract_fields() {
        let prediction = RuntimePrediction::try_new(
            [0.25, 0.5, 0.25],
            Some(0.5),
            Some(true),
            PredictionMetadata::new("lightgbm", ModelFamily::Tree, CapabilityState::Implemented),
        )
        .expect("valid prediction");

        let (probs, confidence, abstain, metadata) = prediction.parts();
        assert_eq!(probs, [0.25, 0.5, 0.25]);
        assert_eq!(confidence, Some(0.5));
        assert_eq!(abstain, Some(true));
        assert_eq!(metadata.model_name, "lightgbm");
        assert_eq!(metadata.family, ModelFamily::Tree);
        assert_eq!(metadata.state, CapabilityState::Implemented);
    }
}
