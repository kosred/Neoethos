use core::fmt;

use serde::{Deserialize, Serialize};

use neoethos_core::{BackendKind, RuntimeDegradedReason, RuntimeMode};

use crate::runtime::capabilities::{
    CapabilityState, ModelFamily, runtime_backend_kind_from_label, runtime_mode_from_details,
    typed_runtime_degraded_reason,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionMetadata {
    pub model_name: String,
    pub family: ModelFamily,
    pub state: CapabilityState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_kind: Option<BackendKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_mode: Option<RuntimeMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_degraded_reason: Option<RuntimeDegradedReason>,
}

impl PredictionMetadata {
    pub fn new(model_name: impl Into<String>, family: ModelFamily, state: CapabilityState) -> Self {
        let model_name = model_name.into();
        assert!(
            !model_name.trim().is_empty(),
            "runtime prediction metadata requires a non-empty model_name"
        );
        Self {
            model_name,
            family,
            state,
            execution_backend: None,
            degraded_reason: None,
            backend_kind: None,
            runtime_mode: None,
            runtime_degraded_reason: None,
        }
    }

    pub fn with_runtime_details(
        mut self,
        execution_backend: Option<String>,
        degraded_reason: Option<String>,
    ) -> Self {
        self.execution_backend = execution_backend.filter(|value| !value.trim().is_empty());
        self.degraded_reason = degraded_reason.filter(|value| !value.trim().is_empty());
        self.backend_kind = runtime_backend_kind_from_label(self.execution_backend.as_deref());
        self.runtime_mode =
            runtime_mode_from_details(self.backend_kind, self.degraded_reason.as_deref());
        self.runtime_degraded_reason =
            typed_runtime_degraded_reason(self.degraded_reason.as_deref());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuntimePredictionError {
    InvalidClassProbability { index: usize, value: f32 },
    InvalidProbabilitySum { sum: f32 },
    InvalidConfidence { value: f32 },
}

impl fmt::Display for RuntimePredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidClassProbability { index, value } => {
                write!(f, "invalid class probability at index {index}: {value}")
            }
            Self::InvalidProbabilitySum { sum } => {
                write!(f, "invalid class probability sum: expected ~1.0, got {sum}")
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
        let mut sum = 0.0_f32;
        for (index, value) in class_probabilities.iter().copied().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(RuntimePredictionError::InvalidClassProbability { index, value });
            }
            sum += value;
        }

        if !sum.is_finite() || (sum - 1.0).abs() > 1e-3 {
            return Err(RuntimePredictionError::InvalidProbabilitySum { sum });
        }

        Ok(())
    }

    fn validate_optional_probability(value: Option<f32>) -> Result<(), RuntimePredictionError> {
        if let Some(value) = value
            && (!value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(RuntimePredictionError::InvalidConfidence { value });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_core::{BackendKind, RuntimeMode};

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
    fn runtime_prediction_try_new_rejects_rows_that_do_not_sum_to_one() {
        let error = RuntimePrediction::try_new(
            [0.1, 0.7, 0.1],
            Some(0.7),
            Some(false),
            PredictionMetadata::new("lightgbm", ModelFamily::Tree, CapabilityState::Implemented),
        )
        .expect_err("probability rows must sum to one");

        match error {
            RuntimePredictionError::InvalidProbabilitySum { sum } => {
                assert!((sum - 0.9).abs() < 1e-6);
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

    #[test]
    fn prediction_metadata_can_carry_runtime_details() {
        let metadata =
            PredictionMetadata::new("lightgbm", ModelFamily::Tree, CapabilityState::Implemented)
                .with_runtime_details(
                    Some("lightgbm_native".to_string()),
                    Some("native booster unavailable".to_string()),
                );

        assert_eq!(
            metadata.execution_backend.as_deref(),
            Some("lightgbm_native")
        );
        assert_eq!(
            metadata.degraded_reason.as_deref(),
            Some("native booster unavailable")
        );
    }

    #[test]
    fn prediction_metadata_attaches_typed_runtime_contract() {
        let canonical =
            PredictionMetadata::new("neat", ModelFamily::Evolutionary, CapabilityState::Verified)
                .with_runtime_details(Some("symbios_neat_cpu".to_string()), None);

        assert_eq!(canonical.backend_kind, Some(BackendKind::NativeCpu));
        assert_eq!(canonical.runtime_mode, Some(RuntimeMode::Canonical));
        assert_eq!(canonical.runtime_degraded_reason, None);

        let degraded = PredictionMetadata::new(
            "neuro_evo",
            ModelFamily::Evolutionary,
            CapabilityState::Implemented,
        )
        .with_runtime_details(
            Some("simple_es_restart_cpu".to_string()),
            Some("crfmnes_backend_degraded_to_simple_es".to_string()),
        );

        assert_eq!(
            degraded.backend_kind,
            Some(BackendKind::LocalSurrogateFallback)
        );
        assert_eq!(degraded.runtime_mode, Some(RuntimeMode::Degraded));
        assert_eq!(
            degraded
                .runtime_degraded_reason
                .as_ref()
                .map(|reason| reason.code.as_str()),
            Some("crfmnes_backend_degraded_to_simple_es")
        );
    }

    #[test]
    #[should_panic(expected = "runtime prediction metadata requires a non-empty model_name")]
    fn prediction_metadata_rejects_blank_model_name() {
        let _ = PredictionMetadata::new("   ", ModelFamily::Tree, CapabilityState::Implemented);
    }
}
