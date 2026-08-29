// TODO(real-data): the calibrator/conformal fixtures and any
// downstream FeatureFrame inputs in this file use synthesised
// probabilities/qhat values. Replace with a cTrader historical sample:
// fit calibrators on real out-of-sample meta-model predictions for the
// target symbol/timeframe and reuse those artifacts here.
use super::*;
use crate::tree_models::XGBoostExpert;
use ndarray::array;
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};

fn fitted_temperature_calibrator() -> ProbabilityCalibrator {
    ProbabilityCalibrator {
        method: CalibrationMethod::Temperature,
        fitted: true,
        models: vec![CalibrationModel::Temperature { temperature: 1.0 }],
    }
}

fn fitted_conformal_gate(alpha: f64) -> ConformalGate {
    ConformalGate {
        alpha,
        qhat: 0.55,
        fitted: true,
        n_calib: 128,
    }
}

fn one_row_frame() -> FeatureFrame {
    let column = FeatureColumnF64::new("feature", vec![1.0], vec![FeatureCellValidity::Valid])
        .expect("valid f64 feature column");
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(1),
        vec![column],
    )
    .expect("valid canonical feature frame")
}

fn one_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(1).expect("one worker");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("test CPU lease")
}

#[test]
fn identity_calibrator_rejects_non_finite_probability_without_neutral_fallback() {
    let calibrator = ProbabilityCalibrator {
        method: CalibrationMethod::Identity,
        fitted: true,
        models: Vec::new(),
    };

    let error = calibrator
        .predict_proba(&array![[f64::NAN, 0.4, 0.6]])
        .expect_err("non-finite probability must fail closed");
    assert!(error.to_string().contains("finite"));
}

#[test]
fn identity_calibrator_rejects_zero_probability_mass_without_neutral_fallback() {
    let calibrator = ProbabilityCalibrator {
        method: CalibrationMethod::Identity,
        fitted: true,
        models: Vec::new(),
    };

    let error = calibrator
        .predict_proba(&array![[0.0, 0.0, 0.0]])
        .expect_err("zero probability mass must fail closed");
    assert!(error.to_string().contains("positive mass"));
}

#[test]
fn conformal_fit_rejects_non_finite_probability_without_fabricating_a_score() {
    let mut gate = ConformalGate::new(0.10);
    let mut probabilities = Array2::<f64>::zeros((32, 3));
    probabilities.column_mut(0).fill(1.0);
    probabilities[(7, 0)] = f64::NAN;

    let error = gate
        .fit_probabilities(&probabilities, &vec![0; 32])
        .expect_err("non-finite conformal input must fail closed");
    assert!(error.to_string().contains("finite"));
}

#[test]
fn legacy_unversioned_probability_calibrator_artifact_is_rejected() {
    let legacy = serde_json::json!({
        "method": "Identity",
        "fitted": true,
        "models": []
    });
    assert!(
        serde_json::from_value::<ProbabilityCalibratorArtifact>(legacy).is_err(),
        "an old f32-derived artifact must not deserialize as the current f64 schema"
    );
}

#[test]
fn legacy_probability_calibrator_schema_version_is_rejected() -> Result<()> {
    let legacy = serde_json::json!({
        "schema_version": 1,
        "method": "Identity",
        "fitted": true,
        "models": []
    });
    let artifact: ProbabilityCalibratorArtifact = serde_json::from_value(legacy)?;
    let error = validate_calibrator_artifact(&artifact)
        .expect_err("legacy f32-derived schema must fail closed");
    assert!(error.to_string().contains("schema version"));
    Ok(())
}

#[test]
fn validate_meta_metadata_rejects_inconsistent_training_summary() {
    let metadata = RuntimeArtifactMetadata::new(
        "meta_stack",
        ModelFamily::Meta,
        CapabilityState::Implemented,
        vec!["feature".to_string()],
        default_three_class_label_mapping(),
        crate::runtime::artifacts::TrainingSummaryMetadata::raw_for_validation(12, 8, 1),
    );

    let err = validate_meta_metadata(&metadata, "meta_stack")
        .expect_err("inconsistent meta training summary must fail");
    assert!(err.to_string().contains("training summary is inconsistent"));
}

#[test]
fn meta_runtime_prediction_uses_shared_three_class_confidence_gate() -> Result<()> {
    let gate = ConformalGate::new(0.10);
    let row = [0.51_f64, 0.49, 0.0];

    let prediction = build_meta_runtime_prediction("meta_stack", row, &gate, 2)?;
    let (expected_confidence, expected_abstain) = three_class_runtime_confidence(row)?;

    assert_eq!(prediction.confidence(), Some(expected_confidence));
    assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
    Ok(())
}

#[test]
fn conformal_prediction_artifact_rejects_invalid_prediction_set() {
    let err = validate_conformal_prediction_expert_artifact(&ConformalPredictionExpertArtifact {
        schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
        fitted: true,
        feature_columns: vec!["f1".to_string()],
        training_rows: 128,
        alpha: 0.10,
        method: CalibrationMethod::Platt,
        min_prediction_set: 4,
        min_fit_rows: 300,
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("min_prediction_set"));
}

#[test]
fn conformal_prediction_runtime_uses_expert_metadata_and_backend_details() -> Result<()> {
    let mut expert = ConformalPredictionExpert::new(CalibrationMethod::Temperature, 0.10);
    expert.fitted = true;
    expert.feature_columns = vec!["feature".to_string()];
    expert.training_rows = 128;
    expert.conformal_gate.fitted = true;
    expert.conformal_gate.n_calib = 128;
    expert.conformal_gate.qhat = 0.20;

    let frame = one_row_frame();
    let lease = one_worker_lease();
    expert.backend = MetaBlender {
        model: None,
        feature_columns: vec!["feature".to_string()],
        fitted: true,
        training_rows: 128,
    };
    expert.calibrator.fitted = true;
    expert.calibrator.method = CalibrationMethod::Temperature;
    expert.calibrator.models = vec![CalibrationModel::Temperature { temperature: 1.0 }];

    let predictions = expert.predict_runtime(&frame, &lease);
    assert!(
        predictions.is_err(),
        "cold backend should still fail prediction"
    );

    let backend = format!(
        "xgboost_meta_blender+{}_calibration+conformal_gate",
        calibration_method_name(CalibrationMethod::Temperature)
    );
    assert_eq!(
        backend,
        "xgboost_meta_blender+temperature_calibration+conformal_gate"
    );
    Ok(())
}

#[test]
fn probability_calibration_artifact_rejects_missing_feature_columns() {
    let err =
        validate_probability_calibration_expert_artifact(&ProbabilityCalibrationExpertArtifact {
            schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
            fitted: true,
            feature_columns: Vec::new(),
            training_rows: 128,
            method: CalibrationMethod::Platt,
            min_fit_rows: 300,
        })
        .unwrap_err()
        .to_string();

    assert!(err.contains("feature column"));
}

#[test]
fn probability_calibration_runtime_uses_shared_confidence_and_backend_details() -> Result<()> {
    let row = [0.52_f64, 0.33, 0.15];
    let prediction =
        build_probability_calibration_runtime_prediction(row, CalibrationMethod::Temperature)?;
    let (expected_confidence, expected_abstain) = three_class_runtime_confidence(row)?;

    assert_eq!(prediction.confidence(), Some(expected_confidence));
    assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
    assert_eq!(
        prediction.metadata().execution_backend.as_deref(),
        Some("xgboost_meta_blender+temperature_calibration")
    );
    Ok(())
}

#[test]
fn probability_calibration_runtime_surfaces_shared_abstain_reason() -> Result<()> {
    let row = [0.50_f64, 0.49, 0.01];
    let prediction =
        build_probability_calibration_runtime_prediction(row, CalibrationMethod::Temperature)?;

    assert_eq!(prediction.abstain_recommended(), Some(true));
    assert!(
        prediction
            .metadata()
            .degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("shared three-class confidence gate recommended abstain")
    );
    Ok(())
}

#[test]
fn meta_stack_artifact_rejects_invalid_prediction_set() {
    let err = validate_meta_stack_artifact(&MetaDecisionStackArtifact {
        schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
        fitted: true,
        feature_columns: vec!["f1".to_string()],
        training_rows: 128,
        method: CalibrationMethod::Platt,
        alpha: 0.10,
        min_prediction_set: 5,
        min_fit_rows: 300,
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("min_prediction_set"));
}

#[test]
fn meta_stack_runtime_uses_backend_details_and_shared_confidence() -> Result<()> {
    let gate = ConformalGate {
        alpha: 0.10,
        qhat: 0.20,
        fitted: true,
        n_calib: 128,
    };
    let row = [0.52_f64, 0.33, 0.15];
    let prediction =
        build_meta_stack_runtime_prediction(row, CalibrationMethod::Temperature, &gate, 2)?;
    let (expected_confidence, expected_abstain) = three_class_runtime_confidence(row)?;

    assert_eq!(prediction.confidence(), Some(expected_confidence));
    assert_eq!(
        prediction.abstain_recommended(),
        Some(expected_abstain || gate.should_abstain(&row, 2)?.0)
    );
    assert_eq!(
        prediction.metadata().execution_backend.as_deref(),
        Some("xgboost_meta_blender+temperature_calibration+conformal_gate")
    );
    Ok(())
}

#[test]
fn meta_stack_runtime_surfaces_combined_abstain_reasons() -> Result<()> {
    let gate = fitted_conformal_gate(0.10);
    let row = [0.50_f64, 0.49, 0.01];
    let prediction =
        build_meta_stack_runtime_prediction(row, CalibrationMethod::Temperature, &gate, 2)?;
    let degraded_reason = prediction
        .metadata()
        .degraded_reason
        .as_deref()
        .unwrap_or_default()
        .to_string();

    assert!(degraded_reason.contains("shared three-class confidence gate recommended abstain"));
    assert!(degraded_reason.contains("conformal prediction set size"));
    Ok(())
}

#[test]
fn conformal_runtime_surfaces_shared_and_conformal_abstain_reasons() -> Result<()> {
    let gate = fitted_conformal_gate(0.10);
    let row = [0.50_f64, 0.49, 0.01];
    let prediction =
        build_conformal_runtime_prediction(row, CalibrationMethod::Temperature, &gate, 2)?;
    let degraded_reason = prediction
        .metadata()
        .degraded_reason
        .as_deref()
        .unwrap_or_default()
        .to_string();

    assert!(degraded_reason.contains("shared three-class confidence gate recommended abstain"));
    assert!(degraded_reason.contains("conformal prediction set size"));
    Ok(())
}

#[test]
fn meta_blender_save_state_rejects_backend_feature_drift() {
    let mut backend = XGBoostExpert::new(0, None);
    backend.feature_columns = vec!["backend".to_string()];
    let blender = MetaBlender {
        model: Some(backend),
        feature_columns: vec!["state".to_string()],
        fitted: true,
        training_rows: 128,
    };

    let err = validate_meta_blender_save_state(&blender)
        .expect_err("feature-column drift must fail")
        .to_string();
    assert!(err.contains("feature-column mismatch"));
}

#[test]
fn probability_calibration_save_state_rejects_backend_training_row_drift() {
    let mut backend = XGBoostExpert::new(0, None);
    backend.feature_columns = vec!["feature".to_string()];
    let expert = ProbabilityCalibrationExpert {
        backend: MetaBlender {
            model: Some(backend),
            feature_columns: vec!["feature".to_string()],
            fitted: true,
            training_rows: 64,
        },
        calibrator: fitted_temperature_calibrator(),
        min_fit_rows: 300,
        fitted: true,
        feature_columns: vec!["feature".to_string()],
        training_rows: 128,
    };

    let err = validate_probability_calibration_expert_save_state(&expert)
        .expect_err("backend/state training-row drift must fail")
        .to_string();
    assert!(err.contains("training row mismatch"));
}

#[test]
fn meta_stack_save_state_rejects_blender_feature_drift() {
    let mut backend = XGBoostExpert::new(0, None);
    backend.feature_columns = vec!["backend".to_string()];
    let stack = MetaDecisionStack {
        blender: MetaBlender {
            model: Some(backend),
            feature_columns: vec!["backend".to_string()],
            fitted: true,
            training_rows: 128,
        },
        calibrator: fitted_temperature_calibrator(),
        conformal_gate: fitted_conformal_gate(0.10),
        min_prediction_set: 2,
        min_fit_rows: 300,
        fitted: true,
        feature_columns: vec!["state".to_string()],
        training_rows: 128,
    };

    let err = validate_meta_stack_save_state(&stack)
        .expect_err("feature-column drift must fail")
        .to_string();
    assert!(err.contains("feature-column mismatch"));
}
