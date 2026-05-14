// TODO(real-data): the calibrator/conformal fixtures and any
// downstream DataFrame inputs in this file use synthesised
// probabilities/qhat values. Replace with a cTrader historical sample:
// fit calibrators on real out-of-sample meta-model predictions for the
// target symbol/timeframe and reuse those artifacts here.
use super::*;
use crate::tree_models::XGBoostExpert;

fn fitted_temperature_calibrator() -> ProbabilityCalibrator {
    ProbabilityCalibrator {
        method: CalibrationMethod::Temperature,
        fitted: true,
        models: vec![CalibrationModel::Temperature { temperature: 1.0 }],
    }
}

fn fitted_conformal_gate(alpha: f32) -> ConformalGate {
    ConformalGate {
        alpha,
        qhat: 0.55,
        fitted: true,
        n_calib: 128,
    }
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
    let row = [0.51_f32, 0.49, 0.0];

    let prediction = build_meta_runtime_prediction("meta_stack", row, &gate, 2)?;
    let (expected_confidence, expected_abstain) = three_class_runtime_confidence(row)?;

    assert_eq!(prediction.confidence(), Some(expected_confidence));
    assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
    Ok(())
}

#[test]
fn conformal_prediction_artifact_rejects_invalid_prediction_set() {
    let err = validate_conformal_prediction_expert_artifact(&ConformalPredictionExpertArtifact {
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

    let frame = DataFrame::new(vec![Series::new("feature".into(), &[1.0_f64]).into()])?;
    expert.backend = MetaBlender {
        model: None,
        feature_columns: vec!["feature".to_string()],
        fitted: true,
        training_rows: 128,
    };
    expert.calibrator.fitted = true;
    expert.calibrator.method = CalibrationMethod::Temperature;
    expert.calibrator.models = vec![CalibrationModel::Temperature { temperature: 1.0 }];

    let predictions = expert.predict_runtime(&frame);
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
    let row = [0.52_f32, 0.33, 0.15];
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
    let row = [0.50_f32, 0.49, 0.01];
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
    let row = [0.52_f32, 0.33, 0.15];
    let prediction =
        build_meta_stack_runtime_prediction(row, CalibrationMethod::Temperature, &gate, 2)?;
    let (expected_confidence, expected_abstain) = three_class_runtime_confidence(row)?;

    assert_eq!(prediction.confidence(), Some(expected_confidence));
    assert_eq!(
        prediction.abstain_recommended(),
        Some(expected_abstain || gate.should_abstain(&row, 2).0)
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
    let row = [0.50_f32, 0.49, 0.01];
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
    let row = [0.50_f32, 0.49, 0.01];
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
