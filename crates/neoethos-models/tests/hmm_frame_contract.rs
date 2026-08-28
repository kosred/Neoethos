use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use ndarray::Array2;
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_models::forecasting::{HmmRegimeConfig, RegimeHmmExpert};

const HMM_F64_SCHEMA: &str = "neoethos.hmm_regime.f64_validity.v2";

struct TempArtifactRoot(PathBuf);

impl TempArtifactRoot {
    fn create() -> Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock predates Unix epoch")?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-hmm-frame-contract-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path)
            .with_context(|| format!("create HMM artifact root {}", path.display()))?;
        Ok(Self(path))
    }
}

impl Drop for TempArtifactRoot {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove HMM contract root {}: {error}",
                self.0.display()
            );
        }
    }
}

fn one_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(1).expect("one worker is valid");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("isolated HMM contract can acquire one worker")
}

fn trained_hmm() -> Result<RegimeHmmExpert> {
    let rows = 32;
    let mut observations = Array2::<f64>::zeros((rows, 2));
    for row in 0..rows {
        observations[(row, 0)] = ((row % 7) as f64 - 3.0) * 0.000_125;
        observations[(row, 1)] = -7.5 + ((row % 5) as f64) * 0.031_25;
    }
    RegimeHmmExpert::train(
        &observations,
        vec![
            "quant_log_return".to_string(),
            "quant_log_volatility".to_string(),
        ],
        HmmRegimeConfig {
            min_training_bars: rows,
            max_em_iterations: 4,
            ..HmmRegimeConfig::default()
        },
    )
}

fn prediction_frame() -> Result<FeatureFrame> {
    let rows = 6;
    let mut return_validity = vec![FeatureCellValidity::Valid; rows];
    return_validity[0] = FeatureCellValidity::Warmup;
    return_validity[3] = FeatureCellValidity::Gap;
    let columns = vec![
        FeatureColumnF64::new(
            "quant_log_return",
            vec![0.0, 0.000_1, 0.000_2, 0.0, -0.000_1, -0.000_2],
            return_validity,
        )?,
        FeatureColumnF64::new(
            "quant_log_volatility",
            vec![-7.0, -7.1, -7.2, -7.3, -7.4, -7.5],
            vec![FeatureCellValidity::Valid; rows],
        )?,
    ];
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
}

fn training_frame() -> Result<FeatureFrame> {
    let rows = 6;
    let columns = vec![
        FeatureColumnF64::new(
            "quant_log_return",
            vec![0.000_1, -0.000_2, 0.000_3, -0.000_4, 0.000_5, -0.000_6],
            vec![FeatureCellValidity::Valid; rows],
        )?,
        FeatureColumnF64::new(
            "quant_log_volatility",
            vec![-7.0, -7.1, -7.2, -7.3, -7.4, -7.5],
            vec![FeatureCellValidity::Valid; rows],
        )?,
    ];
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
}

#[test]
fn hmm_source_has_no_retired_dataframe_f32_or_uniform_fallback() {
    let source = include_str!("../src/forecasting/hmm_regime.rs");
    for forbidden in [
        "polars::",
        "DataFrame",
        "Array2<f32>",
        "Vec<f32>",
        " as f32",
        "uniform prior fallback",
        "predict_proba_from_dataframe",
        "dataframe_to_ohlcv_arrays",
    ] {
        assert!(
            !source.contains(forbidden),
            "HMM still contains retired surface `{forbidden}`"
        );
    }

    let adapter = include_str!("../src/ensemble_inference/meta_adapters.rs");
    assert!(!adapter.contains("predict_proba_from_dataframe"));
    assert!(adapter.contains("project_expert_frame(frame, self.feature_columns(), self.name())"));
    assert!(adapter.contains("predict_feature_frame(&projected, lease)"));

    let contract = include_str!("../src/ensemble_inference/mod.rs");
    assert!(contract.contains("pub values: Vec<f64>"));
    assert!(contract.contains("frame: &FeatureFrame"));
    assert!(contract.contains("lease: &CpuLease"));
}

#[test]
fn hmm_preserves_invalid_rows_instead_of_fabricating_probabilities() -> Result<()> {
    let expert = trained_hmm()?;
    let frame = prediction_frame()?;
    let posterior = expert.predict_feature_frame(&frame, &one_worker_lease())?;

    assert_eq!(posterior.probabilities.dim(), (frame.n_samples(), 3));
    assert_eq!(posterior.validity.len(), frame.n_samples());
    assert_eq!(posterior.validity[0], FeatureCellValidity::Warmup);
    assert_eq!(posterior.validity[3], FeatureCellValidity::Gap);
    for row in [0, 3] {
        assert!(
            posterior
                .probabilities
                .row(row)
                .iter()
                .all(|value| value.is_nan()),
            "invalid HMM row {row} must retain a NaN payload"
        );
    }
    for row in [1, 2, 4, 5] {
        assert_eq!(posterior.validity[row], FeatureCellValidity::Valid);
        let sum = posterior.probabilities.row(row).iter().sum::<f64>();
        assert!((sum - 1.0).abs() <= 1e-12, "row {row} sum is {sum}");
    }
    Ok(())
}

#[test]
fn hmm_training_observations_are_exact_versioned_feature_values() -> Result<()> {
    let frame = training_frame()?;
    let observations =
        RegimeHmmExpert::training_observations_from_feature_frame(&frame, &one_worker_lease())?;

    assert_eq!(observations.dim(), (frame.n_samples(), 2));
    for row in 0..frame.n_samples() {
        assert_eq!(
            observations[(row, 0)].to_bits(),
            frame.feature_column(0)?.values[row].to_bits()
        );
        assert_eq!(
            observations[(row, 1)].to_bits(),
            frame.feature_column(1)?.values[row].to_bits()
        );
    }
    Ok(())
}

#[test]
fn hmm_artifact_is_f64_validity_versioned_and_old_schema_fails() -> Result<()> {
    let expert = trained_hmm()?;
    let root = TempArtifactRoot::create()?;
    expert.save_to_path(&root.0)?;

    let path = root.0.join("hmm_regime.json");
    let mut artifact: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    assert_eq!(artifact["precision_schema"].as_str(), Some(HMM_F64_SCHEMA));

    artifact["precision_schema"] =
        serde_json::Value::String("neoethos.hmm_regime.f32.v1".to_string());
    std::fs::write(&path, serde_json::to_vec_pretty(&artifact)?)?;
    let error = RegimeHmmExpert::load_from_artifact(&root.0)
        .expect_err("old HMM precision schema must fail closed");
    assert!(error.to_string().contains("precision schema"));
    Ok(())
}
