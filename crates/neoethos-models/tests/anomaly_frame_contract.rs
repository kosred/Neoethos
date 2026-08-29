#[cfg(feature = "anomaly-detection")]
use std::path::PathBuf;
#[cfg(feature = "anomaly-detection")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "anomaly-detection")]
use anyhow::Context;
use anyhow::Result;
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_models::IsolationForestExpert;
use neoethos_models::base::ExpertModel;

#[cfg(feature = "anomaly-detection")]
const ANOMALY_F64_SCHEMA: &str = "neoethos.isolation_forest.f64.v2";

#[cfg(feature = "anomaly-detection")]
struct TempArtifactRoot(PathBuf);

#[cfg(feature = "anomaly-detection")]
impl TempArtifactRoot {
    fn create() -> Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock predates Unix epoch")?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-anomaly-frame-contract-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path)
            .with_context(|| format!("create temporary artifact root {}", path.display()))?;
        Ok(Self(path))
    }
}

#[cfg(feature = "anomaly-detection")]
impl Drop for TempArtifactRoot {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove anomaly contract temp root {}: {error}",
                self.0.display()
            );
        }
    }
}

fn frame_with_columns(column_count: usize) -> Result<FeatureFrame> {
    const ROWS: usize = 16;
    let columns = (0..column_count)
        .map(|column| {
            FeatureColumnF64::new(
                format!("f{column}"),
                (0..ROWS)
                    .map(|row| 1.0 + column as f64 * 0.03125 + row as f64 * 0.0078125)
                    .collect(),
                vec![FeatureCellValidity::Valid; ROWS],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(ROWS),
        columns,
    )
}

fn one_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(1).expect("one worker is valid");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("isolated anomaly contract can acquire one worker")
}

#[test]
fn anomaly_source_has_no_retired_precision_or_fallback_surface() {
    let source = include_str!("../src/anomaly/forest_impl.rs");
    for forbidden in [
        "polars::",
        "DataFrame",
        "Series",
        "Array2<f32>",
        "Vec<f32>",
        " as f32",
        "diagonal_profile",
        "gpu_policy_cpu_fallback_reason",
        "feature_matrix_from_dataframe",
    ] {
        assert!(
            !source.contains(forbidden),
            "isolation_forest still contains retired surface `{forbidden}`"
        );
    }
}

#[cfg(feature = "anomaly-detection")]
#[test]
fn isolation_forest_uses_typed_f64_frame_and_versioned_artifact() -> Result<()> {
    let frame = frame_with_columns(4)?;
    let labels = vec![0_i32; frame.n_samples()];
    let lease = one_worker_lease();
    let mut model = IsolationForestExpert::new(64, 16);
    model.fit(&frame, &labels, &lease)?;

    let probabilities = model.predict_proba(&frame, &lease)?;
    assert_eq!(probabilities.dim(), (frame.n_samples(), 3));
    for row in probabilities.outer_iter() {
        assert!(row.iter().all(|value| value.is_finite()));
        let sum = row.iter().sum::<f64>();
        assert!((sum - 1.0).abs() <= 1e-12, "probability sum is {sum}");
    }

    let root = TempArtifactRoot::create()?;
    model.save(&root.0)?;
    let model_path = root.0.join("model.json");
    let artifact: serde_json::Value = serde_json::from_slice(&std::fs::read(&model_path)?)?;
    assert_eq!(
        artifact["precision_schema"].as_str(),
        Some(ANOMALY_F64_SCHEMA)
    );
    assert_eq!(
        artifact["backend_kind"].as_str(),
        Some("extended_isolation_forest_cpu")
    );

    let mut loaded = IsolationForestExpert::new(64, 16);
    loaded.load(&root.0)?;
    assert_eq!(loaded.predict_proba(&frame, &lease)?.dim(), (16, 3));
    Ok(())
}

#[cfg(feature = "anomaly-detection")]
#[test]
fn isolation_forest_rejects_old_precision_schema() -> Result<()> {
    let frame = frame_with_columns(4)?;
    let labels = vec![0_i32; frame.n_samples()];
    let lease = one_worker_lease();
    let mut model = IsolationForestExpert::new(64, 16);
    model.fit(&frame, &labels, &lease)?;

    let root = TempArtifactRoot::create()?;
    model.save(&root.0)?;
    let model_path = root.0.join("model.json");
    let mut artifact: serde_json::Value = serde_json::from_slice(&std::fs::read(&model_path)?)?;
    artifact["precision_schema"] =
        serde_json::Value::String("neoethos.isolation_forest.f32.v1".to_string());
    std::fs::write(&model_path, serde_json::to_vec_pretty(&artifact)?)?;

    let error = IsolationForestExpert::new(64, 16)
        .load(&root.0)
        .expect_err("an old f32 artifact must fail closed");
    assert!(error.to_string().contains("precision schema"));
    Ok(())
}

#[cfg(feature = "anomaly-detection")]
#[test]
fn isolation_forest_rejects_more_than_compiled_dimensions_instead_of_changing_model() -> Result<()>
{
    let frame = frame_with_columns(129)?;
    let labels = vec![0_i32; frame.n_samples()];
    let lease = one_worker_lease();
    let error = IsolationForestExpert::new(64, 16)
        .fit(&frame, &labels, &lease)
        .expect_err("unsupported dimensionality must not select another model");
    assert!(error.to_string().contains("1..=128"));
    assert!(!error.to_string().contains("fallback"));
    Ok(())
}

#[cfg(not(feature = "anomaly-detection"))]
#[test]
fn isolation_forest_without_backend_fails_instead_of_substituting_a_profile() -> Result<()> {
    let frame = frame_with_columns(4)?;
    let labels = vec![0_i32; frame.n_samples()];
    let lease = one_worker_lease();
    let error = IsolationForestExpert::new(64, 16)
        .fit(&frame, &labels, &lease)
        .expect_err("a build without the real backend must fail closed");
    assert!(error.to_string().contains("anomaly-detection"));
    Ok(())
}
