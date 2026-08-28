use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use ndarray::Array2;
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_models::LogisticExpert;
use neoethos_models::base::ExpertModel;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FeatureColumnContract {
    name: String,
    values: Vec<f64>,
    current_f32_bits: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct MetadataContract {
    schema_version: u64,
    model_name: String,
    family: String,
    state: String,
    dataset_rows: u64,
    train_rows: u64,
    val_rows: u64,
}

#[derive(Debug, Deserialize)]
struct ModelFrameContract {
    schema: String,
    features: Vec<FeatureColumnContract>,
    labels: Vec<i32>,
    expected_probability_f32_bits: Vec<Vec<u32>>,
    expected_metadata: MetadataContract,
}

struct TempArtifactRoot(PathBuf);

impl TempArtifactRoot {
    fn create() -> Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock predates Unix epoch")?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-model-frame-contract-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path)
            .with_context(|| format!("create temporary artifact root {}", path.display()))?;
        Ok(Self(path))
    }

    fn artifact_path(&self) -> PathBuf {
        self.0.join("logistic")
    }
}

impl Drop for TempArtifactRoot {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove model contract temp root {}: {error}",
                self.0.display()
            );
        }
    }
}

fn load_contract() -> Result<ModelFrameContract> {
    serde_json::from_str(include_str!("fixtures/model_frame_contract_v2.json"))
        .context("parse model-frame contract fixture")
}

fn build_frame(features: &[FeatureColumnContract]) -> Result<FeatureFrame> {
    let rows = features
        .first()
        .map(|feature| feature.values.len())
        .unwrap_or_default();
    let columns = features
        .iter()
        .map(|feature| {
            FeatureColumnF64::new(
                feature.name.clone(),
                feature.values.clone(),
                vec![FeatureCellValidity::Valid; rows],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
}

fn single_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(1).expect("one worker is a valid lease width");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("the isolated model-contract broker has one available permit")
}

fn backend_f32_probability_bits(probabilities: &Array2<f64>) -> Vec<Vec<u32>> {
    probabilities
        .outer_iter()
        .map(|row| {
            row.iter()
                .copied()
                .map(|value| (value as f32).to_bits())
                .collect()
        })
        .collect()
}

fn metadata_value(path: &Path) -> Result<serde_json::Value> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read persisted metadata {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parse persisted runtime metadata")
}

#[cfg(target_os = "linux")]
fn peak_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_kib() -> Option<u64> {
    None
}

#[test]
fn shared_model_base_has_no_polars_compatibility_surface() {
    let source = include_str!("../src/base.rs");
    for forbidden in [
        "polars::",
        "DataFrame",
        "Series",
        "dataframe_to_float32_array",
        "feature_columns_from_dataframe",
        "strict_numeric_column_values",
    ] {
        assert!(
            !source.contains(forbidden),
            "shared model base still exposes retired Polars compatibility symbol `{forbidden}`"
        );
    }
}

#[test]
fn f64_feature_frame_contract_is_deterministic_and_strict() -> Result<()> {
    let fixture = load_contract()?;
    ensure!(
        fixture.schema == "neoethos.model_frame_contract.v2",
        "unexpected fixture schema {}",
        fixture.schema
    );
    ensure!(!fixture.features.is_empty(), "fixture has no features");
    let rows = fixture.labels.len();
    ensure!(rows > 6, "fixture must exercise a validation split");

    for feature in &fixture.features {
        ensure!(
            feature.values.len() == rows,
            "feature {} row count differs from labels",
            feature.name
        );
        assert_eq!(
            feature
                .values
                .iter()
                .copied()
                .map(|value| (value as f32).to_bits())
                .collect::<Vec<_>>(),
            feature.current_f32_bits,
            "recorded current f64-to-f32 conversion changed for {}",
            feature.name
        );
    }

    let frame = build_frame(&fixture.features)?;
    let labels = fixture.labels.clone();
    let lease = single_worker_lease();
    let mut first = LogisticExpert::new();
    first.fit(&frame, &labels, &lease)?;
    let first_probabilities = first.predict_proba(&frame, &lease)?;
    assert_eq!(first_probabilities.dim(), (rows, 3));
    for (row_idx, row) in first_probabilities.outer_iter().enumerate() {
        assert!(
            row.iter().all(|value| value.is_finite()),
            "row {row_idx} contains a non-finite probability"
        );
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() <= 1e-6,
            "row {row_idx} probability sum is {sum}"
        );
    }
    let first_bits = backend_f32_probability_bits(&first_probabilities);
    assert!(
        !fixture.expected_probability_f32_bits.is_empty(),
        "capture deterministic baseline probability bits: {first_bits:?}"
    );
    assert_eq!(
        first_bits, fixture.expected_probability_f32_bits,
        "deterministic logistic probability contract changed"
    );

    let mut second = LogisticExpert::new();
    second.fit(&frame, &labels, &lease)?;
    assert_eq!(
        backend_f32_probability_bits(&second.predict_proba(&frame, &lease)?),
        first_bits,
        "same input/config produced different predictions in one process"
    );

    let temp_root = TempArtifactRoot::create()?;
    let artifact_path = temp_root.artifact_path();
    first.save(&artifact_path)?;
    let metadata = metadata_value(&artifact_path.join("metadata.json"))?;
    let expected = &fixture.expected_metadata;
    assert_eq!(
        metadata["schema_version"].as_u64(),
        Some(expected.schema_version)
    );
    assert_eq!(
        metadata["model_name"].as_str(),
        Some(expected.model_name.as_str())
    );
    assert_eq!(metadata["family"].as_str(), Some(expected.family.as_str()));
    assert_eq!(metadata["state"].as_str(), Some(expected.state.as_str()));
    assert_eq!(
        metadata["feature_columns"],
        serde_json::json!(
            fixture
                .features
                .iter()
                .map(|feature| feature.name.as_str())
                .collect::<Vec<_>>()
        )
    );
    assert_eq!(
        metadata["training_summary"]["dataset_rows"].as_u64(),
        Some(expected.dataset_rows)
    );
    assert_eq!(
        metadata["training_summary"]["train_rows"].as_u64(),
        Some(expected.train_rows)
    );
    assert_eq!(
        metadata["training_summary"]["val_rows"].as_u64(),
        Some(expected.val_rows)
    );

    let mut loaded = LogisticExpert::new();
    loaded.load(&artifact_path)?;
    assert_eq!(
        backend_f32_probability_bits(&loaded.predict_proba(&frame, &lease)?),
        first_bits,
        "save/load changed deterministic probabilities"
    );

    let reversed_features: Vec<FeatureColumnContract> = fixture
        .features
        .iter()
        .rev()
        .map(|feature| FeatureColumnContract {
            name: feature.name.clone(),
            values: feature.values.clone(),
            current_f32_bits: feature.current_f32_bits.clone(),
        })
        .collect();
    let reordered = build_frame(&reversed_features)?;
    let reorder_error = loaded
        .predict_proba(&reordered, &lease)
        .expect_err("reordered feature columns must fail closed");
    assert!(
        reorder_error
            .to_string()
            .contains("feature column mismatch"),
        "unexpected reordered-column error: {reorder_error:#}"
    );

    let mut invalid_columns = fixture
        .features
        .iter()
        .map(|feature| {
            FeatureColumnF64::new(
                feature.name.clone(),
                feature.values.clone(),
                vec![FeatureCellValidity::Valid; rows],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    invalid_columns[0].invalidate(3, FeatureCellValidity::MissingInput)?;
    let invalid_frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        invalid_columns,
    )?;
    let invalid_error = LogisticExpert::new()
        .fit(&invalid_frame, &labels, &lease)
        .expect_err("invalid model input must fail closed");
    assert!(
        invalid_error.to_string().contains("row 3")
            && invalid_error.to_string().contains("missing_input"),
        "unexpected invalid-cell error: {invalid_error:#}"
    );

    let malformed = FeatureColumnF64::new(
        "bad",
        vec![0.0, f64::NAN],
        vec![FeatureCellValidity::Valid; 2],
    )
    .expect_err("valid-marked NaN must fail at the shared feature boundary");
    assert!(
        malformed
            .to_string()
            .contains("marked valid with non-finite")
    );

    Ok(())
}

#[test]
#[ignore = "explicit Task-1 performance baseline; three warmups plus ten measured runs"]
fn baseline_fixed_logistic_training_metrics() -> Result<()> {
    const TRAININGS_PER_SAMPLE: usize = 64;

    let fixture = load_contract()?;
    let frame = build_frame(&fixture.features)?;
    let labels = fixture.labels.clone();
    let expected = fixture.expected_probability_f32_bits;
    let lease = single_worker_lease();

    let run_sample = || -> Result<u64> {
        let mut checksum = 0_u64;
        for _ in 0..TRAININGS_PER_SAMPLE {
            let mut model = LogisticExpert::new();
            model.fit(&frame, &labels, &lease)?;
            let probabilities = model.predict_proba(&frame, &lease)?;
            let bits = backend_f32_probability_bits(&probabilities);
            ensure!(bits == expected, "training benchmark changed contract bits");
            checksum = checksum.wrapping_add(
                bits.iter()
                    .flatten()
                    .fold(0_u64, |sum, value| sum.wrapping_add(u64::from(*value))),
            );
        }
        Ok(checksum)
    };

    let mut checksum = 0_u64;
    for _ in 0..3 {
        checksum = checksum.wrapping_add(run_sample()?);
    }
    let mut samples_ns = Vec::with_capacity(10);
    for _ in 0..10 {
        let started = Instant::now();
        checksum = checksum.wrapping_add(run_sample()?);
        samples_ns.push(started.elapsed().as_nanos());
    }

    let report = serde_json::json!({
        "schema": "neoethos.task1_model_baseline.v1",
        "rows": frame.n_samples(),
        "features": frame.n_features(),
        "trainings_per_sample": TRAININGS_PER_SAMPLE,
        "warmups": 3,
        "measured_runs": 10,
        "device_policy": "cpu",
        "peak_rss_kib": peak_rss_kib(),
        "training_predict_ns": samples_ns,
        "checksum": checksum
    });
    println!("NEOETHOS_BASELINE_JSON={report}");
    Ok(())
}
