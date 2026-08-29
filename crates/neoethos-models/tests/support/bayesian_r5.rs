use ndarray::{Array1, Array2, Axis};
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_models::BayesianLogitExpert;
use neoethos_models::base::ExpertModel;
use neoethos_models::statistical::common::install_statistical_runtime_from_settings;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const PRIOR_PRECISION: f64 = 0.05;
const LEARNING_RATE: f64 = 0.05;
const EPOCHS: usize = 120;
const TIMED_SAMPLES: usize = 3;
const MINIMUM_KERNEL_DURATION_NS: u64 = 1_000_000;
const MINIMUM_STAGE_DURATION_NS: u64 = 10_000;

#[derive(Debug, Clone)]
pub struct OracleCase {
    pub name: &'static str,
    pub train_features: Array2<f64>,
    pub train_labels: Vec<i32>,
    pub oos_features: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct OraclePosterior {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub covariance: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct OracleFit {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub classes: Vec<OraclePosterior>,
    pub oos_probabilities: Array2<f64>,
}

#[derive(Debug, Deserialize)]
struct SavedScaler {
    means: Vec<f64>,
    stds: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct SavedPosterior {
    weights: Array1<f64>,
    bias: f64,
    covariance: Array2<f64>,
}

#[derive(Debug, Deserialize)]
struct SavedArtifact {
    precision_schema: String,
    model_name: String,
    scaler: SavedScaler,
    runtime_backend: String,
    classes: Vec<SavedPosterior>,
}

struct ArtifactDir(PathBuf);

impl ArtifactDir {
    fn create(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-bayesian-r5-oracle-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create parent-owned oracle artifact directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ArtifactDir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove Bayesian R5 oracle artifact {}: {error}",
                self.0.display()
            );
        }
    }
}

pub fn fixture_cases() -> Vec<OracleCase> {
    vec![normal_case(), extreme_finite_case(), ill_conditioned_case()]
}

fn labels(rows: usize) -> Vec<i32> {
    (0..rows)
        .map(|row| {
            let class = if rows > 30 && row % 29 == 0 {
                (row + 1) % 3
            } else {
                row % 3
            };
            [-1, 0, 1][class]
        })
        .collect()
}

fn normal_case() -> OracleCase {
    // Six training rows deliberately exercise the production branch with no
    // validation split. This keeps the math oracle independent of the separate
    // four-way training-summary embargo RED contract.
    let train_rows = 6;
    let oos_rows = 7;
    OracleCase {
        name: "normal",
        train_features: Array2::from_shape_fn((train_rows, 4), |(row, column)| {
            let class = row % 3;
            let class_signal = match (column, class) {
                (0, 0) | (1, 1) | (2, 2) => 1.4,
                (0, 2) | (1, 0) | (2, 1) => -1.0,
                _ => 0.2,
            };
            let phase = (row * (column + 3)) as f64 * 0.071;
            class_signal + phase.sin() * 0.17 + (phase * 0.41).cos() * 0.06
        }),
        train_labels: labels(train_rows),
        oos_features: Array2::from_shape_fn((oos_rows, 4), |(row, column)| {
            let source_row = row + train_rows + 7;
            let class = source_row % 3;
            let class_signal = match (column, class) {
                (0, 0) | (1, 1) | (2, 2) => 1.4,
                (0, 2) | (1, 0) | (2, 1) => -1.0,
                _ => 0.2,
            };
            let phase = (source_row * (column + 3)) as f64 * 0.071;
            class_signal + phase.sin() * 0.17 + (phase * 0.41).cos() * 0.06
        }),
    }
}

fn extreme_finite_case() -> OracleCase {
    let train_rows = 6;
    let oos_rows = 7;
    let value = |row: usize, column: usize| {
        let centered = row as f64 - 2.5;
        match column {
            0 => 1.0e6 + centered * 17.0,
            1 => -7.5e5 + ((row * 13 % 37) as f64 - 18.0) * 31.0,
            2 => 1.0e-6 * (centered + (row % 3) as f64 * 0.25),
            _ => ((row * 17 % 41) as f64 - 20.0) * 9.0e3,
        }
    };
    OracleCase {
        name: "extreme-finite",
        train_features: Array2::from_shape_fn((train_rows, 4), |(row, column)| value(row, column)),
        train_labels: labels(train_rows),
        oos_features: Array2::from_shape_fn((oos_rows, 4), |(row, column)| {
            value(row + train_rows + 3, column)
        }),
    }
}

fn ill_conditioned_case() -> OracleCase {
    let train_rows = 6;
    let oos_rows = 7;
    let value = |row: usize, column: usize| {
        let base = (row as f64 * 0.113).sin() + (row % 3) as f64 * 0.4;
        match column {
            0 => base,
            1 => 2.0 * base,
            2 => base + (row as f64 * 1.0e-11),
            _ => 3.0 * base + ((row * 7 % 5) as f64 - 2.0) * 1.0e-10,
        }
    };
    OracleCase {
        name: "ill-conditioned",
        train_features: Array2::from_shape_fn((train_rows, 4), |(row, column)| value(row, column)),
        train_labels: labels(train_rows),
        oos_features: Array2::from_shape_fn((oos_rows, 4), |(row, column)| {
            value(row + train_rows + 11, column)
        }),
    }
}

fn remap_labels(labels: &[i32]) -> Vec<usize> {
    labels
        .iter()
        .map(|label| match label {
            -1 => 2,
            0 => 0,
            1 => 1,
            unexpected => panic!("oracle fixture contains unsupported label {unexpected}"),
        })
        .collect()
}

fn split_train_validation(rows: usize) -> (Vec<usize>, Vec<usize>) {
    if rows <= 6 {
        return ((0..rows).collect(), Vec::new());
    }
    let validation_rows = ((rows as f64) * 0.2).round() as usize;
    let validation_rows = validation_rows.clamp(1, rows.saturating_sub(2));
    let embargo_rows = if rows >= 20 {
        ((rows as f64) * 0.02).round() as usize
    } else {
        0
    };
    let embargo_rows = embargo_rows.clamp(0, rows.saturating_sub(validation_rows + 1));
    let training_rows = rows.saturating_sub(validation_rows + embargo_rows);
    if training_rows == 0 {
        return ((0..rows).collect(), Vec::new());
    }
    (
        (0..training_rows).collect(),
        (training_rows + embargo_rows..rows).collect(),
    )
}

fn fit_scaler(features: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    assert!(features.nrows() > 0 && features.ncols() > 0);
    assert!(features.iter().all(|value| value.is_finite()));
    let mut means = vec![0.0; features.ncols()];
    let mut stds = vec![1.0; features.ncols()];
    for column in 0..features.ncols() {
        let mut sum = 0.0;
        for row in 0..features.nrows() {
            sum += features[(row, column)];
        }
        let mean = sum / features.nrows() as f64;
        means[column] = mean;
        let mut variance = 0.0;
        for row in 0..features.nrows() {
            let centered = features[(row, column)] - mean;
            variance += centered * centered;
        }
        let std = (variance / features.nrows() as f64).sqrt();
        stds[column] = if std.is_finite() && std > 1.0e-12 {
            std
        } else {
            1.0
        };
    }
    (means, stds)
}

fn scale(features: &Array2<f64>, means: &[f64], stds: &[f64]) -> Array2<f64> {
    assert_eq!(features.ncols(), means.len());
    assert_eq!(features.ncols(), stds.len());
    let mut result = features.clone();
    for row in 0..result.nrows() {
        for column in 0..result.ncols() {
            result[(row, column)] = (result[(row, column)] - means[column]) / stds[column];
        }
    }
    assert!(result.iter().all(|value| value.is_finite()));
    result
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn independent_cholesky_inverse(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    let dimension = matrix.nrows();
    if dimension == 0 || matrix.ncols() != dimension {
        return None;
    }
    let mut lower = Array2::<f64>::zeros((dimension, dimension));
    for row in 0..dimension {
        for column in 0..=row {
            let mut residual = matrix[(row, column)];
            for inner in 0..column {
                residual -= lower[(row, inner)] * lower[(column, inner)];
            }
            if row == column {
                if !residual.is_finite() || residual <= 0.0 {
                    return None;
                }
                lower[(row, column)] = residual.sqrt();
            } else {
                let divisor = lower[(column, column)];
                if !divisor.is_finite() || divisor <= 0.0 {
                    return None;
                }
                lower[(row, column)] = residual / divisor;
            }
        }
    }

    let mut inverse = Array2::<f64>::zeros((dimension, dimension));
    for target in 0..dimension {
        let mut forward = vec![0.0; dimension];
        for row in 0..dimension {
            let mut residual = f64::from(row == target);
            for column in 0..row {
                residual -= lower[(row, column)] * forward[column];
            }
            forward[row] = residual / lower[(row, row)];
        }
        let mut backward = vec![0.0; dimension];
        for row in (0..dimension).rev() {
            let mut residual = forward[row];
            for column in row + 1..dimension {
                residual -= lower[(column, row)] * backward[column];
            }
            backward[row] = residual / lower[(row, row)];
        }
        for row in 0..dimension {
            inverse[(row, target)] = backward[row];
        }
    }
    inverse
        .iter()
        .all(|value| value.is_finite())
        .then_some(inverse)
}

fn fit_binary_oracle(
    train_features: &Array2<f64>,
    train_labels: &[f64],
    validation_features: Option<&Array2<f64>>,
    validation_labels: Option<&[f64]>,
) -> OraclePosterior {
    let rows = train_features.nrows();
    let columns = train_features.ncols();
    assert!(rows > 0 && columns > 0);
    assert_eq!(train_labels.len(), rows);
    let prior = PRIOR_PRECISION.max(1.0e-6);
    let learning_rate = LEARNING_RATE.max(1.0e-4);
    let mut weights = Array1::<f64>::zeros(columns);
    let mut bias = 0.0;
    let mut best_weights = weights.clone();
    let mut best_bias = bias;
    let mut best_validation_loss = f64::INFINITY;
    let mut stale_epochs = 0usize;

    for _ in 0..EPOCHS.max(1) {
        let mut weight_gradient = Array1::<f64>::zeros(columns);
        let mut bias_gradient = 0.0;
        for row in 0..rows {
            let mut logit = bias;
            for column in 0..columns {
                logit += weights[column] * train_features[(row, column)];
            }
            let error = sigmoid(logit) - train_labels[row];
            for column in 0..columns {
                weight_gradient[column] += error * train_features[(row, column)];
            }
            bias_gradient += error;
        }
        for column in 0..columns {
            weight_gradient[column] =
                weight_gradient[column] / rows as f64 + prior * weights[column];
            weights[column] -= learning_rate * weight_gradient[column];
        }
        bias_gradient /= rows as f64;
        bias -= learning_rate * bias_gradient;

        if let (Some(features), Some(labels)) = (validation_features, validation_labels)
            && features.nrows() > 0
        {
            let mut loss = 0.0;
            for row in 0..features.nrows() {
                let mut logit = bias;
                for column in 0..columns {
                    logit += weights[column] * features[(row, column)];
                }
                let probability = sigmoid(logit).clamp(1.0e-6, 1.0 - 1.0e-6);
                loss -=
                    labels[row] * probability.ln() + (1.0 - labels[row]) * (1.0 - probability).ln();
            }
            loss /= features.nrows() as f64;
            if loss + 1.0e-6 < best_validation_loss {
                best_validation_loss = loss;
                best_weights = weights.clone();
                best_bias = bias;
                stale_epochs = 0;
            } else {
                stale_epochs += 1;
                if stale_epochs >= 25 {
                    break;
                }
            }
        }
    }
    if best_validation_loss.is_finite() {
        weights = best_weights;
        bias = best_bias;
    }

    let augmented = columns + 1;
    let mut hessian = Array2::<f64>::zeros((augmented, augmented));
    for row in 0..rows {
        let mut logit = bias;
        for column in 0..columns {
            logit += weights[column] * train_features[(row, column)];
        }
        let probability = sigmoid(logit);
        let curvature = (probability * (1.0 - probability)).max(1.0e-6);
        for left in 0..augmented {
            let left_value = if left < columns {
                train_features[(row, left)]
            } else {
                1.0
            };
            if left_value == 0.0 {
                continue;
            }
            for right in 0..augmented {
                let right_value = if right < columns {
                    train_features[(row, right)]
                } else {
                    1.0
                };
                hessian[(left, right)] += curvature * left_value * right_value;
            }
        }
    }
    let prior_alpha = prior * rows as f64;
    for diagonal in 0..columns {
        hessian[(diagonal, diagonal)] += prior_alpha;
    }
    hessian[(columns, columns)] += 1.0e-6;

    let mut jitter = 0.0;
    let covariance = loop {
        let mut candidate = hessian.clone();
        for diagonal in 0..augmented {
            candidate[(diagonal, diagonal)] += jitter;
        }
        if let Some(inverse) = independent_cholesky_inverse(&candidate) {
            break inverse;
        }
        jitter = if jitter == 0.0 { 1.0e-8 } else { jitter * 10.0 };
        assert!(
            jitter <= 1.0,
            "independent oracle Cholesky jitter exhausted"
        );
    };

    OraclePosterior {
        weights,
        bias,
        covariance,
    }
}

fn predictive_logit(posterior: &OraclePosterior, features: &[f64]) -> f64 {
    let mut mean = posterior.bias;
    for (weight, feature) in posterior.weights.iter().zip(features) {
        mean += weight * feature;
    }
    let augmented = features.len() + 1;
    assert_eq!(posterior.covariance.dim(), (augmented, augmented));
    let values = features
        .iter()
        .copied()
        .chain(std::iter::once(1.0))
        .collect::<Vec<_>>();
    let mut variance = 0.0;
    for row in 0..augmented {
        for column in 0..augmented {
            variance += values[row] * posterior.covariance[(row, column)] * values[column];
        }
    }
    assert!(mean.is_finite() && variance.is_finite());
    let correction = (1.0 + std::f64::consts::PI * variance.max(0.0) / 8.0).sqrt();
    mean / correction.max(1.0e-6)
}

fn softmax(logits: &Array2<f64>) -> Array2<f64> {
    let mut probabilities = logits.clone();
    for row in 0..probabilities.nrows() {
        let mut maximum = f64::NEG_INFINITY;
        for column in 0..probabilities.ncols() {
            maximum = maximum.max(probabilities[(row, column)]);
        }
        let mut denominator = 0.0;
        for column in 0..probabilities.ncols() {
            probabilities[(row, column)] = (probabilities[(row, column)] - maximum).exp();
            denominator += probabilities[(row, column)];
        }
        assert!(denominator.is_finite() && denominator > 0.0);
        for column in 0..probabilities.ncols() {
            probabilities[(row, column)] /= denominator;
        }
    }
    probabilities
}

pub fn fit_oracle(case: &OracleCase) -> OracleFit {
    let remapped = remap_labels(&case.train_labels);
    let (training_indices, validation_indices) =
        split_train_validation(case.train_features.nrows());
    let training_features = case.train_features.select(Axis(0), &training_indices);
    let validation_features = case.train_features.select(Axis(0), &validation_indices);
    let training_labels = training_indices
        .iter()
        .map(|index| remapped[*index])
        .collect::<Vec<_>>();
    let validation_labels = validation_indices
        .iter()
        .map(|index| remapped[*index])
        .collect::<Vec<_>>();
    let (means, stds) = fit_scaler(&training_features);
    let scaled_training = scale(&training_features, &means, &stds);
    let scaled_validation = scale(&validation_features, &means, &stds);
    let scaled_oos = scale(&case.oos_features, &means, &stds);

    let mut classes = Vec::with_capacity(3);
    for class in 0..3 {
        let binary_training = training_labels
            .iter()
            .map(|label| f64::from(*label == class))
            .collect::<Vec<_>>();
        let binary_validation = validation_labels
            .iter()
            .map(|label| f64::from(*label == class))
            .collect::<Vec<_>>();
        classes.push(fit_binary_oracle(
            &scaled_training,
            &binary_training,
            (!validation_indices.is_empty()).then_some(&scaled_validation),
            (!validation_indices.is_empty()).then_some(binary_validation.as_slice()),
        ));
    }

    let logits = Array2::from_shape_fn((scaled_oos.nrows(), 3), |(row, class)| {
        predictive_logit(&classes[class], &scaled_oos.row(row).to_vec())
    });
    let oos_probabilities = softmax(&logits);
    OracleFit {
        means,
        stds,
        classes,
        oos_probabilities,
    }
}

fn feature_frame(matrix: &Array2<f64>) -> FeatureFrame {
    let rows = matrix.nrows();
    let columns = (0..matrix.ncols())
        .map(|column| {
            FeatureColumnF64::new(
                format!("bayesian_r5_feature_{column}"),
                matrix.column(column).to_vec(),
                vec![FeatureCellValidity::Valid; rows],
            )
            .expect("construct finite Bayesian R5 feature column")
        })
        .collect::<Vec<_>>();
    neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
    .expect("construct public Bayesian R5 FeatureFrame")
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance * scale,
        "{context}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleParityReceipt {
    pub case_name: String,
    pub train_feature_sha256: String,
    pub train_label_sha256: String,
    pub oos_feature_sha256: String,
    pub runtime_backend: String,
    pub model_artifact_sha256: String,
    pub posterior_values: Vec<f64>,
    pub oracle_posterior_values: Vec<f64>,
    pub probability_rows: usize,
    pub probability_columns: usize,
    pub probability_values: Vec<f64>,
    pub oracle_probability_values: Vec<f64>,
    pub max_normalized_oracle_error: f64,
}

fn observe_oracle_value(
    actual: f64,
    expected: f64,
    tolerance: f64,
    context: &str,
    maximum: &mut f64,
) {
    assert!(actual.is_finite(), "{context}: public value is non-finite");
    assert!(
        expected.is_finite(),
        "{context}: oracle value is non-finite"
    );
    let scale = actual.abs().max(expected.abs()).max(1.0);
    let normalized_error = (actual - expected).abs() / scale;
    *maximum = maximum.max(normalized_error);
    assert_close(actual, expected, tolerance, context);
}

pub fn public_model_oracle_receipt(
    case: &OracleCase,
    lease: &CpuLease,
    artifact_dir: &Path,
    tolerance: f64,
) -> OracleParityReceipt {
    assert!(tolerance.is_finite() && tolerance > 0.0);
    let oracle = fit_oracle(case);
    let train = feature_frame(&case.train_features);
    let oos = feature_frame(&case.oos_features);
    let mut model = BayesianLogitExpert::new();
    model.prior_precision = PRIOR_PRECISION;
    model.learning_rate = LEARNING_RATE;
    model.epochs = EPOCHS;
    ExpertModel::fit(&mut model, &train, &case.train_labels, lease)
        .unwrap_or_else(|error| panic!("{} public fit failed: {error:#}", case.name));
    let probabilities = ExpertModel::predict_proba(&model, &oos, lease)
        .unwrap_or_else(|error| panic!("{} public predict failed: {error:#}", case.name));
    fs::create_dir_all(artifact_dir).unwrap_or_else(|error| {
        panic!(
            "create parent-owned {} oracle artifact directory {}: {error}",
            case.name,
            artifact_dir.display()
        )
    });
    ExpertModel::save(&model, artifact_dir)
        .unwrap_or_else(|error| panic!("{} public save failed: {error:#}", case.name));
    let model_bytes =
        fs::read(artifact_dir.join("model.json")).expect("read genuine public Bayesian artifact");
    let artifact: SavedArtifact =
        serde_json::from_slice(&model_bytes).expect("parse genuine public Bayesian artifact");

    assert_eq!(artifact.precision_schema, "neoethos.bayesian_logit.f64.v2");
    assert_eq!(artifact.model_name, "bayes_logit");
    assert_eq!(artifact.classes.len(), 3);
    assert_eq!(artifact.scaler.means.len(), case.train_features.ncols());

    let mut posterior_values = Vec::new();
    let mut oracle_posterior_values = Vec::new();
    let mut max_normalized_oracle_error = 0.0_f64;
    for (index, (actual, expected)) in artifact.scaler.means.iter().zip(&oracle.means).enumerate() {
        observe_oracle_value(
            *actual,
            *expected,
            tolerance,
            &format!("{} mean {index}", case.name),
            &mut max_normalized_oracle_error,
        );
        posterior_values.push(*actual);
        oracle_posterior_values.push(*expected);
    }
    for (index, (actual, expected)) in artifact.scaler.stds.iter().zip(&oracle.stds).enumerate() {
        observe_oracle_value(
            *actual,
            *expected,
            tolerance,
            &format!("{} std {index}", case.name),
            &mut max_normalized_oracle_error,
        );
        posterior_values.push(*actual);
        oracle_posterior_values.push(*expected);
    }
    for (class, (actual, expected)) in artifact.classes.iter().zip(&oracle.classes).enumerate() {
        assert_eq!(actual.weights.len(), expected.weights.len());
        for (index, (left, right)) in actual.weights.iter().zip(&expected.weights).enumerate() {
            observe_oracle_value(
                *left,
                *right,
                tolerance,
                &format!("{} class {class} weight {index}", case.name),
                &mut max_normalized_oracle_error,
            );
            posterior_values.push(*left);
            oracle_posterior_values.push(*right);
        }
        observe_oracle_value(
            actual.bias,
            expected.bias,
            tolerance,
            &format!("{} class {class} bias", case.name),
            &mut max_normalized_oracle_error,
        );
        posterior_values.push(actual.bias);
        oracle_posterior_values.push(expected.bias);
        assert_eq!(actual.covariance.dim(), expected.covariance.dim());
        for row in 0..actual.covariance.nrows() {
            assert!(actual.covariance[(row, row)] > 0.0);
            for column in 0..actual.covariance.ncols() {
                let left = actual.covariance[(row, column)];
                let right = expected.covariance[(row, column)];
                observe_oracle_value(
                    left,
                    right,
                    tolerance,
                    &format!("{} class {class} covariance ({row},{column})", case.name),
                    &mut max_normalized_oracle_error,
                );
                posterior_values.push(left);
                oracle_posterior_values.push(right);
            }
        }
    }
    assert_eq!(probabilities.dim(), oracle.oos_probabilities.dim());
    assert!(!probabilities.is_empty());
    for (index, (actual, expected)) in probabilities
        .iter()
        .zip(oracle.oos_probabilities.iter())
        .enumerate()
    {
        observe_oracle_value(
            *actual,
            *expected,
            tolerance,
            &format!("{} OOS probability {index}", case.name),
            &mut max_normalized_oracle_error,
        );
    }

    OracleParityReceipt {
        case_name: case.name.to_string(),
        train_feature_sha256: hash_f64_matrix(&case.train_features),
        train_label_sha256: hash_labels(&case.train_labels),
        oos_feature_sha256: hash_f64_matrix(&case.oos_features),
        runtime_backend: artifact.runtime_backend,
        model_artifact_sha256: format!("{:x}", Sha256::digest(model_bytes)),
        posterior_values,
        oracle_posterior_values,
        probability_rows: probabilities.nrows(),
        probability_columns: probabilities.ncols(),
        probability_values: probabilities.iter().copied().collect(),
        oracle_probability_values: oracle.oos_probabilities.iter().copied().collect(),
        max_normalized_oracle_error,
    }
}

pub fn assert_public_cpu_matches_oracle(cases: &[OracleCase]) {
    let mut settings = neoethos_core::Settings::default();
    settings.models.statistical_device = "cpu".to_string();
    install_statistical_runtime_from_settings(&settings);

    let width = WorkerLimit::new(1).expect("one worker is a legal oracle budget");
    let broker = CpuPermitBroker::new(width);
    for case in cases {
        let lease = broker
            .acquire(CpuPermitRequest::local(width))
            .expect("acquire isolated oracle lease");
        let artifact_dir = ArtifactDir::create(case.name);
        let receipt = public_model_oracle_receipt(case, &lease, artifact_dir.path(), 1.0e-8);
        assert!(receipt.runtime_backend.contains("cpu"));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EvidenceDimensions {
    pub train_rows: usize,
    pub feature_columns: usize,
    pub oos_rows: usize,
    pub classes: usize,
}

impl EvidenceDimensions {
    pub fn minimum_host_to_device_bytes(self) -> u64 {
        let train_features = self
            .train_rows
            .checked_mul(self.feature_columns)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
            .expect("training feature byte count must fit usize");
        let labels = self
            .train_rows
            .checked_mul(std::mem::size_of::<i32>())
            .expect("label byte count must fit usize");
        let oos_features = self
            .oos_rows
            .checked_mul(self.feature_columns)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
            .expect("OOS feature byte count must fit usize");
        u64::try_from(train_features + labels + oos_features)
            .expect("host-to-device byte count must fit u64")
    }

    pub fn minimum_device_to_host_bytes(self) -> u64 {
        let probability_bytes = self
            .oos_rows
            .checked_mul(self.classes)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
            .expect("probability byte count must fit usize");
        u64::try_from(probability_bytes).expect("device-to-host byte count must fit u64")
    }

    pub fn minimum_grid_blocks(self) -> u64 {
        let training = self.train_rows.div_ceil(256);
        let inference = self.oos_rows.div_ceil(256);
        u64::try_from(training + inference).expect("grid block count must fit u64")
    }
}

#[derive(Debug, Clone)]
pub struct KernelActivity {
    pub name: String,
    pub start_ns: u64,
    pub end_ns: u64,
    pub grid_blocks: u64,
}

impl KernelActivity {
    pub fn new(name: impl Into<String>, start_ns: u64, end_ns: u64, grid_blocks: u64) -> Self {
        Self {
            name: name.into(),
            start_ns,
            end_ns,
            grid_blocks,
        }
    }

    fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
}

#[derive(Debug, Clone)]
pub struct TransferActivity {
    pub direction: TransferDirection,
    pub start_ns: u64,
    pub end_ns: u64,
    pub bytes: u64,
}

impl TransferActivity {
    pub fn new(direction: TransferDirection, start_ns: u64, end_ns: u64, bytes: u64) -> Self {
        Self {
            direction,
            start_ns,
            end_ns,
            bytes,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidatedCudaEvidence {
    pub named_stage_count: usize,
    pub total_kernel_duration_ns: u64,
    pub total_grid_blocks: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
}

pub fn validate_cuda_evidence(
    runtime_backend: &str,
    dimensions: EvidenceDimensions,
    kernels: &[KernelActivity],
    transfers: &[TransferActivity],
) -> Result<ValidatedCudaEvidence, Vec<String>> {
    let mut errors = Vec::new();
    let backend = runtime_backend.to_ascii_lowercase();
    if !backend.contains("cuda") || backend.contains("cpu") || backend.contains("fallback") {
        errors.push(format!(
            "native CUDA backend required, received `{runtime_backend}`"
        ));
    }

    let stage_requirements = [
        ("preprocessing", &["preprocess", "scaler"][..]),
        ("MAP update", &["map_update", "gradient_update"][..]),
        ("Hessian", &["hessian"][..]),
        ("Cholesky", &["cholesky"][..]),
        ("inference", &["inference", "predict"][..]),
    ];
    let mut named_stage_count = 0usize;
    let mut used_kernel_indices = BTreeSet::new();
    for (stage, tokens) in stage_requirements {
        let found = kernels.iter().enumerate().find(|(index, kernel)| {
            let name = kernel.name.to_ascii_lowercase();
            !used_kernel_indices.contains(index)
                && name.contains("neoethos_bayesian_")
                && tokens.iter().any(|token| name.contains(token))
                && kernel.duration_ns() >= MINIMUM_STAGE_DURATION_NS
                && kernel.grid_blocks > 0
        });
        if let Some((index, _)) = found {
            used_kernel_indices.insert(index);
            named_stage_count += 1;
        } else {
            errors.push(format!(
                "missing distinct named Bayesian {stage} kernel activity with at least {MINIMUM_STAGE_DURATION_NS}ns"
            ));
        }
    }

    let total_kernel_duration_ns = used_kernel_indices
        .iter()
        .map(|index| kernels[*index].duration_ns())
        .fold(0u64, u64::saturating_add);
    if total_kernel_duration_ns < MINIMUM_KERNEL_DURATION_NS {
        errors.push(format!(
            "meaningful kernel duration required: {total_kernel_duration_ns}ns < {MINIMUM_KERNEL_DURATION_NS}ns"
        ));
    }
    let total_grid_blocks = used_kernel_indices
        .iter()
        .map(|index| kernels[*index].grid_blocks)
        .fold(0u64, u64::saturating_add);
    if total_grid_blocks < dimensions.minimum_grid_blocks() {
        errors.push(format!(
            "meaningful grid work required: {total_grid_blocks} blocks < {}",
            dimensions.minimum_grid_blocks()
        ));
    }

    let host_to_device_bytes = transfers
        .iter()
        .filter(|transfer| transfer.direction == TransferDirection::HostToDevice)
        .map(|transfer| transfer.bytes)
        .fold(0u64, u64::saturating_add);
    if host_to_device_bytes < dimensions.minimum_host_to_device_bytes() {
        errors.push(format!(
            "host-to-device bytes {host_to_device_bytes} < dimension-bound minimum {}",
            dimensions.minimum_host_to_device_bytes()
        ));
    }
    let device_to_host_bytes = transfers
        .iter()
        .filter(|transfer| transfer.direction == TransferDirection::DeviceToHost)
        .map(|transfer| transfer.bytes)
        .fold(0u64, u64::saturating_add);
    if device_to_host_bytes < dimensions.minimum_device_to_host_bytes() {
        errors.push(format!(
            "device-to-host bytes {device_to_host_bytes} < dimension-bound minimum {}",
            dimensions.minimum_device_to_host_bytes()
        ));
    }
    for (index, transfer) in transfers.iter().enumerate() {
        if transfer.end_ns <= transfer.start_ns {
            errors.push(format!("transfer {index} has zero or negative duration"));
        }
    }

    if errors.is_empty() {
        Ok(ValidatedCudaEvidence {
            named_stage_count,
            total_kernel_duration_ns,
            total_grid_blocks,
            host_to_device_bytes,
            device_to_host_bytes,
        })
    } else {
        Err(errors)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TimingReceipt {
    pub warmup_ns: u64,
    pub raw_sample_ns: Vec<u64>,
    pub median_ns: u64,
}

impl TimingReceipt {
    pub fn new(warmup: Duration, samples: [Duration; TIMED_SAMPLES]) -> Result<Self, String> {
        Self::from_slice(warmup, &samples)
    }

    pub fn from_slice(warmup: Duration, samples: &[Duration]) -> Result<Self, String> {
        if samples.len() != TIMED_SAMPLES {
            return Err(format!(
                "exactly three timed samples required, received {}",
                samples.len()
            ));
        }
        let warmup_ns = u64::try_from(warmup.as_nanos())
            .map_err(|_| "warm-up duration does not fit u64 nanoseconds".to_string())?;
        let raw_sample_ns = samples
            .iter()
            .map(|duration| {
                u64::try_from(duration.as_nanos())
                    .map_err(|_| "timed duration does not fit u64 nanoseconds".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if raw_sample_ns.iter().any(|duration| *duration == 0) {
            return Err("timed samples must be non-zero".to_string());
        }
        let mut sorted = raw_sample_ns.clone();
        sorted.sort_unstable();
        Ok(Self {
            warmup_ns,
            raw_sample_ns,
            median_ns: sorted[sorted.len() / 2],
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GitIdentity {
    pub commit: String,
    pub tree: String,
}

impl GitIdentity {
    pub fn parse(
        commit_output: &str,
        tree_output: &str,
        status_output: &str,
    ) -> Result<Self, String> {
        if !status_output.trim().is_empty() {
            return Err(format!(
                "tested implementation tree is dirty: {}",
                status_output.trim()
            ));
        }
        let commit = commit_output.trim().to_ascii_lowercase();
        let tree = tree_output.trim().to_ascii_lowercase();
        if !is_git_object_id(&commit) {
            return Err(format!(
                "commit must be a 40- or 64-digit hex Git object, got `{commit}`"
            ));
        }
        if !is_git_object_id(&tree) {
            return Err(format!(
                "tree must be a 40- or 64-digit hex Git object, got `{tree}`"
            ));
        }
        Ok(Self { commit, tree })
    }
}

fn is_git_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn hash_f64_matrix(matrix: &Array2<f64>) -> String {
    hash_f64_values(matrix.nrows(), matrix.ncols(), matrix.iter().copied())
}

pub fn hash_f64_values(
    rows: usize,
    columns: usize,
    values: impl IntoIterator<Item = f64>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.bayesian-r5.f64-matrix.v1\0");
    hasher.update(u64::try_from(rows).expect("rows fit u64").to_le_bytes());
    hasher.update(
        u64::try_from(columns)
            .expect("columns fit u64")
            .to_le_bytes(),
    );
    let mut count = 0usize;
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
        count = count.checked_add(1).expect("f64 value count overflow");
    }
    assert_eq!(
        count,
        rows.checked_mul(columns)
            .expect("matrix element count overflow"),
        "f64 hash iterator length must match its declared shape"
    );
    format!("{:x}", hasher.finalize())
}

pub fn hash_labels(labels: &[i32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.bayesian-r5.i32-labels.v1\0");
    hasher.update(
        u64::try_from(labels.len())
            .expect("label count fits u64")
            .to_le_bytes(),
    );
    for label in labels {
        hasher.update(label.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}
