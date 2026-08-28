use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Axis};
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
use serde::{Deserialize, Serialize};
use std::path::Path;

use neoethos_core::BackendKind;
use neoethos_core::storage::json::{DirBackupWriteConfig, write_dir_with_backup};

use crate::base::{
    ExpertModel, build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    three_class_runtime_confidence, try_build_runtime_artifact_metadata,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily, runtime_backend_kind_from_label};
use crate::runtime::prediction::RuntimePrediction;

use super::common::statistical_device_policy;
use super::common::{
    FeatureScaler, METADATA_FILE_NAME, MODEL_FILE_NAME, cpu_backend_for_policy,
    ensure_feature_columns_match, feature_matrix_from_frame, read_json, remap_three_class_labels,
    softmax_rows, write_json,
};
#[cfg(feature = "statistical-gpu")]
use super::linear_gpu::{try_fit_linear_softmax_cuda, try_predict_linear_softmax_cuda};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearSoftmaxArtifact {
    precision_schema: String,
    weights: Array2<f64>,
    bias: Array1<f64>,
    scaler: FeatureScaler,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_metadata: Option<RuntimeArtifactMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_backend_kind: Option<BackendKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_degraded_reason: Option<String>,
    requested_device_policy: String,
    effective_device_policy: String,
    alpha: f64,
    l1_ratio: f64,
    learning_rate: f64,
    epochs: usize,
    model_name: String,
}

const LINEAR_F64_SCHEMA: &str = "neoethos.linear_softmax.f64.v2";

fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Soft-threshold proximal operator for the L1 penalty:
/// `prox_{t|·|}(w) = sign(w)·max(|w|−t, 0)`. Drives small coefficients to EXACT
/// zero — the defining ElasticNet/Lasso sparsity that a subgradient cannot give.
fn soft_threshold(w: f64, t: f64) -> f64 {
    sign(w) * (w.abs() - t).max(0.0)
}

fn split_train_val_indices(rows: usize) -> (Vec<usize>, Vec<usize>) {
    if rows <= 6 {
        return ((0..rows).collect(), Vec::new());
    }

    let val_rows = ((rows as f64) * 0.2).round() as usize;
    let val_rows = val_rows.clamp(1, rows.saturating_sub(2));
    let embargo_rows = if rows >= 20 {
        ((rows as f64) * 0.02).round() as usize
    } else {
        0
    };
    let embargo_rows = embargo_rows.clamp(0, rows.saturating_sub(val_rows + 1));
    let train_rows = rows.saturating_sub(val_rows + embargo_rows);

    if train_rows == 0 {
        return ((0..rows).collect(), Vec::new());
    }

    let train = (0..train_rows).collect::<Vec<_>>();
    let val = (train_rows + embargo_rows..rows).collect::<Vec<_>>();
    (train, val)
}

fn resolved_linear_device_policy(model_name: &str, requested: &str) -> Result<String> {
    #[cfg(feature = "statistical-gpu")]
    let _ = model_name;
    match crate::common::resolve_cuda_device_policy(
        requested,
        crate::tree_models::config::nvidia_gpu_count(),
    )? {
        crate::common::ResolvedCudaDevicePolicy::Cpu => Ok("cpu".to_string()),
        crate::common::ResolvedCudaDevicePolicy::Cuda { ordinal } => {
            #[cfg(feature = "statistical-gpu")]
            {
                Ok(format!("gpu:{ordinal}"))
            }
            #[cfg(not(feature = "statistical-gpu"))]
            {
                bail!(
                    "{model_name} resolved CUDA ordinal {ordinal} from policy `{requested}`, but this binary was built without `statistical-gpu`"
                )
            }
        }
    }
}

/// GROUP E remediation 2026-05-25: 5 hand-rolled functions replaced
/// with a single delegation to the canonical `write_dir_with_backup`
/// helper in `neoethos-core::storage::json`. Saves ~60 LOC of duplicate
/// staged-tmp+backup logic (this file is one of 4).
fn with_staged_linear_artifact_dir<F>(path: &Path, writer: F) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    write_dir_with_backup(
        path,
        DirBackupWriteConfig {
            artifact_label: "linear artifact",
            temp_extension: "tmp_linear_artifact",
            backup_extension: "bak_linear_artifact",
        },
        writer,
    )
}

fn logits_from_features(
    features: &Array2<f64>,
    weights: &Array2<f64>,
    bias: &Array1<f64>,
) -> Result<Array2<f64>> {
    if features.ncols() != weights.nrows() {
        bail!(
            "feature dimension mismatch: {} features vs {} weights",
            features.ncols(),
            weights.nrows()
        );
    }
    if weights.ncols() != bias.len() {
        bail!(
            "class dimension mismatch: {} weights cols vs {} bias terms",
            weights.ncols(),
            bias.len()
        );
    }

    let mut logits = features.dot(weights);
    for row in 0..logits.nrows() {
        for class_idx in 0..bias.len() {
            logits[(row, class_idx)] += bias[class_idx];
        }
    }
    if logits.iter().any(|value| !value.is_finite()) {
        bail!("linear model produced non-finite logits");
    }

    Ok(logits)
}

fn cross_entropy_loss(probabilities: &Array2<f64>, labels: &[usize]) -> Result<f64> {
    if probabilities.nrows() != labels.len() {
        bail!(
            "validation label mismatch: {} rows vs {} labels",
            probabilities.nrows(),
            labels.len()
        );
    }

    let mut loss = 0.0_f64;
    for (row_idx, class_idx) in labels.iter().copied().enumerate() {
        let probability = probabilities[(row_idx, class_idx)].clamp(1e-6, 1.0 - 1e-6);
        loss -= probability.ln();
    }

    Ok(loss / labels.len().max(1) as f64)
}

fn normalize_linear_softmax_params(
    alpha: f64,
    l1_ratio: f64,
    learning_rate: f64,
    epochs: usize,
) -> Result<(f64, f64, f64, usize)> {
    if !alpha.is_finite() {
        bail!("linear model alpha must be finite");
    }
    if !l1_ratio.is_finite() {
        bail!("linear model l1_ratio must be finite");
    }
    if !learning_rate.is_finite() {
        bail!("linear model learning_rate must be finite");
    }
    Ok((
        alpha.max(0.0),
        l1_ratio.clamp(0.0, 1.0),
        learning_rate.max(1e-5),
        epochs.max(1),
    ))
}

fn runtime_metadata(
    model_name: &str,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    train_rows: usize,
    val_rows: usize,
) -> Result<RuntimeArtifactMetadata> {
    try_build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Meta,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new_unchecked(dataset_rows, train_rows, val_rows),
    )
}

fn runtime_predictions(
    model_name: &str,
    probabilities: &Array2<f64>,
    execution_backend: Option<String>,
    degraded_reason: Option<String>,
) -> Result<Vec<RuntimePrediction>> {
    let mut predictions = Vec::with_capacity(probabilities.nrows());
    for row in probabilities.outer_iter() {
        let row_values = [row[0], row[1], row[2]];
        let (confidence, abstain_recommended) = three_class_runtime_confidence(row_values)?;
        let reason = degraded_reason.clone().or_else(|| {
            abstain_recommended
                .then(|| "shared three-class confidence gate recommended abstain".to_string())
        });
        predictions.push(build_runtime_prediction_with_details(
            model_name,
            ModelFamily::Meta,
            CapabilityState::Implemented,
            row_values,
            Some(confidence),
            Some(abstain_recommended),
            execution_backend.clone(),
            reason,
        )?);
    }

    Ok(predictions)
}

struct LinearSoftmaxPrediction {
    probabilities: Array2<f64>,
    execution_backend: Option<String>,
    degraded_reason: Option<String>,
}

fn validate_runtime_metadata(
    metadata: &RuntimeArtifactMetadata,
    expected_model_name: &str,
    expected_feature_columns: &[String],
    expected_dataset_rows: usize,
) -> Result<()> {
    if expected_feature_columns.is_empty() {
        bail!("persisted {expected_model_name} artifact is missing feature columns");
    }
    if metadata.model_name != expected_model_name {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected model name {expected_model_name}, got {}",
            metadata.model_name
        );
    }
    if metadata.family != ModelFamily::Meta {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected family {:?}, got {:?}",
            ModelFamily::Meta,
            metadata.family
        );
    }
    if metadata.state != CapabilityState::Implemented {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected state {:?}, got {:?}",
            CapabilityState::Implemented,
            metadata.state
        );
    }
    if metadata.label_mapping != canonical_three_class_label_mapping() {
        bail!("runtime metadata mismatch for {expected_model_name}: unexpected label mapping");
    }
    if metadata.feature_columns != expected_feature_columns {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected feature columns {:?}, got {:?}",
            expected_feature_columns,
            metadata.feature_columns
        );
    }
    if metadata.training_summary.dataset_rows != expected_dataset_rows {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected {} dataset rows, got {}",
            expected_dataset_rows,
            metadata.training_summary.dataset_rows
        );
    }
    if metadata.training_summary.train_rows == 0 {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: training rows must be non-zero"
        );
    }
    if metadata.training_summary.train_rows + metadata.training_summary.val_rows
        != metadata.training_summary.dataset_rows
    {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: training rows {} + validation rows {} must equal dataset rows {}",
            metadata.training_summary.train_rows,
            metadata.training_summary.val_rows,
            metadata.training_summary.dataset_rows
        );
    }

    Ok(())
}

fn resolve_runtime_metadata_from_artifact(
    path: &Path,
    model_name: &str,
    artifact: &LinearSoftmaxArtifact,
) -> Result<RuntimeArtifactMetadata> {
    let metadata_path = path.join(METADATA_FILE_NAME);
    match read_json::<RuntimeArtifactMetadata>(&metadata_path) {
        Ok(metadata) => {
            validate_runtime_metadata(
                &metadata,
                model_name,
                &artifact.feature_columns,
                artifact.dataset_rows,
            )
            .with_context(|| {
                format!(
                    "runtime metadata sidecar mismatch with embedded {} metadata at {}",
                    model_name,
                    metadata_path.display()
                )
            })?;
            if let Some(embedded) = artifact.runtime_metadata.as_ref()
                && (embedded.model_name != metadata.model_name
                    || embedded.family != metadata.family
                    || embedded.state != metadata.state
                    || embedded.feature_columns != metadata.feature_columns
                    || embedded.label_mapping != metadata.label_mapping
                    || embedded.training_summary.dataset_rows
                        != metadata.training_summary.dataset_rows
                    || embedded.training_summary.train_rows != metadata.training_summary.train_rows
                    || embedded.training_summary.val_rows != metadata.training_summary.val_rows)
            {
                bail!(
                    "runtime metadata sidecar mismatch with embedded {} metadata at {}",
                    model_name,
                    metadata_path.display()
                );
            }
            Ok(metadata)
        }
        Err(file_err) => {
            let fallback = artifact
                .runtime_metadata
                .clone()
                .with_context(|| format!("missing runtime metadata file {} and artifact has no embedded metadata: {file_err}", metadata_path.display()))?;
            validate_runtime_metadata(
                &fallback,
                model_name,
                &artifact.feature_columns,
                artifact.dataset_rows,
            )?;
            tracing::warn!(
                path = %metadata_path.display(),
                error = %file_err,
                model = %model_name,
                "linear model metadata sidecar missing/unreadable; using embedded runtime metadata"
            );
            Ok(fallback)
        }
    }
}

fn validate_linear_artifact(artifact: &LinearSoftmaxArtifact) -> Result<()> {
    if artifact.precision_schema != LINEAR_F64_SCHEMA {
        bail!(
            "unsupported linear artifact precision schema {}; expected {LINEAR_F64_SCHEMA}",
            artifact.precision_schema
        );
    }
    if artifact.model_name != "elasticnet" && artifact.model_name != "logistic" {
        bail!(
            "unexpected linear artifact model name {}",
            artifact.model_name
        );
    }
    if artifact.feature_columns.is_empty() {
        bail!("linear artifact must contain at least one feature column");
    }
    if artifact.weights.nrows() != artifact.feature_columns.len() {
        bail!(
            "linear artifact feature-column mismatch: {} weights rows vs {} feature columns",
            artifact.weights.nrows(),
            artifact.feature_columns.len()
        );
    }
    if artifact.weights.ncols() != 3 || artifact.bias.len() != 3 {
        bail!(
            "linear artifact must persist exactly three classes, found {} weight columns and {} bias terms",
            artifact.weights.ncols(),
            artifact.bias.len()
        );
    }
    if artifact.scaler.means.len() != artifact.feature_columns.len()
        || artifact.scaler.stds.len() != artifact.feature_columns.len()
    {
        bail!(
            "linear artifact scaler dimension mismatch: means {}, stds {}, features {}",
            artifact.scaler.means.len(),
            artifact.scaler.stds.len(),
            artifact.feature_columns.len()
        );
    }
    if artifact.runtime_metadata.is_none() {
        bail!("linear artifact must persist runtime metadata");
    }
    validate_runtime_metadata(
        artifact
            .runtime_metadata
            .as_ref()
            .expect("checked runtime metadata presence"),
        &artifact.model_name,
        &artifact.feature_columns,
        artifact.dataset_rows,
    )?;
    let requested_device = crate::common::parse_cuda_device_policy(
        &artifact.requested_device_policy,
    )
    .with_context(|| {
        format!(
            "linear artifact has invalid requested device policy `{}`",
            artifact.requested_device_policy
        )
    })?;
    let effective_label = artifact.effective_device_policy.trim().to_ascii_lowercase();
    let effective_device = crate::common::parse_cuda_device_policy(
        &artifact.effective_device_policy,
    )
    .with_context(|| {
        format!(
            "linear artifact has invalid effective device policy `{}`",
            artifact.effective_device_policy
        )
    })?;
    let effective_cuda_ordinal = match effective_device {
        crate::common::CudaDevicePolicy::Cpu if effective_label == "cpu" => None,
        crate::common::CudaDevicePolicy::Gpu { ordinal }
            if effective_label == format!("gpu:{ordinal}") =>
        {
            Some(ordinal)
        }
        _ => bail!(
            "linear artifact effective device must be `cpu` or an exact `gpu:<ordinal>`, got `{}`",
            artifact.effective_device_policy
        ),
    };
    match (requested_device, effective_cuda_ordinal) {
        (crate::common::CudaDevicePolicy::Cpu, Some(_)) => {
            bail!("linear artifact requested CPU but recorded CUDA execution")
        }
        (crate::common::CudaDevicePolicy::Gpu { .. }, None) => {
            bail!("linear artifact requested explicit CUDA but recorded CPU execution")
        }
        (crate::common::CudaDevicePolicy::Gpu { ordinal: requested }, Some(recorded))
            if requested != recorded =>
        {
            bail!(
                "linear artifact CUDA ordinal mismatch: requested {requested}, recorded {recorded}"
            )
        }
        (crate::common::CudaDevicePolicy::Auto, Some(recorded)) if recorded != 0 => {
            bail!("linear Auto artifact must record CUDA ordinal 0, got {recorded}")
        }
        _ => {}
    }
    let backend_is_cuda = artifact
        .runtime_backend
        .as_deref()
        .is_some_and(|backend| backend.contains("cuda"));
    if backend_is_cuda != effective_cuda_ordinal.is_some() {
        bail!(
            "linear artifact runtime backend {:?} is inconsistent with effective device `{}`",
            artifact.runtime_backend,
            artifact.effective_device_policy
        );
    }
    if effective_cuda_ordinal.is_some()
        && artifact.runtime_backend_kind != Some(BackendKind::NativeCuda)
    {
        bail!("linear CUDA artifact must record runtime_backend_kind=NativeCuda");
    }
    if !artifact.alpha.is_finite() || artifact.alpha < 0.0 {
        bail!("linear artifact alpha must be finite and non-negative");
    }
    if !artifact.l1_ratio.is_finite() || !(0.0..=1.0).contains(&artifact.l1_ratio) {
        bail!("linear artifact l1_ratio must be finite and inside [0, 1]");
    }
    if !artifact.learning_rate.is_finite() || artifact.learning_rate <= 0.0 {
        bail!("linear artifact learning_rate must be finite and positive");
    }
    if artifact.epochs == 0 {
        bail!("linear artifact epochs must be positive");
    }
    if artifact.weights.iter().any(|value| !value.is_finite())
        || artifact.bias.iter().any(|value| !value.is_finite())
        || artifact.scaler.means.iter().any(|value| !value.is_finite())
        || artifact.scaler.stds.iter().any(|value| !value.is_finite())
        || artifact.scaler.stds.iter().any(|value| *value <= 0.0)
    {
        bail!("linear artifact contains non-finite parameters");
    }
    if artifact.dataset_rows == 0 {
        bail!("linear artifact must persist a non-zero dataset row count");
    }

    Ok(())
}

fn validate_linear_artifact_device_for_load(artifact: &LinearSoftmaxArtifact) -> Result<()> {
    let resolved =
        resolved_linear_device_policy(&artifact.model_name, &artifact.requested_device_policy)?;
    let requested = crate::common::parse_cuda_device_policy(&artifact.requested_device_policy)?;
    let auto_cpu_relocation = matches!(requested, crate::common::CudaDevicePolicy::Auto)
        && artifact.effective_device_policy.starts_with("gpu:")
        && resolved == "cpu";
    if !auto_cpu_relocation && resolved != artifact.effective_device_policy {
        bail!(
            "linear runtime device drift on load: recorded `{}`, resolved `{}` from policy `{}`",
            artifact.effective_device_policy,
            resolved,
            artifact.requested_device_policy
        );
    }
    Ok(())
}

fn fit_linear_softmax(
    model_name: &str,
    x: &FeatureFrame,
    y: &[i32],
    alpha: f64,
    l1_ratio: f64,
    learning_rate: f64,
    epochs: usize,
) -> Result<LinearSoftmaxArtifact> {
    let (alpha, l1_ratio, learning_rate, epochs) =
        normalize_linear_softmax_params(alpha, l1_ratio, learning_rate, epochs)?;
    let (features, feature_columns) = feature_matrix_from_frame(x)?;
    let rows = features.nrows();
    let cols = features.ncols();
    let n_classes = 3usize;

    if y.len() != rows {
        bail!(
            "{model_name} requires matching feature and label rows: {} features vs {} labels",
            rows,
            y.len()
        );
    }

    let labels = remap_three_class_labels(y)?;

    if rows == 0 || cols == 0 {
        bail!("{model_name} requires a non-empty feature matrix");
    }

    let (train_indices, val_indices) = split_train_val_indices(rows);
    let train_labels = train_indices
        .iter()
        .map(|idx| labels[*idx])
        .collect::<Vec<_>>();
    let val_labels = val_indices
        .iter()
        .map(|idx| labels[*idx])
        .collect::<Vec<_>>();
    let train_features = features.select(Axis(0), &train_indices);
    let val_features = if val_indices.is_empty() {
        None
    } else {
        Some(features.select(Axis(0), &val_indices))
    };

    let scaler = FeatureScaler::fit(&train_features)?;
    let train_features = scaler.transform(&train_features)?;
    let val_features = if let Some(val_features) = val_features {
        Some(scaler.transform(&val_features)?)
    } else {
        None
    };

    let requested_device_policy = statistical_device_policy(model_name);
    let resolved_device_policy =
        resolved_linear_device_policy(model_name, &requested_device_policy)?;
    #[cfg(feature = "statistical-gpu")]
    if crate::common::cuda_kernel_enabled(&resolved_device_policy)? {
        let cuda_fit = try_fit_linear_softmax_cuda(
            model_name,
            &resolved_device_policy,
            &train_features,
            &train_labels,
            val_features.as_ref(),
            &val_labels,
            alpha,
            l1_ratio,
            learning_rate,
            epochs,
        )
        .with_context(|| format!("fit {model_name} through the required statistical CUDA lane"))?;
        let train_rows = train_labels.len();
        let val_rows = val_labels.len();
        let runtime_metadata = runtime_metadata(
            model_name,
            feature_columns.clone(),
            rows,
            train_rows,
            val_rows,
        )?;
        return Ok(LinearSoftmaxArtifact {
            precision_schema: LINEAR_F64_SCHEMA.to_string(),
            weights: cuda_fit.weights,
            bias: cuda_fit.bias,
            scaler,
            feature_columns,
            dataset_rows: rows,
            runtime_metadata: Some(runtime_metadata),
            runtime_backend: Some(cuda_fit.runtime_backend),
            runtime_backend_kind: Some(cuda_fit.runtime_backend_kind),
            runtime_degraded_reason: None,
            requested_device_policy,
            effective_device_policy: resolved_device_policy,
            alpha,
            l1_ratio,
            learning_rate,
            epochs,
            model_name: model_name.to_string(),
        });
    }

    let cpu_backend = cpu_backend_for_policy(
        &resolved_device_policy,
        &format!("{model_name}_softmax_cpu"),
    )?;
    let runtime_backend = Some(cpu_backend);
    let runtime_degraded_reason = None;

    let mut weights = Array2::<f64>::zeros((cols, n_classes));
    let mut bias = Array1::<f64>::zeros(n_classes);
    let lr = learning_rate.max(1e-5);
    let regularization = alpha.max(0.0);
    let mut best_weights = weights.clone();
    let mut best_bias = bias.clone();
    let mut best_val_loss = f64::INFINITY;
    let mut stale_epochs = 0usize;
    let patience = 25usize;

    for _ in 0..epochs.max(1) {
        let logits = logits_from_features(&train_features, &weights, &bias)?;
        let probabilities = softmax_rows(&logits)?;
        let mut error = probabilities;
        for (row_idx, class_idx) in train_labels.iter().copied().enumerate() {
            error[(row_idx, class_idx)] -= 1.0;
        }

        let mut grad_w = train_features.t().dot(&error) / train_features.nrows() as f64;
        let grad_b = error.sum_axis(Axis(0)) / train_features.nrows() as f64;

        // ElasticNet via PROXIMAL gradient (ISTA): the SMOOTH L2 (ridge) term goes
        // in the gradient; the NON-smooth L1 (lasso) term is applied by the
        // soft-threshold proximal operator AFTER the gradient step (below), which
        // drives coefficients to EXACT zero — the sparsity the previous
        // subgradient-L1 (`grad += sign(w)`) could never produce.
        let l1r = l1_ratio.clamp(0.0, 1.0);
        if regularization > 0.0 {
            for feature_idx in 0..cols {
                for class_idx in 0..n_classes {
                    let l2 = (1.0 - l1r) * weights[(feature_idx, class_idx)];
                    grad_w[(feature_idx, class_idx)] += regularization * l2;
                }
            }
        }

        for feature_idx in 0..cols {
            for class_idx in 0..n_classes {
                weights[(feature_idx, class_idx)] -= lr * grad_w[(feature_idx, class_idx)];
            }
        }
        for class_idx in 0..n_classes {
            bias[class_idx] -= lr * grad_b[class_idx];
        }

        // Proximal soft-threshold for L1 (the ISTA prox step). Threshold
        // t = lr·alpha·l1_ratio; the bias is NOT regularised (standard).
        let threshold = lr * regularization * l1r;
        if threshold > 0.0 {
            for feature_idx in 0..cols {
                for class_idx in 0..n_classes {
                    weights[(feature_idx, class_idx)] =
                        soft_threshold(weights[(feature_idx, class_idx)], threshold);
                }
            }
        }

        if let Some(val_features) = val_features.as_ref() {
            let val_logits = logits_from_features(val_features, &weights, &bias)?;
            let val_probabilities = softmax_rows(&val_logits)?;
            let val_loss = cross_entropy_loss(&val_probabilities, &val_labels)?;
            if val_loss + 1e-6 < best_val_loss {
                best_val_loss = val_loss;
                best_weights = weights.clone();
                best_bias = bias.clone();
                stale_epochs = 0;
            } else {
                stale_epochs += 1;
                if stale_epochs >= patience {
                    break;
                }
            }
        }
    }

    if best_val_loss.is_finite() {
        weights = best_weights;
        bias = best_bias;
    }

    let train_rows = train_labels.len();
    let val_rows = val_labels.len();
    let runtime_metadata = runtime_metadata(
        model_name,
        feature_columns.clone(),
        rows,
        train_rows,
        val_rows,
    )?;

    Ok(LinearSoftmaxArtifact {
        precision_schema: LINEAR_F64_SCHEMA.to_string(),
        weights,
        bias,
        scaler,
        feature_columns,
        dataset_rows: rows,
        runtime_metadata: Some(runtime_metadata),
        runtime_backend_kind: runtime_backend_kind_from_label(runtime_backend.as_deref()),
        runtime_backend,
        runtime_degraded_reason,
        requested_device_policy,
        effective_device_policy: resolved_device_policy,
        alpha,
        l1_ratio,
        learning_rate,
        epochs,
        model_name: model_name.to_string(),
    })
}

fn predict_linear_softmax_with_runtime(
    artifact: &LinearSoftmaxArtifact,
    x: &FeatureFrame,
) -> Result<LinearSoftmaxPrediction> {
    ensure_feature_columns_match(&artifact.feature_columns, x)?;
    let (features, _) = feature_matrix_from_frame(x)?;
    let features = artifact.scaler.transform(&features)?;

    let cpu_backend = format!("{}_softmax_cpu", artifact.model_name);
    let resolved_device_policy =
        resolved_linear_device_policy(&artifact.model_name, &artifact.requested_device_policy)?;
    #[cfg(feature = "statistical-gpu")]
    if crate::common::cuda_kernel_enabled(&resolved_device_policy)? {
        let probabilities = try_predict_linear_softmax_cuda(
            &artifact.model_name,
            &resolved_device_policy,
            &features,
            &artifact.weights,
            &artifact.bias,
        )
        .with_context(|| {
            format!(
                "predict {} through the required statistical CUDA lane",
                artifact.model_name
            )
        })?;
        let execution_backend = artifact
            .runtime_backend
            .as_ref()
            .filter(|backend| backend.contains("cuda"))
            .cloned()
            .unwrap_or_else(|| {
                format!(
                    "{}_softmax_cuda[{resolved_device_policy}]",
                    artifact.model_name
                )
            });
        return Ok(LinearSoftmaxPrediction {
            probabilities,
            execution_backend: Some(execution_backend),
            degraded_reason: None,
        });
    }

    let execution_backend = Some(cpu_backend_for_policy(
        &resolved_device_policy,
        &cpu_backend,
    )?);
    let auto_cpu_relocation = artifact.effective_device_policy.starts_with("gpu:")
        && matches!(
            crate::common::parse_cuda_device_policy(&artifact.requested_device_policy)?,
            crate::common::CudaDevicePolicy::Auto
        )
        && resolved_device_policy == "cpu";
    if artifact
        .runtime_backend
        .as_deref()
        .is_some_and(|backend| backend.contains("cuda"))
        && !auto_cpu_relocation
    {
        bail!(
            "persisted CUDA statistical artifact cannot execute through the CpuOnly f64 implementation"
        );
    }
    let degraded_reason = if auto_cpu_relocation {
        Some(
            "statistical Auto CUDA artifact relocated to CPU because no NVIDIA device is visible"
                .to_string(),
        )
    } else {
        artifact.runtime_degraded_reason.clone()
    };

    let logits = logits_from_features(&features, &artifact.weights, &artifact.bias)?;
    Ok(LinearSoftmaxPrediction {
        probabilities: softmax_rows(&logits)?,
        execution_backend,
        degraded_reason,
    })
}

fn predict_linear_softmax(
    artifact: &LinearSoftmaxArtifact,
    x: &FeatureFrame,
) -> Result<Array2<f64>> {
    Ok(predict_linear_softmax_with_runtime(artifact, x)?.probabilities)
}

pub struct ElasticNetExpert {
    model: Option<LinearSoftmaxArtifact>,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub learning_rate: f64,
    pub epochs: usize,
}

impl ElasticNetExpert {
    /// Read-only view of the trained feature column names + ordering.
    /// Returns an empty slice when the model has not been trained
    /// or loaded yet. Required by the
    /// [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        match &self.model {
            Some(m) => &m.feature_columns,
            None => &[],
        }
    }

    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            model: None,
            alpha,
            l1_ratio,
            learning_rate: 0.05,
            epochs: 300,
        }
    }

    pub fn ranked_feature_importance(&self) -> Result<Vec<(String, f64)>> {
        let model = self
            .model
            .as_ref()
            .context("ElasticNetExpert not trained")?;

        let mut ranked = model
            .feature_columns
            .iter()
            .enumerate()
            .map(|(feature_idx, name)| {
                let importance = model
                    .weights
                    .row(feature_idx)
                    .iter()
                    .map(|weight| weight.abs())
                    .sum::<f64>()
                    / model.weights.ncols().max(1) as f64;
                (name.clone(), importance)
            })
            .collect::<Vec<_>>();

        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        Ok(ranked)
    }

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        lease.scope(|| self.predict_runtime_scoped(x))
    }

    fn predict_runtime_scoped(&self, x: &FeatureFrame) -> Result<Vec<RuntimePrediction>> {
        let model = self
            .model
            .as_ref()
            .context("ElasticNetExpert not trained")?;
        let runtime_metadata = model
            .runtime_metadata
            .as_ref()
            .context("ElasticNetExpert runtime metadata missing")?;
        validate_runtime_metadata(
            runtime_metadata,
            &model.model_name,
            &model.feature_columns,
            model.dataset_rows,
        )?;
        let prediction = predict_linear_softmax_with_runtime(model, x)?;
        runtime_predictions(
            &model.model_name,
            &prediction.probabilities,
            prediction.execution_backend,
            prediction.degraded_reason,
        )
    }
}

impl ExpertModel for ElasticNetExpert {
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        lease.scope(|| {
            self.model = Some(fit_linear_softmax(
                "elasticnet",
                x,
                y,
                self.alpha,
                self.l1_ratio,
                self.learning_rate,
                self.epochs,
            )?);
            Ok(())
        })
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        lease.scope(|| {
            let model = self
                .model
                .as_ref()
                .context("ElasticNetExpert not trained")?;
            predict_linear_softmax(model, x)
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let model = self
            .model
            .as_ref()
            .context("ElasticNetExpert not trained")?;
        validate_linear_artifact(model)?;
        let runtime_metadata = model
            .runtime_metadata
            .as_ref()
            .context("ElasticNetExpert artifact missing runtime metadata")?;
        validate_runtime_metadata(
            runtime_metadata,
            "elasticnet",
            &model.feature_columns,
            model.dataset_rows,
        )?;
        with_staged_linear_artifact_dir(path, |staged_path| {
            write_json(&staged_path.join(MODEL_FILE_NAME), model)?;
            write_json(&staged_path.join(METADATA_FILE_NAME), &runtime_metadata)
        })
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let mut model: LinearSoftmaxArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        if model.model_name != "elasticnet" {
            bail!("expected elasticnet artifact, got {}", model.model_name);
        }
        validate_linear_artifact(&model)?;
        validate_linear_artifact_device_for_load(&model)?;
        let runtime_metadata = resolve_runtime_metadata_from_artifact(path, "elasticnet", &model)?;
        model.runtime_metadata = Some(runtime_metadata);
        self.alpha = model.alpha;
        self.l1_ratio = model.l1_ratio;
        self.learning_rate = model.learning_rate;
        self.epochs = model.epochs;
        self.model = Some(model);
        Ok(())
    }
}

pub struct LogisticExpert {
    model: Option<LinearSoftmaxArtifact>,
    pub alpha: f64,
    pub learning_rate: f64,
    pub epochs: usize,
}

impl LogisticExpert {
    pub fn new() -> Self {
        Self {
            model: None,
            alpha: 0.01,
            learning_rate: 0.05,
            epochs: 250,
        }
    }

    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        match &self.model {
            Some(m) => &m.feature_columns,
            None => &[],
        }
    }
}

impl Default for LogisticExpert {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertModel for LogisticExpert {
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        lease.scope(|| {
            self.model = Some(fit_linear_softmax(
                "logistic",
                x,
                y,
                self.alpha,
                0.0,
                self.learning_rate,
                self.epochs,
            )?);
            Ok(())
        })
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        lease.scope(|| {
            let model = self.model.as_ref().context("LogisticExpert not trained")?;
            predict_linear_softmax(model, x)
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let model = self.model.as_ref().context("LogisticExpert not trained")?;
        validate_linear_artifact(model)?;
        let runtime_metadata = model
            .runtime_metadata
            .as_ref()
            .context("LogisticExpert artifact missing runtime metadata")?;
        validate_runtime_metadata(
            runtime_metadata,
            "logistic",
            &model.feature_columns,
            model.dataset_rows,
        )?;
        with_staged_linear_artifact_dir(path, |staged_path| {
            write_json(&staged_path.join(MODEL_FILE_NAME), model)?;
            write_json(&staged_path.join(METADATA_FILE_NAME), &runtime_metadata)
        })
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let mut model: LinearSoftmaxArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        if model.model_name != "logistic" {
            bail!("expected logistic artifact, got {}", model.model_name);
        }
        validate_linear_artifact(&model)?;
        validate_linear_artifact_device_for_load(&model)?;
        let runtime_metadata = resolve_runtime_metadata_from_artifact(path, "logistic", &model)?;
        model.runtime_metadata = Some(runtime_metadata);
        self.alpha = model.alpha;
        self.learning_rate = model.learning_rate;
        self.epochs = model.epochs;
        self.model = Some(model);
        Ok(())
    }
}

impl LogisticExpert {
    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        lease.scope(|| self.predict_runtime_scoped(x))
    }

    fn predict_runtime_scoped(&self, x: &FeatureFrame) -> Result<Vec<RuntimePrediction>> {
        let model = self.model.as_ref().context("LogisticExpert not trained")?;
        let runtime_metadata = model
            .runtime_metadata
            .as_ref()
            .context("LogisticExpert runtime metadata missing")?;
        validate_runtime_metadata(
            runtime_metadata,
            &model.model_name,
            &model.feature_columns,
            model.dataset_rows,
        )?;
        let prediction = predict_linear_softmax_with_runtime(model, x)?;
        runtime_predictions(
            &model.model_name,
            &prediction.probabilities,
            prediction.execution_backend,
            prediction.degraded_reason,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::three_class_runtime_confidence;
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64};
    use neoethos_execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};

    #[test]
    fn soft_threshold_produces_exact_zeros_and_shrinks() {
        // The L1 proximal operator: |w| <= t -> EXACTLY 0 (the ElasticNet sparsity
        // a subgradient cannot produce); |w| > t -> shrunk toward 0 by t.
        assert_eq!(soft_threshold(0.2, 0.3), 0.0);
        assert_eq!(soft_threshold(-0.2, 0.3), 0.0);
        assert_eq!(soft_threshold(0.0, 0.3), 0.0);
        assert!((soft_threshold(0.5, 0.3) - 0.2).abs() < 1e-6);
        assert!((soft_threshold(-0.5, 0.3) + 0.2).abs() < 1e-6);
    }

    fn sample_frame() -> FeatureFrame {
        let values = [
            ("open", vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
            ("high", vec![1.2, 1.3, 1.4, 1.5, 1.6, 1.7]),
            ("low", vec![0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),
            ("close", vec![1.05, 1.15, 1.25, 1.35, 1.45, 1.55]),
        ];
        let columns = values
            .into_iter()
            .map(|(name, values)| {
                FeatureColumnF64::new(name, values, vec![FeatureCellValidity::Valid; 6])
                    .expect("valid sample feature")
            })
            .collect();
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(6),
            columns,
        )
        .expect("sample feature frame")
    }

    fn sample_labels() -> Vec<i32> {
        vec![-1, 0, 1, -1, 0, 1]
    }

    fn one_worker_lease() -> CpuLease {
        let width = WorkerLimit::new(1).expect("one worker");
        CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .expect("isolated model test lease")
    }

    #[test]
    fn logistic_expert_rejects_label_row_mismatch() {
        let frame = sample_frame();
        let y = vec![-1, 0, 1];
        let lease = one_worker_lease();
        let mut model = LogisticExpert::new();

        let err = model
            .fit(&frame, &y, &lease)
            .expect_err("mismatched labels should fail");
        assert!(err.to_string().contains("matching feature and label rows"));
    }

    #[test]
    fn logistic_expert_trains_and_persists_runtime_metadata() -> Result<()> {
        let frame = sample_frame();
        let y = sample_labels();
        let lease = one_worker_lease();
        let mut model = LogisticExpert::new();

        model.fit(&frame, &y, &lease)?;

        let artifact = model.model.as_ref().expect("trained model");
        let metadata = artifact
            .runtime_metadata
            .as_ref()
            .expect("runtime metadata to be recorded");

        assert_eq!(metadata.model_name, "logistic");
        assert_eq!(metadata.family, ModelFamily::Meta);
        assert_eq!(metadata.state, CapabilityState::Implemented);
        assert_eq!(metadata.training_summary.dataset_rows, 6);
        assert_eq!(
            metadata.training_summary.train_rows + metadata.training_summary.val_rows,
            6
        );
        assert_eq!(artifact.runtime_backend_kind, Some(BackendKind::NativeCpu));

        let runtime_predictions = model.predict_runtime(&frame, &lease)?;
        assert_eq!(runtime_predictions.len(), 6);
        let prediction_metadata = runtime_predictions[0].metadata();
        assert_eq!(
            prediction_metadata.backend_kind,
            Some(BackendKind::NativeCpu)
        );
        let expected_runtime_mode = if prediction_metadata.degraded_reason.is_some() {
            neoethos_core::RuntimeMode::Degraded
        } else {
            neoethos_core::RuntimeMode::Canonical
        };
        assert_eq!(
            prediction_metadata.runtime_mode,
            Some(expected_runtime_mode)
        );
        assert_eq!(
            prediction_metadata.runtime_degraded_reason.is_some(),
            prediction_metadata.degraded_reason.is_some()
        );
        Ok(())
    }

    #[test]
    fn elasticnet_runtime_predictions_validate_probability_contract() -> Result<()> {
        let frame = sample_frame();
        let y = sample_labels();
        let lease = one_worker_lease();
        let mut model = ElasticNetExpert::new(0.01, 0.5);
        model.fit(&frame, &y, &lease)?;

        let probabilities = model.predict_proba(&frame, &lease)?;
        assert_eq!(probabilities.ncols(), 3);

        let runtime_predictions = model.predict_runtime(&frame, &lease)?;
        assert_eq!(runtime_predictions.len(), 6);
        Ok(())
    }

    #[test]
    fn elasticnet_l1_drives_weights_to_exact_zero() -> Result<()> {
        // End-to-end CPU-policy proof that ElasticNet is genuinely real for
        // l1>0: pure L1 (l1_ratio=1.0) uses the canonical ISTA proximal path,
        // which must pin at least one coefficient to EXACTLY 0.0 while
        // predictions stay a valid simplex. The CUDA policy has the same
        // proximal update and is covered by the real-device statistical gate.
        let frame = sample_frame();
        let y = sample_labels();
        let lease = one_worker_lease();
        let mut model = ElasticNetExpert::new(0.5, 1.0);
        model.fit(&frame, &y, &lease)?;
        let artifact = model.model.as_ref().expect("trained ElasticNet artifact");
        assert!(
            artifact.weights.iter().any(|w| *w == 0.0),
            "pure-L1 ElasticNet must zero at least one coefficient exactly: {:?}",
            artifact.weights
        );
        let probabilities = model.predict_proba(&frame, &lease)?;
        for row in probabilities.outer_iter() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "probability row must sum to 1, got {sum}"
            );
        }
        Ok(())
    }

    #[test]
    fn runtime_predictions_use_shared_three_class_confidence_gate() -> Result<()> {
        let probabilities = Array2::from_shape_vec((1, 3), vec![0.58_f64, 0.20, 0.22])?;
        let predictions = runtime_predictions(
            "logistic",
            &probabilities,
            Some("logistic_softmax_cpu".to_string()),
            None,
        )?;
        let prediction = predictions
            .first()
            .expect("one runtime prediction should be produced");
        let (expected_confidence, expected_abstain) =
            three_class_runtime_confidence([0.58, 0.20, 0.22])?;

        assert!((prediction.confidence().expect("confidence") - expected_confidence).abs() < 1e-6);
        assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
        Ok(())
    }

    #[test]
    fn runtime_predictions_persist_linear_backend_details() -> Result<()> {
        let probabilities = Array2::from_shape_vec((1, 3), vec![0.58_f64, 0.20, 0.22])?;
        let prediction = runtime_predictions(
            "logistic",
            &probabilities,
            Some("logistic_softmax_cpu".to_string()),
            None,
        )?
        .into_iter()
        .next()
        .expect("one runtime prediction");

        assert_eq!(
            prediction.metadata().execution_backend.as_deref(),
            Some("logistic_softmax_cpu")
        );
        Ok(())
    }

    #[test]
    fn split_train_val_indices_leaves_temporal_embargo_gap() {
        let (train, val) = split_train_val_indices(50);
        assert!(!val.is_empty(), "validation split should be present");
        let last_train = *train.last().expect("train rows");
        let first_val = *val.first().expect("val rows");
        assert!(
            first_val > last_train + 1,
            "expected embargo gap between train and val"
        );
    }

    #[test]
    fn validate_linear_artifact_rejects_missing_runtime_metadata() {
        let artifact = LinearSoftmaxArtifact {
            precision_schema: LINEAR_F64_SCHEMA.to_string(),
            weights: Array2::zeros((2, 3)),
            bias: Array1::zeros(3),
            scaler: FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            },
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 8,
            runtime_metadata: None,
            runtime_backend: Some("logistic_softmax_cpu".to_string()),
            runtime_backend_kind: Some(BackendKind::NativeCpu),
            runtime_degraded_reason: None,
            requested_device_policy: "cpu".to_string(),
            effective_device_policy: "cpu".to_string(),
            alpha: 0.01,
            l1_ratio: 0.0,
            learning_rate: 0.05,
            epochs: 64,
            model_name: "logistic".to_string(),
        };

        let err = validate_linear_artifact(&artifact)
            .expect_err("artifact without runtime metadata should fail");
        assert!(err.to_string().contains("runtime metadata"));
    }

    #[test]
    fn validate_linear_artifact_rejects_non_positive_scaler_stds() {
        let artifact = LinearSoftmaxArtifact {
            precision_schema: LINEAR_F64_SCHEMA.to_string(),
            weights: Array2::zeros((2, 3)),
            bias: Array1::zeros(3),
            scaler: FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 0.0],
            },
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 8,
            runtime_metadata: Some(
                runtime_metadata(
                    "logistic",
                    vec!["f1".to_string(), "f2".to_string()],
                    8,
                    6,
                    2,
                )
                .expect("build metadata"),
            ),
            runtime_backend: Some("logistic_softmax_cpu".to_string()),
            runtime_backend_kind: Some(BackendKind::NativeCpu),
            runtime_degraded_reason: None,
            requested_device_policy: "cpu".to_string(),
            effective_device_policy: "cpu".to_string(),
            alpha: 0.01,
            l1_ratio: 0.0,
            learning_rate: 0.05,
            epochs: 64,
            model_name: "logistic".to_string(),
        };

        let err = validate_linear_artifact(&artifact)
            .expect_err("artifact with non-positive scaler std should fail");
        assert!(err.to_string().contains("non-finite parameters"));
    }

    #[test]
    fn validate_runtime_metadata_rejects_zero_train_rows() {
        let metadata = RuntimeArtifactMetadata {
            schema_version: crate::runtime::artifacts::RUNTIME_ARTIFACT_METADATA_SCHEMA_VERSION,
            model_name: "logistic".to_string(),
            family: ModelFamily::Meta,
            state: CapabilityState::Implemented,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            label_mapping: canonical_three_class_label_mapping(),
            training_summary: TrainingSummaryMetadata::raw_for_validation(8, 0, 8),
        };

        let err = validate_runtime_metadata(
            &metadata,
            "logistic",
            &["f1".to_string(), "f2".to_string()],
            8,
        )
        .expect_err("zero train rows must fail");
        assert!(err.to_string().contains("training rows must be non-zero"));
    }

    #[test]
    fn logistic_load_uses_embedded_runtime_metadata_when_metadata_file_missing() -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let frame = sample_frame();
        let y = sample_labels();
        let lease = one_worker_lease();
        let mut model = LogisticExpert::new();
        model.fit(&frame, &y, &lease)?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir =
            std::env::temp_dir().join(format!("neoethos-models-logistic-embed-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;
        std::fs::remove_file(artifact_dir.join(METADATA_FILE_NAME)).expect("remove metadata file");

        let mut reloaded = LogisticExpert::new();
        reloaded.load(&artifact_dir)?;
        assert!(reloaded.model.is_some());

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }

    #[test]
    fn logistic_expert_rejects_tampered_metadata_on_load() -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let frame = sample_frame();
        let y = sample_labels();
        let lease = one_worker_lease();
        let mut model = LogisticExpert::new();
        model.fit(&frame, &y, &lease)?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir = std::env::temp_dir().join(format!("neoethos-models-logistic-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;

        let metadata_path = artifact_dir.join(METADATA_FILE_NAME);
        let mut metadata: RuntimeArtifactMetadata =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("parse metadata");
        metadata.model_name = "tampered".to_string();
        std::fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
        )
        .expect("write tampered metadata");

        let mut reloaded = LogisticExpert::new();
        let err = reloaded
            .load(&artifact_dir)
            .expect_err("tampered metadata should fail");
        assert!(err.to_string().contains("runtime metadata"));

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }

    #[test]
    fn logistic_load_rejects_sidecar_drift_against_embedded_metadata() -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let frame = sample_frame();
        let y = sample_labels();
        let lease = one_worker_lease();
        let mut model = LogisticExpert::new();
        model.fit(&frame, &y, &lease)?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir =
            std::env::temp_dir().join(format!("neoethos-models-logistic-sidecar-drift-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;

        let metadata_path = artifact_dir.join(METADATA_FILE_NAME);
        let mut metadata: RuntimeArtifactMetadata =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("parse metadata");
        metadata.training_summary.dataset_rows += 1;
        std::fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
        )
        .expect("write drifted metadata");

        let mut reloaded = LogisticExpert::new();
        let err = reloaded
            .load(&artifact_dir)
            .expect_err("sidecar drift should fail load");
        assert!(err.to_string().contains("sidecar mismatch"));

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }
}
