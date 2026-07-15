use anyhow::{Result, bail};
use ndarray::Array2;
use neoethos_core::storage::json::{JsonBackupWriteConfig, write_json_with_backup};
use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::capabilities::{CapabilityState, ModelFamily};

pub const OPTIMIZATION_REPORT_FILE_NAME: &str = "optimization.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationMetrics {
    pub objective_score: f64,
    pub log_loss: f64,
    pub accuracy: f64,
    pub confident_accuracy: f64,
    pub confidence_coverage: f64,
    pub mean_confidence: f64,
    pub rows_evaluated: usize,
    pub confident_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationTrialRecord {
    pub index: usize,
    pub backend: String,
    pub params: HashMap<String, String>,
    pub metrics: Option<ValidationMetrics>,
    pub error: Option<String>,
    pub selected: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub model_name: String,
    pub capability_family: ModelFamily,
    pub capability_state: CapabilityState,
    pub backend: String,
    pub trials_requested: usize,
    pub trials_completed: usize,
    pub holdout_pct: f64,
    pub train_rows: usize,
    pub val_rows: usize,
    pub selected_trial_index: usize,
    pub selected_params: HashMap<String, String>,
    pub selected_metrics: Option<ValidationMetrics>,
    pub row_budget_applied: Option<usize>,
    pub hpo_rows_applied: Option<usize>,
    pub notes: Vec<String>,
    pub trials: Vec<OptimizationTrialRecord>,
}

pub type HoldoutSplit = (DataFrame, Vec<i32>, DataFrame, Vec<i32>);

fn validate_optimization_report(report: &OptimizationReport) -> Result<()> {
    if report.model_name.trim().is_empty() {
        bail!("optimization report model_name must not be empty");
    }
    if report.backend.trim().is_empty() {
        bail!("optimization report backend must not be empty");
    }
    if report.trials_requested == 0 {
        bail!("optimization report trials_requested must be non-zero");
    }
    if report.trials_completed > report.trials_requested {
        bail!("optimization report trials_completed must not exceed trials_requested");
    }
    if !(0.0..1.0).contains(&report.holdout_pct) {
        bail!("optimization report holdout_pct must be inside [0, 1)");
    }
    if report.train_rows == 0 {
        bail!("optimization report train_rows must be non-zero");
    }
    if report.train_rows + report.val_rows == 0 {
        bail!("optimization report total rows must be non-zero");
    }
    if report.selected_trial_index >= report.trials_requested {
        bail!("optimization report selected_trial_index is out of range");
    }
    if report.trials.is_empty() {
        bail!("optimization report must contain at least one trial record");
    }
    if report.trials.iter().all(|trial| !trial.selected) {
        bail!("optimization report must mark one trial as selected");
    }
    if report.trials.iter().filter(|trial| trial.selected).count() != 1 {
        bail!("optimization report must mark exactly one selected trial");
    }
    Ok(())
}

fn label_to_probability_index(label: i32) -> Result<usize> {
    match label {
        0 => Ok(0),
        1 => Ok(1),
        -1 => Ok(2),
        other => bail!("unsupported label {other}; expected -1, 0, 1"),
    }
}

pub fn time_series_holdout_split(
    frame: &DataFrame,
    labels: &[i32],
    holdout_pct: f64,
    embargo_rows: usize,
    min_train_rows: usize,
    min_val_rows: usize,
) -> Result<Option<HoldoutSplit>> {
    if frame.height() != labels.len() {
        bail!(
            "holdout split row/label mismatch: {} rows vs {} labels",
            frame.height(),
            labels.len()
        );
    }
    if frame.height()
        < min_train_rows
            .saturating_add(min_val_rows)
            .saturating_add(embargo_rows)
    {
        return Ok(None);
    }

    let holdout_pct = holdout_pct.clamp(0.05, 0.45);
    let val_rows = ((frame.height() as f64) * holdout_pct).round() as usize;
    let val_rows = val_rows
        .max(min_val_rows)
        .min(frame.height().saturating_sub(min_train_rows));
    if val_rows == 0 {
        return Ok(None);
    }

    let embargo_adjusted_height = frame.height().saturating_sub(embargo_rows);
    if embargo_adjusted_height < min_train_rows + min_val_rows {
        tracing::debug!(
            target: "hpo",
            "embargo {embargo_rows} consumes too much of frame {}",
            frame.height()
        );
        return Ok(None);
    }
    let train_end = frame
        .height()
        .saturating_sub(val_rows)
        .saturating_sub(embargo_rows);
    if train_end < min_train_rows {
        return Ok(None);
    }

    let val_start = train_end.saturating_add(embargo_rows);
    let val_len = frame.height().saturating_sub(val_start);
    if val_len < min_val_rows {
        return Ok(None);
    }

    let train_frame = frame.slice(0, train_end);
    let train_labels = labels.iter().take(train_end).copied().collect::<Vec<_>>();
    let val_frame = frame.slice(val_start as i64, val_len);
    let val_labels = labels
        .iter()
        .skip(val_start)
        .take(val_len)
        .copied()
        .collect::<Vec<_>>();

    if train_frame.height() != train_labels.len() || val_frame.height() != val_labels.len() {
        bail!("holdout split produced inconsistent frame/label sizes");
    }

    Ok(Some((train_frame, train_labels, val_frame, val_labels)))
}

pub fn evaluate_prediction_quality(
    probabilities: &Array2<f32>,
    labels: &[i32],
    confidence_threshold: f32,
    metric_weight: f64,
    accuracy_weight: f64,
) -> Result<ValidationMetrics> {
    if probabilities.nrows() != labels.len() {
        bail!(
            "prediction evaluation row mismatch: {} rows vs {} labels",
            probabilities.nrows(),
            labels.len()
        );
    }
    if probabilities.ncols() != 3 {
        bail!(
            "prediction evaluation expected 3 class columns, got {}",
            probabilities.ncols()
        );
    }
    if labels.is_empty() {
        bail!("prediction evaluation requires at least one label");
    }

    let mut log_loss = 0.0_f64;
    let mut correct = 0_usize;
    let mut confident_correct = 0_usize;
    let mut confident_rows = 0_usize;
    let mut confidence_sum = 0.0_f64;

    for (row_idx, label) in labels.iter().copied().enumerate() {
        let true_idx = label_to_probability_index(label)?;
        let row = probabilities.row(row_idx);
        let mut max_idx = 0_usize;
        let mut max_prob = f32::NEG_INFINITY;
        let mut row_sum = 0.0_f32;
        for (col_idx, value) in row.iter().copied().enumerate() {
            if !value.is_finite() || value < 0.0 {
                bail!("prediction row {row_idx} contains invalid probability {value}");
            }
            row_sum += value;
            if value > max_prob {
                max_prob = value;
                max_idx = col_idx;
            }
        }

        let true_prob = row[true_idx].clamp(1e-6, 1.0);
        let normalizer = if row_sum.is_finite() && row_sum > 1e-6 {
            row_sum as f64
        } else {
            1.0
        };
        log_loss -= ((true_prob as f64) / normalizer).ln();
        confidence_sum += max_prob.max(0.0) as f64;

        if max_idx == true_idx {
            correct += 1;
        }
        if max_prob >= confidence_threshold {
            confident_rows += 1;
            if max_idx == true_idx {
                confident_correct += 1;
            }
        }
    }

    let rows = labels.len();
    let accuracy = correct as f64 / rows as f64;
    let confident_accuracy = if confident_rows == 0 {
        accuracy
    } else {
        confident_correct as f64 / confident_rows as f64
    };
    let confidence_coverage = confident_rows as f64 / rows as f64;
    let mean_confidence = confidence_sum / rows as f64;
    let log_loss = log_loss / rows as f64;

    let metric_weight = metric_weight.max(0.0);
    let accuracy_weight = accuracy_weight.clamp(0.0, 1.0);
    let confident_component = confident_accuracy * confidence_coverage;
    let objective_score = -(metric_weight * log_loss)
        + (accuracy_weight * accuracy)
        + ((1.0 - accuracy_weight) * confident_component);

    Ok(ValidationMetrics {
        objective_score,
        log_loss,
        accuracy,
        confident_accuracy,
        confidence_coverage,
        mean_confidence,
        rows_evaluated: rows,
        confident_rows,
    })
}

pub fn write_optimization_report(path: &Path, report: &OptimizationReport) -> Result<()> {
    validate_optimization_report(report)?;
    write_json_with_backup(
        path,
        report,
        JsonBackupWriteConfig {
            artifact_label: "optimization report",
            temp_extension: "tmp_optimization",
            backup_extension: "bak_optimization",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report() -> OptimizationReport {
        OptimizationReport {
            model_name: "lightgbm".to_string(),
            capability_family: ModelFamily::Tree,
            capability_state: CapabilityState::Implemented,
            backend: "lightgbm".to_string(),
            trials_requested: 4,
            trials_completed: 4,
            holdout_pct: 0.2,
            train_rows: 800,
            val_rows: 200,
            selected_trial_index: 1,
            selected_params: HashMap::from([("learning_rate".to_string(), "0.05".to_string())]),
            selected_metrics: None,
            row_budget_applied: Some(900),
            hpo_rows_applied: Some(900),
            notes: Vec::new(),
            trials: vec![
                OptimizationTrialRecord {
                    index: 0,
                    backend: "lightgbm".to_string(),
                    params: HashMap::new(),
                    metrics: None,
                    error: None,
                    selected: false,
                },
                OptimizationTrialRecord {
                    index: 1,
                    backend: "lightgbm".to_string(),
                    params: HashMap::new(),
                    metrics: None,
                    error: None,
                    selected: true,
                },
            ],
        }
    }

    #[test]
    fn optimization_report_rejects_empty_trials() {
        // Audit B10: the old small-data path emitted trials: vec![] — a
        // structurally invalid report. Confirm validation rejects it, so a
        // regression can't silently ship one again.
        let mut report = sample_report();
        report.trials = vec![];
        report.trials_completed = 0;
        report.selected_trial_index = 0;
        let err = validate_optimization_report(&report)
            .expect_err("empty trials must fail")
            .to_string();
        assert!(err.contains("at least one trial"), "got: {err}");
    }

    #[test]
    fn optimization_report_no_hpo_single_base_trial_is_valid() {
        // The B10 replacement shape: one selected base trial with no metrics
        // (HPO skipped for lack of data) must be a VALID report.
        let report = OptimizationReport {
            trials_completed: 0,
            selected_trial_index: 0,
            selected_metrics: None,
            trials: vec![OptimizationTrialRecord {
                index: 0,
                backend: "lightgbm".to_string(),
                params: HashMap::new(),
                metrics: None,
                error: Some("dataset too small for holdout-driven HPO".to_string()),
                selected: true,
            }],
            ..sample_report()
        };
        validate_optimization_report(&report).expect("no-HPO base-trial report must be valid");
    }

    #[test]
    fn optimization_report_rejects_blank_backend() {
        let mut report = sample_report();
        report.backend = " ".to_string();
        let err = validate_optimization_report(&report)
            .expect_err("blank backend must fail")
            .to_string();
        assert!(err.contains("backend"));
    }

    #[test]
    fn optimization_report_requires_single_selected_trial() {
        let mut report = sample_report();
        report.trials[0].selected = true;
        let err = validate_optimization_report(&report)
            .expect_err("multiple selected trials must fail")
            .to_string();
        assert!(err.contains("exactly one selected"));
    }
}
