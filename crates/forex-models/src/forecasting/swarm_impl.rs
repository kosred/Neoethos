use anyhow::{Context, Result, bail};
use polars::prelude::{DataFrame, DataType, Series};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "swarm-forecasting")]
use ruv_swarm_ml::agent_forecasting::{AgentForecastingManager, ForecastRequirements};
#[cfg(feature = "swarm-forecasting")]
use ruv_swarm_ml::ensemble::{
    EnsembleConfig, EnsembleForecaster, EnsembleModel, EnsembleStrategy, ModelPerformanceMetrics,
    OptimizationMetric,
};
#[cfg(feature = "swarm-forecasting")]
use ruv_swarm_ml::models::ModelType;
#[cfg(feature = "swarm-forecasting")]
use ruv_swarm_ml::time_series::{SeasonalityInfo, TimeSeriesData, TimeSeriesProcessor};

const SWARM_ARTIFACT_FILE_NAME: &str = "swarm_forecaster.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmEnsembleStrategy {
    SimpleAverage,
    WeightedAverage,
    Median,
    TrimmedMean,
    BayesianModelAveraging,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmForecastConfig {
    pub memory_limit_mb: f32,
    pub agent_id: String,
    pub agent_type: String,
    pub frequency: String,
    pub horizon: usize,
    pub accuracy_target: f32,
    pub latency_requirement_ms: f32,
    pub interpretability_needed: bool,
    pub online_learning: bool,
    pub strategy: SwarmEnsembleStrategy,
}

impl Default for SwarmForecastConfig {
    fn default() -> Self {
        Self {
            memory_limit_mb: 256.0,
            agent_id: "swarm_forecaster".to_string(),
            agent_type: "analyst".to_string(),
            frequency: "H".to_string(),
            horizon: 24,
            accuracy_target: 0.90,
            latency_requirement_ms: 200.0,
            interpretability_needed: false,
            online_learning: true,
            strategy: SwarmEnsembleStrategy::BayesianModelAveraging,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmForecastSnapshot {
    pub last_value: f32,
    pub rolling_mean: f32,
    pub drift_slope: f32,
    pub volatility: f32,
    pub has_trend: bool,
    pub has_seasonality: bool,
    pub seasonal_periods: Vec<usize>,
    #[serde(default)]
    pub short_mean: f32,
    #[serde(default)]
    pub medium_mean: f32,
    #[serde(default)]
    pub long_mean: f32,
    #[serde(default)]
    pub recent_return: f32,
    #[serde(default)]
    pub trend_strength: f32,
    #[serde(default)]
    pub mean_reversion_strength: f32,
    #[serde(default)]
    pub volatility_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmForecastResult {
    pub point_forecast: Vec<f32>,
    pub level_80_lower: Vec<f32>,
    pub level_80_upper: Vec<f32>,
    pub diversity_score: f32,
    pub effective_models: f32,
    pub prediction_variance: f32,
    pub models_used: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SwarmForecasterArtifact {
    config: SwarmForecastConfig,
    runtime_mode: SwarmRuntimeMode,
    #[serde(default)]
    runtime_degraded_reason: Option<String>,
    fitted: bool,
    values: Vec<f32>,
    timestamps: Vec<f64>,
    unique_id: String,
    snapshot: Option<SwarmForecastSnapshot>,
    last_result: Option<SwarmForecastResult>,
    last_horizon: Option<usize>,
    candidate_reports: Vec<SwarmCandidateReport>,
    updated_at_unix_ms: Option<u64>,
    training_report: Option<SwarmTrainingReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum SwarmRuntimeMode {
    LocalFallback,
    ExternalSwarm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SwarmCandidateReport {
    name: String,
    model_type: String,
    source: String,
    weight: f32,
    prediction_length: usize,
    prediction_mean: f32,
    prediction_std: f32,
    mae: f32,
    mse: f32,
    mape: f32,
    smape: f32,
    coverage: f32,
    #[serde(default)]
    bias: f32,
    #[serde(default)]
    directional_accuracy: f32,
    #[serde(default)]
    regime_fit: f32,
    #[serde(default)]
    stability_score: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SwarmTrainingReport {
    training_rows: usize,
    validation_windows: usize,
    fitted_horizon: usize,
    best_candidate: Option<String>,
    aggregate_mae: f32,
    aggregate_smape: f32,
    aggregate_directional_accuracy: f32,
    aggregate_coverage: f32,
    diversity_score: f32,
    regime_bias: f32,
    updated_at_unix_ms: Option<u64>,
}

fn sanitize_runtime_degraded_reason(reason: Option<String>) -> Option<String> {
    reason.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn sanitize_forecaster_artifact(
    mut artifact: SwarmForecasterArtifact,
) -> Result<SwarmForecasterArtifact> {
    if artifact.values.len() != artifact.timestamps.len() {
        bail!(
            "swarm forecaster value/timestamp mismatch: {} values vs {} timestamps",
            artifact.values.len(),
            artifact.timestamps.len()
        );
    }

    if artifact.values.iter().any(|value| !value.is_finite()) {
        bail!("swarm forecaster artifact contains non-finite values");
    }
    if artifact
        .timestamps
        .iter()
        .any(|timestamp| !timestamp.is_finite())
    {
        bail!("swarm forecaster artifact contains non-finite timestamps");
    }

    artifact.runtime_degraded_reason =
        sanitize_runtime_degraded_reason(artifact.runtime_degraded_reason);

    if !artifact.fitted {
        artifact.runtime_mode = SwarmRuntimeMode::LocalFallback;
        artifact.runtime_degraded_reason = None;
        artifact.snapshot = None;
        artifact.last_result = None;
        artifact.last_horizon = None;
        artifact.candidate_reports.clear();
        artifact.updated_at_unix_ms = None;
        artifact.training_report = None;
        return Ok(artifact);
    }

    if artifact
        .snapshot
        .as_ref()
        .is_none_or(|snapshot| !snapshot_is_valid(snapshot))
    {
        artifact.snapshot = Some(build_local_snapshot_with_min(
            &artifact.values,
            &artifact.timestamps,
            snapshot_rebuild_min_observations(artifact.values.len()),
        )?);
    }

    repair_forecaster_artifact_state(&mut artifact)?;
    Ok(artifact)
}

fn snapshot_is_valid(snapshot: &SwarmForecastSnapshot) -> bool {
    snapshot.last_value.is_finite()
        && snapshot.rolling_mean.is_finite()
        && snapshot.drift_slope.is_finite()
        && snapshot.volatility.is_finite()
        && snapshot.short_mean.is_finite()
        && snapshot.medium_mean.is_finite()
        && snapshot.long_mean.is_finite()
        && snapshot.recent_return.is_finite()
        && snapshot.trend_strength.is_finite()
        && snapshot.mean_reversion_strength.is_finite()
        && snapshot.volatility_ratio.is_finite()
        && snapshot.seasonal_periods.iter().all(|period| *period > 0)
}

fn candidate_reports_are_valid(reports: &[SwarmCandidateReport], horizon: usize) -> bool {
    if reports.is_empty() {
        return false;
    }

    let mut total_weight = 0.0_f32;
    let mut seen_names = HashSet::with_capacity(reports.len());
    for report in reports {
        if report.name.is_empty()
            || report.model_type.is_empty()
            || report.source.is_empty()
            || report.prediction_length == 0
            || report.prediction_length != horizon
            || report.weight <= 0.0
            || !report.weight.is_finite()
            || !report.prediction_mean.is_finite()
            || !report.prediction_std.is_finite()
            || !report.mae.is_finite()
            || !report.mse.is_finite()
            || !report.mape.is_finite()
            || !report.smape.is_finite()
            || !report.coverage.is_finite()
            || !report.bias.is_finite()
            || !report.directional_accuracy.is_finite()
            || !report.regime_fit.is_finite()
            || !report.stability_score.is_finite()
            || !seen_names.insert(report.name.as_str())
        {
            return false;
        }
        total_weight += report.weight;
    }

    total_weight > f32::EPSILON
}

fn training_report_matches(
    report: &SwarmTrainingReport,
    reports: &[SwarmCandidateReport],
    training_rows: usize,
    validation_windows: usize,
    horizon: usize,
) -> bool {
    report.training_rows == training_rows
        && report.validation_windows == validation_windows
        && report.fitted_horizon == horizon
        && report.aggregate_mae.is_finite()
        && report.aggregate_smape.is_finite()
        && report.aggregate_directional_accuracy.is_finite()
        && report.aggregate_coverage.is_finite()
        && report.diversity_score.is_finite()
        && report.regime_bias.is_finite()
        && report
            .best_candidate
            .as_ref()
            .is_none_or(|name| reports.iter().any(|candidate| candidate.name == *name))
}

fn artifact_target_horizon(artifact: &SwarmForecasterArtifact) -> usize {
    artifact
        .last_horizon
        .filter(|horizon| *horizon > 0)
        .or_else(|| {
            artifact
                .training_report
                .as_ref()
                .map(|report| report.fitted_horizon)
                .filter(|horizon| *horizon > 0)
        })
        .unwrap_or_else(|| artifact.config.horizon.max(1))
}

fn repair_forecaster_artifact_state(artifact: &mut SwarmForecasterArtifact) -> Result<()> {
    let snapshot = artifact
        .snapshot
        .clone()
        .context("swarm forecaster artifact missing snapshot after repair")?;
    let horizon = artifact_target_horizon(artifact);
    let validation_windows =
        build_validation_windows(&artifact.values, &artifact.timestamps, horizon);

    let has_valid_state = artifact
        .last_result
        .as_ref()
        .is_some_and(|result| result_is_valid(result, horizon))
        && candidate_reports_are_valid(&artifact.candidate_reports, horizon)
        && artifact.training_report.as_ref().is_some_and(|report| {
            training_report_matches(
                report,
                &artifact.candidate_reports,
                artifact.values.len(),
                validation_windows.len(),
                horizon,
            )
        });
    if has_valid_state {
        artifact.last_horizon = Some(horizon);
        if artifact.runtime_mode == SwarmRuntimeMode::ExternalSwarm {
            artifact.runtime_degraded_reason = None;
        } else {
            artifact.runtime_degraded_reason = Some("swarm_local_fallback_active".to_string());
        }
        return Ok(());
    }

    rebuild_forecaster_artifact_state(artifact, &snapshot, horizon, &validation_windows)
}

#[cfg(feature = "swarm-forecasting")]
fn rebuild_forecaster_artifact_state(
    artifact: &mut SwarmForecasterArtifact,
    snapshot: &SwarmForecastSnapshot,
    horizon: usize,
    validation_windows: &[(Vec<f32>, Vec<f64>, Vec<f32>)],
) -> Result<()> {
    if artifact.runtime_mode == SwarmRuntimeMode::ExternalSwarm {
        let candidates = candidate_predictions(&artifact.values, snapshot, horizon);
        if candidates.len() < 2 {
            artifact.runtime_mode = SwarmRuntimeMode::LocalFallback;
            artifact.runtime_degraded_reason =
                Some("external_swarm_state_unrecoverable".to_string());
            artifact.last_result = None;
            artifact.last_horizon = None;
            artifact.candidate_reports.clear();
            artifact.updated_at_unix_ms = None;
            artifact.training_report = None;
            return Ok(());
        }

        let point_candidates = candidates
            .iter()
            .map(|(_, _, forecast)| forecast.clone())
            .collect::<Vec<_>>();
        let mut reports = if !validation_windows.is_empty() {
            build_weighted_reports_external(
                validation_windows,
                &artifact.config.frequency,
                &artifact.unique_id,
                horizon,
            )?
        } else {
            let reference = aggregate_average(&point_candidates, horizon, snapshot.last_value);
            let mut reports =
                build_candidate_reports(&candidates, &reference, "consensus", Some(snapshot));
            apply_candidate_weights(&mut reports);
            reports
        };
        if reports.is_empty() {
            artifact.runtime_mode = SwarmRuntimeMode::LocalFallback;
            artifact.runtime_degraded_reason =
                Some("external_swarm_state_unrecoverable".to_string());
            artifact.last_result = None;
            artifact.last_horizon = None;
            artifact.candidate_reports.clear();
            artifact.updated_at_unix_ms = None;
            artifact.training_report = None;
            return Ok(());
        }

        normalize_candidate_weights(&mut reports);
        let fallback_candidates = candidates
            .iter()
            .map(|(name, _, forecast)| (name.clone(), forecast.clone()))
            .collect::<Vec<_>>();
        let weight_map = build_candidate_weight_map(&reports);
        let result = fallback_forecast_with_strategy(
            snapshot,
            &fallback_candidates,
            &weight_map,
            &reports,
            artifact.config.strategy,
            horizon,
        );

        artifact.runtime_mode = SwarmRuntimeMode::LocalFallback;
        artifact.runtime_degraded_reason =
            Some("external_swarm_result_rebuilt_from_local_consensus".to_string());
        artifact.candidate_reports = reports;
        artifact.last_result = Some(result.clone());
        artifact.last_horizon = Some(horizon);
        artifact.updated_at_unix_ms = current_unix_ms().or(artifact.updated_at_unix_ms);
        artifact.training_report = Some(build_training_report(
            &artifact.candidate_reports,
            validation_windows.len(),
            artifact.values.len(),
            horizon,
            result.diversity_score,
            snapshot.trend_strength - snapshot.mean_reversion_strength,
        ));
        return Ok(());
    }

    rebuild_forecaster_artifact_state_local(artifact, snapshot, horizon, validation_windows)
}

#[cfg(not(feature = "swarm-forecasting"))]
fn rebuild_forecaster_artifact_state(
    artifact: &mut SwarmForecasterArtifact,
    snapshot: &SwarmForecastSnapshot,
    horizon: usize,
    validation_windows: &[(Vec<f32>, Vec<f64>, Vec<f32>)],
) -> Result<()> {
    rebuild_forecaster_artifact_state_local(artifact, snapshot, horizon, validation_windows)
}

fn rebuild_forecaster_artifact_state_local(
    artifact: &mut SwarmForecasterArtifact,
    snapshot: &SwarmForecastSnapshot,
    horizon: usize,
    validation_windows: &[(Vec<f32>, Vec<f64>, Vec<f32>)],
) -> Result<()> {
    let candidates = candidate_forecasts_local(&artifact.values, snapshot, horizon);
    if candidates.len() < 2 {
        artifact.runtime_mode = SwarmRuntimeMode::LocalFallback;
        artifact.runtime_degraded_reason = Some("swarm_local_fallback_unrecoverable".to_string());
        artifact.last_result = None;
        artifact.last_horizon = None;
        artifact.candidate_reports.clear();
        artifact.updated_at_unix_ms = None;
        artifact.training_report = None;
        return Ok(());
    }

    let report_reference = aggregate_average(
        &candidates
            .iter()
            .map(|(_, forecast)| forecast.clone())
            .collect::<Vec<_>>(),
        horizon,
        snapshot.last_value,
    );
    let mut reports = if !validation_windows.is_empty() {
        build_weighted_reports_local(validation_windows, horizon)?
    } else {
        let mut reports = candidates
            .iter()
            .map(|(name, forecast)| {
                candidate_report(
                    name,
                    "local_ensemble",
                    forecast,
                    "consensus",
                    &report_reference,
                    1.0 / candidates.len().max(1) as f32,
                    Some(snapshot),
                )
            })
            .collect::<Vec<_>>();
        apply_candidate_weights(&mut reports);
        reports
    };
    if reports.is_empty() {
        artifact.runtime_mode = SwarmRuntimeMode::LocalFallback;
        artifact.runtime_degraded_reason = Some("swarm_local_fallback_unrecoverable".to_string());
        artifact.last_result = None;
        artifact.last_horizon = None;
        artifact.candidate_reports.clear();
        artifact.updated_at_unix_ms = None;
        artifact.training_report = None;
        return Ok(());
    }

    normalize_candidate_weights(&mut reports);
    let weight_map = build_candidate_weight_map(&reports);
    let result = fallback_forecast_with_strategy(
        snapshot,
        &candidates,
        &weight_map,
        &reports,
        artifact.config.strategy,
        horizon,
    );

    artifact.runtime_mode = SwarmRuntimeMode::LocalFallback;
    artifact.runtime_degraded_reason = Some("swarm_local_fallback_active".to_string());
    artifact.candidate_reports = reports;
    artifact.last_result = Some(result.clone());
    artifact.last_horizon = Some(horizon);
    artifact.updated_at_unix_ms = current_unix_ms().or(artifact.updated_at_unix_ms);
    artifact.training_report = Some(build_training_report(
        &artifact.candidate_reports,
        validation_windows.len(),
        artifact.values.len(),
        horizon,
        result.diversity_score,
        snapshot.trend_strength - snapshot.mean_reversion_strength,
    ));
    Ok(())
}

#[cfg(feature = "swarm-forecasting")]
fn map_strategy(strategy: SwarmEnsembleStrategy) -> EnsembleStrategy {
    match strategy {
        SwarmEnsembleStrategy::SimpleAverage => EnsembleStrategy::SimpleAverage,
        SwarmEnsembleStrategy::WeightedAverage => EnsembleStrategy::WeightedAverage,
        SwarmEnsembleStrategy::Median => EnsembleStrategy::Median,
        SwarmEnsembleStrategy::TrimmedMean => EnsembleStrategy::TrimmedMean(0.1),
        SwarmEnsembleStrategy::BayesianModelAveraging => EnsembleStrategy::BayesianModelAveraging,
    }
}

fn mae(prediction: &[f32], actuals: &[f32]) -> f32 {
    if prediction.is_empty() || actuals.is_empty() {
        return 1.0;
    }

    prediction
        .iter()
        .zip(actuals.iter())
        .map(|(predicted, actual)| (*predicted - *actual).abs())
        .sum::<f32>()
        / prediction.len().min(actuals.len()) as f32
}

fn mse(prediction: &[f32], actuals: &[f32]) -> f32 {
    if prediction.is_empty() || actuals.is_empty() {
        return 1.0;
    }

    prediction
        .iter()
        .zip(actuals.iter())
        .map(|(predicted, actual)| (*predicted - *actual).powi(2))
        .sum::<f32>()
        / prediction.len().min(actuals.len()) as f32
}

fn mape(prediction: &[f32], actuals: &[f32]) -> f32 {
    if prediction.is_empty() || actuals.is_empty() {
        return 100.0;
    }

    let mut total = 0.0_f32;
    let mut count = 0usize;

    for (predicted, actual) in prediction.iter().zip(actuals.iter()) {
        if actual.abs() > 1e-6 {
            total += ((*predicted - *actual) / *actual).abs();
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f32 * 100.0
    }
}

fn smape(prediction: &[f32], actuals: &[f32]) -> f32 {
    if prediction.is_empty() || actuals.is_empty() {
        return 100.0;
    }

    let mut total = 0.0_f32;
    let mut count = 0usize;

    for (predicted, actual) in prediction.iter().zip(actuals.iter()) {
        let denominator = (predicted.abs() + actual.abs()) * 0.5;
        if denominator > 1e-6 {
            total += (*predicted - *actual).abs() / denominator;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f32 * 100.0
    }
}

fn volatility(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (*value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    variance.sqrt()
}

fn trend_slope(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f32;
    let mean_x = (n - 1.0) * 0.5;
    let mean_y = values.iter().copied().sum::<f32>() / n;
    let mut numerator = 0.0_f32;
    let mut denominator = 0.0_f32;

    for (idx, value) in values.iter().copied().enumerate() {
        let x = idx as f32;
        numerator += (x - mean_x) * (value - mean_y);
        denominator += (x - mean_x).powi(2);
    }

    if denominator <= 1e-6 {
        0.0
    } else {
        numerator / denominator
    }
}

fn persistence_forecast(last_value: f32, horizon: usize) -> Vec<f32> {
    vec![last_value; horizon]
}

fn moving_average_forecast(values: &[f32], horizon: usize, window: usize) -> Vec<f32> {
    let effective_window = window.min(values.len()).max(1);
    let mean = values[values.len() - effective_window..]
        .iter()
        .copied()
        .sum::<f32>()
        / effective_window as f32;
    vec![mean; horizon]
}

fn drift_forecast(last_value: f32, slope: f32, horizon: usize) -> Vec<f32> {
    (0..horizon)
        .map(|step| last_value + slope * (step as f32 + 1.0))
        .collect()
}

fn seasonal_forecast(
    values: &[f32],
    seasonal_periods: &[usize],
    horizon: usize,
) -> Option<Vec<f32>> {
    let period = seasonal_periods
        .iter()
        .copied()
        .find(|period| *period > 0 && values.len() >= *period)?;

    Some(
        (0..horizon)
            .map(|step| {
                let source_idx = values.len() - period + (step % period);
                values[source_idx]
            })
            .collect(),
    )
}

#[cfg(feature = "swarm-forecasting")]
fn build_training_series(
    values: &[f32],
    timestamps: &[f64],
    frequency: &str,
    unique_id: &str,
) -> Result<TimeSeriesData> {
    if values.len() != timestamps.len() {
        bail!(
            "swarm forecaster value/timestamp mismatch: {} values vs {} timestamps",
            values.len(),
            timestamps.len()
        );
    }
    if values.len() < 32 {
        bail!(
            "swarm forecaster requires at least 32 observations, received {}",
            values.len()
        );
    }

    Ok(TimeSeriesData {
        values: values.to_vec(),
        timestamps: timestamps.to_vec(),
        frequency: frequency.to_string(),
        unique_id: unique_id.to_string(),
    })
}

#[cfg(feature = "swarm-forecasting")]
fn infer_snapshot(
    series: &TimeSeriesData,
    processor: &TimeSeriesProcessor,
) -> SwarmForecastSnapshot {
    let seasonality: SeasonalityInfo = processor.detect_seasonality(series);
    let last_window = series.values.len().min(32);
    let short_window = series.values.len().min(8);
    let medium_window = series.values.len().min(16);
    let long_window = series.values.len().min(48);
    let rolling_mean = series.values[series.values.len() - last_window..]
        .iter()
        .copied()
        .sum::<f32>()
        / last_window as f32;
    let short_mean = series.values[series.values.len() - short_window..]
        .iter()
        .copied()
        .sum::<f32>()
        / short_window as f32;
    let medium_mean = series.values[series.values.len() - medium_window..]
        .iter()
        .copied()
        .sum::<f32>()
        / medium_window as f32;
    let long_mean = series.values[series.values.len() - long_window..]
        .iter()
        .copied()
        .sum::<f32>()
        / long_window as f32;
    let slope = trend_slope(&series.values[series.values.len() - last_window..]);
    let window_volatility = volatility(&series.values[series.values.len() - last_window..]);
    let baseline_volatility = volatility(&series.values).max(1e-6);

    SwarmForecastSnapshot {
        last_value: *series
            .values
            .last()
            .expect("training series snapshot requires at least one value"),
        rolling_mean,
        drift_slope: slope,
        volatility: window_volatility,
        has_trend: seasonality.has_trend || slope.abs() > 1e-3,
        has_seasonality: seasonality.has_seasonality,
        seasonal_periods: seasonality.seasonal_periods,
        short_mean,
        medium_mean,
        long_mean,
        recent_return: series
            .values
            .last()
            .zip(series.values.get(series.values.len().saturating_sub(2)))
            .map(|(last, prev)| last - prev)
            .unwrap_or_default(),
        trend_strength: clamp_unit(
            ((short_mean - long_mean).abs() + slope.abs()) / (window_volatility + 1e-6),
        ),
        mean_reversion_strength: clamp_unit(
            (1.0 - ((short_mean - rolling_mean).abs() / (window_volatility + 1e-6))).max(0.0),
        ),
        volatility_ratio: (window_volatility / baseline_volatility).max(0.0),
    }
}

#[cfg(feature = "swarm-forecasting")]
fn candidate_predictions(
    values: &[f32],
    snapshot: &SwarmForecastSnapshot,
    horizon: usize,
) -> Vec<(String, ModelType, Vec<f32>)> {
    let trend_weight = snapshot.trend_strength.clamp(0.1, 0.9);
    let mean_reversion = snapshot.mean_reversion_strength.clamp(0.05, 0.9);
    let trend_dominant = swarm_trend_dominant(snapshot);
    let mean_reversion_dominant = swarm_mean_reversion_dominant(snapshot);
    let high_volatility = swarm_high_volatility(snapshot);
    let mut predictions = vec![(
        "persistence".to_string(),
        ModelType::MLP,
        persistence_forecast(snapshot.last_value, horizon),
    )];

    predictions.push((
        "moving_average_medium".to_string(),
        ModelType::DLinear,
        moving_average_forecast(values, horizon, 8),
    ));

    if mean_reversion_dominant || high_volatility {
        predictions.push((
            "moving_average_short".to_string(),
            ModelType::DLinear,
            moving_average_forecast(values, horizon, 4),
        ));
    }

    if trend_dominant || !high_volatility {
        predictions.push((
            "moving_average_long".to_string(),
            ModelType::DLinear,
            moving_average_forecast(values, horizon, 16),
        ));
    }

    if trend_dominant || snapshot.recent_return.abs() > snapshot.volatility.max(1e-6) * 0.35 {
        predictions.push((
            "ewma_fast".to_string(),
            ModelType::MLP,
            ewma_forecast(values, horizon, 0.45),
        ));
    }

    if !trend_dominant || snapshot.volatility_ratio < 1.4 {
        predictions.push((
            "ewma_slow".to_string(),
            ModelType::MLP,
            ewma_forecast(values, horizon, 0.22),
        ));
    }

    if trend_dominant || snapshot.has_trend {
        predictions.push((
            "damped_drift".to_string(),
            ModelType::TiDE,
            damped_drift_forecast(snapshot.last_value, snapshot.drift_slope, horizon, 0.82),
        ));
        predictions.push((
            "drift".to_string(),
            ModelType::TiDE,
            drift_forecast(snapshot.last_value, snapshot.drift_slope, horizon),
        ));
    }

    if mean_reversion_dominant || !snapshot.has_trend || high_volatility {
        predictions.push((
            "mean_reversion".to_string(),
            ModelType::TiDE,
            mean_reversion_forecast(
                snapshot.last_value,
                snapshot.rolling_mean,
                mean_reversion,
                horizon,
            ),
        ));
        predictions.push((
            "regime_anchor".to_string(),
            ModelType::NHITS,
            blend_forecasts(
                &moving_average_forecast(values, horizon, 12),
                &mean_reversion_forecast(
                    snapshot.last_value,
                    snapshot.medium_mean.max(snapshot.rolling_mean),
                    mean_reversion,
                    horizon,
                ),
                0.5 + 0.3 * snapshot.trend_strength,
            ),
        ));
    }

    if trend_dominant && snapshot.recent_return.abs() > 1e-6 {
        predictions.push((
            "momentum_blend".to_string(),
            ModelType::TiDE,
            blend_forecasts(
                &damped_drift_forecast(snapshot.last_value, snapshot.drift_slope, horizon, 0.90),
                &ewma_forecast(values, horizon, 0.35),
                trend_weight,
            ),
        ));
    }

    if snapshot.has_seasonality
        && let Some(seasonal) = seasonal_forecast(values, &snapshot.seasonal_periods, horizon)
    {
        predictions.push(("seasonal".to_string(), ModelType::NHITS, seasonal));
        if trend_dominant
            && let Some((_, _, drift)) = predictions.iter().find(|(name, _, _)| name == "drift")
        {
            let seasonal_drift = blend_forecasts(
                drift,
                predictions
                    .iter()
                    .find(|(name, _, _)| name == "seasonal")
                    .map(|(_, _, forecast)| forecast.as_slice())
                    .unwrap_or(&[]),
                0.55 + 0.25 * snapshot.trend_strength,
            );
            predictions.push((
                "seasonal_drift".to_string(),
                ModelType::NHITS,
                seasonal_drift,
            ));
        }
    }

    prune_external_candidates(snapshot, predictions)
}

fn current_unix_ms() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis() as u64)
}

fn prediction_stats(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (*value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    (mean, variance.sqrt())
}

fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn swarm_trend_dominant(snapshot: &SwarmForecastSnapshot) -> bool {
    snapshot.trend_strength >= 0.55
        && snapshot.trend_strength >= snapshot.mean_reversion_strength * 1.10
}

fn swarm_mean_reversion_dominant(snapshot: &SwarmForecastSnapshot) -> bool {
    snapshot.mean_reversion_strength >= 0.45
        && snapshot.mean_reversion_strength >= snapshot.trend_strength * 0.95
}

fn swarm_high_volatility(snapshot: &SwarmForecastSnapshot) -> bool {
    snapshot.volatility_ratio >= 1.35
}

fn is_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn signed_direction(value: f32) -> f32 {
    if value > 1e-6 {
        1.0
    } else if value < -1e-6 {
        -1.0
    } else {
        0.0
    }
}

fn ewma_forecast(values: &[f32], horizon: usize, alpha: f32) -> Vec<f32> {
    let alpha = alpha.clamp(0.05, 0.95);
    let mut smoothed = *values
        .last()
        .expect("EWMA forecast requires at least one observation");
    for value in values.iter().copied() {
        smoothed = alpha * value + (1.0 - alpha) * smoothed;
    }
    vec![smoothed; horizon]
}

fn damped_drift_forecast(last_value: f32, slope: f32, horizon: usize, damping: f32) -> Vec<f32> {
    let damping = damping.clamp(0.2, 0.995);
    (0..horizon)
        .scan((last_value, slope), |state, _| {
            state.0 += state.1;
            state.1 *= damping;
            Some(state.0)
        })
        .collect()
}

fn mean_reversion_forecast(
    last_value: f32,
    anchor: f32,
    strength: f32,
    horizon: usize,
) -> Vec<f32> {
    let strength = strength.clamp(0.0, 1.0);
    let mut state = last_value;
    (0..horizon)
        .map(|_| {
            state += (anchor - state) * strength.max(0.05);
            state
        })
        .collect()
}

fn blend_forecasts(primary: &[f32], secondary: &[f32], primary_weight: f32) -> Vec<f32> {
    let primary_weight = primary_weight.clamp(0.0, 1.0);
    let secondary_weight = 1.0 - primary_weight;
    primary
        .iter()
        .zip(secondary.iter())
        .map(|(left, right)| *left * primary_weight + *right * secondary_weight)
        .collect()
}

fn interval_coverage(prediction: &[f32], reference: &[f32], band: f32) -> f32 {
    if prediction.is_empty() || reference.is_empty() {
        return 0.0;
    }

    let band = band.max(1e-6);
    let mut inside = 0usize;
    let mut count = 0usize;
    for (predicted, actual) in prediction.iter().zip(reference.iter()) {
        if predicted.is_finite() && actual.is_finite() {
            count += 1;
            if (*predicted - *actual).abs() <= band {
                inside += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        inside as f32 / count as f32
    }
}

fn aggregate_average(predictions: &[Vec<f32>], horizon: usize, baseline: f32) -> Vec<f32> {
    if predictions.is_empty() {
        return vec![baseline; horizon];
    }

    (0..horizon)
        .map(|step| {
            let mut sum = 0.0_f32;
            let mut count = 0usize;
            for forecast in predictions {
                if let Some(value) = forecast.get(step)
                    && value.is_finite()
                {
                    sum += *value;
                    count += 1;
                }
            }
            if count == 0 {
                baseline
            } else {
                sum / count as f32
            }
        })
        .collect()
}

fn step_statistic_with_trim<F>(
    candidates: &[(String, Vec<f32>)],
    horizon: usize,
    baseline: f32,
    trim_fraction: f32,
    statistic: F,
) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    (0..horizon)
        .map(|step| {
            let mut values = candidates
                .iter()
                .filter_map(|(_, forecast)| forecast.get(step).copied())
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            if values.is_empty() {
                return baseline;
            }

            values.sort_by(|left, right| left.total_cmp(right));
            let max_trim = values.len().saturating_sub(1) / 2;
            let trim = ((values.len() as f32) * trim_fraction.clamp(0.0, 0.45)).floor() as usize;
            let trim = trim.min(max_trim);
            let effective = &values[trim..values.len() - trim];
            if effective.is_empty() {
                baseline
            } else {
                statistic(effective)
            }
        })
        .collect()
}

fn aggregate_median(candidates: &[(String, Vec<f32>)], horizon: usize, baseline: f32) -> Vec<f32> {
    step_statistic_with_trim(candidates, horizon, baseline, 0.0, |values| {
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) * 0.5
        } else {
            values[mid]
        }
    })
}

fn aggregate_trimmed_mean(
    candidates: &[(String, Vec<f32>)],
    horizon: usize,
    baseline: f32,
    trim_fraction: f32,
) -> Vec<f32> {
    step_statistic_with_trim(candidates, horizon, baseline, trim_fraction, |values| {
        values.iter().copied().sum::<f32>() / values.len().max(1) as f32
    })
}

fn bayesian_model_average_weights(
    reports: &[SwarmCandidateReport],
    fallback_weights: &HashMap<String, f32>,
) -> HashMap<String, f32> {
    if reports.is_empty() {
        return fallback_weights.clone();
    }

    let mut scored = Vec::with_capacity(reports.len());
    let mut max_log_score = f32::NEG_INFINITY;
    for report in reports {
        let prior = fallback_weights
            .get(&report.name)
            .copied()
            .unwrap_or(report.weight.max(1e-6))
            .max(1e-6);
        let loss = report.mae.max(0.0)
            + report.mse.max(0.0).sqrt()
            + 0.01 * report.smape.max(0.0)
            + 0.5 * report.bias.abs()
            + 0.4 * (1.0 - clamp_unit(report.coverage))
            + 0.5 * (1.0 - clamp_unit(report.directional_accuracy))
            + 0.35 * (1.0 - clamp_unit(report.regime_fit))
            + 0.25 * (1.0 - clamp_unit(report.stability_score));
        let log_score = prior.ln() - loss;
        max_log_score = max_log_score.max(log_score);
        scored.push((report.name.clone(), log_score));
    }

    let mut posterior = HashMap::with_capacity(scored.len());
    let mut total = 0.0_f32;
    for (name, log_score) in &scored {
        let weight = (*log_score - max_log_score).exp().max(1e-6);
        total += weight;
        posterior.insert(name.clone(), weight);
    }

    if total <= f32::EPSILON {
        return fallback_weights.clone();
    }

    for weight in posterior.values_mut() {
        *weight /= total;
    }
    posterior
}

fn effective_strategy_weights(
    strategy: SwarmEnsembleStrategy,
    candidates: &[(String, Vec<f32>)],
    weights: &HashMap<String, f32>,
    reports: &[SwarmCandidateReport],
) -> HashMap<String, f32> {
    match strategy {
        SwarmEnsembleStrategy::WeightedAverage => weights.clone(),
        SwarmEnsembleStrategy::BayesianModelAveraging => {
            bayesian_model_average_weights(reports, weights)
        }
        SwarmEnsembleStrategy::SimpleAverage
        | SwarmEnsembleStrategy::Median
        | SwarmEnsembleStrategy::TrimmedMean => {
            let uniform = 1.0 / candidates.len().max(1) as f32;
            candidates
                .iter()
                .map(|(name, _)| (name.clone(), uniform))
                .collect()
        }
    }
}

fn active_candidate_names(reports: &[SwarmCandidateReport]) -> HashSet<String> {
    reports
        .iter()
        .filter(|report| report.weight.is_finite() && report.weight > f32::EPSILON)
        .map(|report| report.name.clone())
        .collect()
}

fn select_active_candidates(
    candidates: &[(String, Vec<f32>)],
    reports: &[SwarmCandidateReport],
) -> Vec<(String, Vec<f32>)> {
    if reports.is_empty() {
        return candidates.to_vec();
    }

    let active = active_candidate_names(reports);
    let filtered = candidates
        .iter()
        .filter(|(name, _)| active.contains(name))
        .cloned()
        .collect::<Vec<_>>();
    if filtered.len() >= 2 {
        filtered
    } else {
        candidates.to_vec()
    }
}

fn active_report_refs<'a>(
    reports: &'a [SwarmCandidateReport],
    candidates: &[(String, Vec<f32>)],
) -> Vec<&'a SwarmCandidateReport> {
    if reports.is_empty() {
        return Vec::new();
    }
    let active_names = candidates
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<HashSet<_>>();
    reports
        .iter()
        .filter(|report| active_names.contains(report.name.as_str()))
        .collect()
}

fn weighted_report_mean(
    reports: &[&SwarmCandidateReport],
    selector: fn(&SwarmCandidateReport) -> f32,
) -> Option<f32> {
    let total_weight = reports
        .iter()
        .map(|report| report.weight.max(0.0))
        .sum::<f32>();
    if total_weight <= f32::EPSILON {
        return None;
    }
    Some(
        reports
            .iter()
            .map(|report| selector(report) * report.weight.max(0.0))
            .sum::<f32>()
            / total_weight,
    )
}

fn percentile(sorted_values: &[f32], quantile: f32) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let position = ((sorted_values.len() - 1) as f32 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted_values[position.min(sorted_values.len() - 1)]
}

fn calibrated_interval_spread(
    center: f32,
    step_values: &[f32],
    reports: &[&SwarmCandidateReport],
    snapshot: &SwarmForecastSnapshot,
) -> f32 {
    let mut sorted = step_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return snapshot.volatility.max(1e-6);
    }
    sorted.sort_by(|left, right| left.total_cmp(right));
    let (_, std_dev) = prediction_stats(&sorted);
    let lower_q = percentile(&sorted, 0.10);
    let upper_q = percentile(&sorted, 0.90);
    let quantile_half_width = (center - lower_q)
        .abs()
        .max((upper_q - center).abs())
        .max(std_dev * 1.2816);

    let weighted_mae = weighted_report_mean(reports, |report| report.mae.max(0.0)).unwrap_or(0.0);
    let weighted_std =
        weighted_report_mean(reports, |report| report.prediction_std.max(0.0)).unwrap_or(0.0);
    let weighted_bias = weighted_report_mean(reports, |report| report.bias.abs()).unwrap_or(0.0);
    let weighted_coverage =
        weighted_report_mean(reports, |report| clamp_unit(report.coverage)).unwrap_or(0.8);
    let coverage_scale = if weighted_coverage < 0.8 {
        1.0 + (0.8 - weighted_coverage) * 1.5
    } else {
        1.0 - ((weighted_coverage - 0.8) * 0.25).min(0.10)
    };
    let validation_floor = weighted_mae
        .max(weighted_std * 0.85)
        .max(weighted_bias * 0.75)
        .max(snapshot.volatility * 0.35)
        .max(1e-6);

    quantile_half_width.max(validation_floor * coverage_scale)
}

#[allow(dead_code)]
fn fallback_forecast_from_forecasts(
    snapshot: &SwarmForecastSnapshot,
    candidates: &[(String, Vec<f32>)],
    horizon: usize,
) -> SwarmForecastResult {
    let weights = build_candidate_weight_map_from_consensus(snapshot, candidates);
    fallback_forecast_with_strategy(
        snapshot,
        candidates,
        &weights,
        &[],
        SwarmEnsembleStrategy::WeightedAverage,
        horizon,
    )
}

fn fallback_forecast_with_strategy(
    snapshot: &SwarmForecastSnapshot,
    candidates: &[(String, Vec<f32>)],
    weights: &HashMap<String, f32>,
    reports: &[SwarmCandidateReport],
    strategy: SwarmEnsembleStrategy,
    horizon: usize,
) -> SwarmForecastResult {
    let active_candidates = select_active_candidates(candidates, reports);
    let active_reports = active_report_refs(reports, &active_candidates);
    let forecasts = active_candidates
        .iter()
        .map(|(_, forecast)| forecast.clone())
        .collect::<Vec<_>>();
    let effective_weights =
        effective_strategy_weights(strategy, &active_candidates, weights, reports);
    let point_forecast = match strategy {
        SwarmEnsembleStrategy::SimpleAverage => {
            aggregate_average(&forecasts, horizon, snapshot.last_value)
        }
        SwarmEnsembleStrategy::WeightedAverage | SwarmEnsembleStrategy::BayesianModelAveraging => {
            aggregate_weighted(
                &active_candidates,
                &effective_weights,
                horizon,
                snapshot.last_value,
            )
        }
        SwarmEnsembleStrategy::Median => {
            aggregate_median(&active_candidates, horizon, snapshot.last_value)
        }
        SwarmEnsembleStrategy::TrimmedMean => {
            aggregate_trimmed_mean(&active_candidates, horizon, snapshot.last_value, 0.15)
        }
    };
    let mut lower = Vec::with_capacity(horizon);
    let mut upper = Vec::with_capacity(horizon);
    let mut variance_sum = 0.0_f32;

    for step in 0..horizon {
        let step_values = forecasts
            .iter()
            .filter_map(|forecast| forecast.get(step).copied())
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let (mean, _) = prediction_stats(&step_values);
        let center = point_forecast.get(step).copied().unwrap_or(mean);
        let spread = calibrated_interval_spread(center, &step_values, &active_reports, snapshot);
        lower.push(center - 1.2816 * spread);
        upper.push(center + 1.2816 * spread);
        variance_sum += spread * spread;
    }

    let diversity_score = if forecasts.len() <= 1 {
        0.0
    } else {
        (variance_sum / horizon.max(1) as f32).sqrt()
    };
    let effective_models = candidate_effective_models(&effective_weights, active_candidates.len());

    SwarmForecastResult {
        point_forecast,
        level_80_lower: lower,
        level_80_upper: upper,
        diversity_score,
        effective_models,
        prediction_variance: if horizon == 0 {
            0.0
        } else {
            variance_sum / horizon as f32
        },
        models_used: forecasts.len(),
    }
}

fn result_is_valid(result: &SwarmForecastResult, horizon: usize) -> bool {
    let valid_len = result.point_forecast.len() == horizon
        && result.level_80_lower.len() == horizon
        && result.level_80_upper.len() == horizon;
    let valid_values = result
        .point_forecast
        .iter()
        .zip(result.level_80_lower.iter())
        .zip(result.level_80_upper.iter())
        .all(|((point, lower), upper)| {
            let point = *point;
            let lower = *lower;
            let upper = *upper;
            point.is_finite()
                && lower.is_finite()
                && upper.is_finite()
                && lower <= upper
                && point >= lower.min(upper)
                && point <= lower.max(upper)
        });
    valid_len && valid_values
}

#[cfg(feature = "swarm-forecasting")]
fn select_external_or_fallback_result(
    snapshot: &SwarmForecastSnapshot,
    candidates: &[(String, Vec<f32>)],
    weights: &HashMap<String, f32>,
    reports: &[SwarmCandidateReport],
    strategy: SwarmEnsembleStrategy,
    horizon: usize,
    external_result: SwarmForecastResult,
) -> (SwarmForecastResult, SwarmRuntimeMode, Option<String>) {
    if result_is_valid(&external_result, horizon) {
        (external_result, SwarmRuntimeMode::ExternalSwarm, None)
    } else {
        (
            fallback_forecast_with_strategy(
                snapshot, candidates, weights, reports, strategy, horizon,
            ),
            SwarmRuntimeMode::LocalFallback,
            Some("external_swarm_result_invalid".to_string()),
        )
    }
}

fn candidate_report(
    name: &str,
    model_type: &str,
    forecast: &[f32],
    source: &str,
    reference: &[f32],
    weight: f32,
    snapshot: Option<&SwarmForecastSnapshot>,
) -> SwarmCandidateReport {
    let (mean, std_dev) = prediction_stats(forecast);
    let band = std_dev.max(1e-6);
    let mae = mae(forecast, reference);
    let mse = mse(forecast, reference);
    let mape = mape(forecast, reference);
    let smape = smape(forecast, reference);
    let bias = if forecast.is_empty() || reference.is_empty() {
        0.0
    } else {
        forecast
            .iter()
            .zip(reference.iter())
            .map(|(predicted, actual)| *predicted - *actual)
            .sum::<f32>()
            / forecast.len().min(reference.len()) as f32
    };
    let coverage = {
        let mut inside = 0usize;
        let mut count = 0usize;
        for (predicted, actual) in forecast.iter().zip(reference.iter()) {
            if predicted.is_finite() && actual.is_finite() {
                count += 1;
                if (*predicted - *actual).abs() <= band {
                    inside += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            inside as f32 / count as f32
        }
    };
    let directional_accuracy = directional_accuracy(forecast, reference);
    let regime_fit = snapshot
        .map(|snapshot| candidate_regime_fit(snapshot, forecast))
        .unwrap_or(0.5);
    let volatility_scale = snapshot.map(|s| s.volatility.max(1e-6)).unwrap_or(1.0) * 2.0;
    let stability_score = clamp_unit(1.0 - (std_dev / volatility_scale.max(1e-6)).min(1.0));

    SwarmCandidateReport {
        name: name.to_string(),
        model_type: model_type.to_string(),
        source: source.to_string(),
        weight,
        prediction_length: forecast.len(),
        prediction_mean: mean,
        prediction_std: std_dev,
        mae,
        mse,
        mape,
        smape,
        coverage,
        bias,
        directional_accuracy,
        regime_fit,
        stability_score,
    }
}

fn directional_accuracy(forecast: &[f32], reference: &[f32]) -> f32 {
    if forecast.len() < 2 || reference.len() < 2 {
        return 0.5;
    }

    let comparisons = forecast
        .windows(2)
        .zip(reference.windows(2))
        .map(|(predicted, actual)| {
            let predicted_direction = signed_direction(predicted[1] - predicted[0]);
            let actual_direction = signed_direction(actual[1] - actual[0]);
            if predicted_direction == 0.0 && actual_direction == 0.0 {
                1.0
            } else if predicted_direction == 0.0 || actual_direction == 0.0 {
                0.5
            } else if predicted_direction == actual_direction {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();

    if comparisons.is_empty() {
        0.5
    } else {
        comparisons.iter().copied().sum::<f32>() / comparisons.len() as f32
    }
}

fn candidate_regime_fit(snapshot: &SwarmForecastSnapshot, forecast: &[f32]) -> f32 {
    if forecast.len() < 2 {
        return 0.5;
    }

    let forecast_slope = trend_slope(forecast);
    let direction_alignment = 1.0
        - ((forecast_slope - snapshot.drift_slope).abs() / (snapshot.volatility + 1e-6)).min(1.0);
    let anchor = if snapshot.trend_strength >= snapshot.mean_reversion_strength {
        snapshot.short_mean.max(snapshot.medium_mean)
    } else {
        snapshot.rolling_mean
    };
    let terminal = *forecast.last().unwrap_or(&snapshot.last_value);
    let anchor_fit = 1.0 - ((terminal - anchor).abs() / (snapshot.volatility + 1e-6)).min(1.0);
    clamp_unit(direction_alignment * 0.6 + anchor_fit * 0.4)
}

fn candidate_family_alignment(name: &str, snapshot: &SwarmForecastSnapshot) -> f32 {
    let name = name.to_ascii_lowercase();
    if name.contains("seasonal") {
        if snapshot.has_seasonality { 0.95 } else { 0.15 }
    } else if name.contains("mean_reversion") || name.contains("regime_anchor") {
        0.25 + 0.75 * clamp_unit(snapshot.mean_reversion_strength)
    } else if name.contains("drift") || name.contains("momentum") || name.contains("ewma_fast") {
        0.25 + 0.75 * clamp_unit(snapshot.trend_strength)
    } else if name.contains("moving_average_long") {
        0.35 + 0.45
            * clamp_unit(
                snapshot
                    .trend_strength
                    .max(snapshot.mean_reversion_strength),
            )
    } else if name.contains("moving_average_short")
        || name.contains("moving_average_medium")
        || name.contains("ewma_slow")
    {
        0.40 + 0.40 * clamp_unit((snapshot.trend_strength + snapshot.mean_reversion_strength) * 0.5)
    } else if name.contains("persistence") {
        0.45
    } else {
        0.50
    }
}

fn candidate_preselection_score(
    name: &str,
    forecast: &[f32],
    snapshot: &SwarmForecastSnapshot,
) -> f32 {
    if forecast.is_empty() || forecast.iter().any(|value| !value.is_finite()) {
        return 0.0;
    }
    let (_, std_dev) = prediction_stats(forecast);
    let stability = clamp_unit(1.0 - std_dev / (snapshot.volatility.max(1e-6) * 2.0).max(1e-6));
    let regime_fit = candidate_regime_fit(snapshot, forecast);
    let family_alignment = candidate_family_alignment(name, snapshot);
    0.55 * regime_fit + 0.25 * stability + 0.20 * family_alignment
}

fn forecast_distance_ratio(left: &[f32], right: &[f32], scale: f32) -> f32 {
    let compared = left
        .iter()
        .zip(right.iter())
        .filter_map(|(left, right)| {
            if left.is_finite() && right.is_finite() {
                Some((*left - *right).abs())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if compared.is_empty() {
        return f32::INFINITY;
    }
    compared.iter().copied().sum::<f32>() / compared.len() as f32 / scale.max(1e-6)
}

fn prune_external_candidates(
    snapshot: &SwarmForecastSnapshot,
    candidates: Vec<(String, ModelType, Vec<f32>)>,
) -> Vec<(String, ModelType, Vec<f32>)> {
    if candidates.len() <= 3 {
        return candidates;
    }

    let mut scored = candidates
        .into_iter()
        .map(|candidate| {
            let score = candidate_preselection_score(&candidate.0, &candidate.2, snapshot);
            (score, candidate)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.0.total_cmp(&left.0));

    let duplicate_threshold = if snapshot.volatility_ratio > 1.25 {
        0.05
    } else {
        0.035
    };
    let scale = snapshot
        .volatility
        .max(snapshot.last_value.abs() * 0.005)
        .max(1e-4);
    let max_keep = if snapshot.has_seasonality { 6 } else { 5 };

    let mut kept = Vec::new();
    for (_, candidate) in scored {
        let is_duplicate = kept.iter().any(|existing: &(String, ModelType, Vec<f32>)| {
            forecast_distance_ratio(&candidate.2, &existing.2, scale) <= duplicate_threshold
        });
        if !is_duplicate || kept.len() < 2 {
            kept.push(candidate);
        }
        if kept.len() >= max_keep {
            break;
        }
    }

    kept
}

fn report_weight(report: &SwarmCandidateReport) -> f32 {
    let mae_term = 1.0 / (1.0 + report.mae.max(0.0));
    let mse_term = 1.0 / (1.0 + report.mse.max(0.0).sqrt());
    let percentage_term = 1.0 / (1.0 + 0.01 * (report.mape.max(0.0) + report.smape.max(0.0)));
    let bias_penalty = 1.0 / (1.0 + report.bias.abs());
    let quality = mae_term * mse_term * percentage_term * bias_penalty;
    let calibration = 0.35 + 0.65 * clamp_unit(report.coverage);
    let direction = 0.35 + 0.65 * clamp_unit(report.directional_accuracy);
    let regime = 0.35 + 0.65 * clamp_unit(report.regime_fit);
    let stability = 0.35 + 0.65 * clamp_unit(report.stability_score);
    (quality * calibration * direction * regime * stability).max(1e-6)
}

fn apply_candidate_weights(reports: &mut [SwarmCandidateReport]) {
    let total = reports
        .iter()
        .map(report_weight)
        .fold(0.0_f32, |acc, weight| acc + weight);
    if total <= f32::EPSILON {
        let uniform = if reports.is_empty() {
            1.0
        } else {
            1.0 / reports.len() as f32
        };
        for report in reports {
            report.weight = uniform;
        }
        return;
    }

    for report in reports {
        report.weight = report_weight(report) / total;
    }
}

fn normalize_candidate_weights(reports: &mut [SwarmCandidateReport]) {
    let total = reports
        .iter()
        .map(|report| report.weight.max(0.0))
        .sum::<f32>();
    if total <= f32::EPSILON {
        let uniform = if reports.is_empty() {
            1.0
        } else {
            1.0 / reports.len() as f32
        };
        for report in reports {
            report.weight = uniform;
        }
        return;
    }

    for report in reports {
        report.weight = report.weight.max(0.0) / total;
    }
}

fn derive_validation_candidate_weights(
    losses: &HashMap<String, f32>,
    support: &HashMap<String, f32>,
) -> HashMap<String, f32> {
    let mut candidates = losses
        .iter()
        .filter_map(|(name, loss_sum)| {
            let support_weight = support.get(name).copied().unwrap_or(0.0);
            if !loss_sum.is_finite()
                || support_weight <= f32::EPSILON
                || !support_weight.is_finite()
            {
                return None;
            }
            Some((name.clone(), (loss_sum / support_weight).max(0.0)))
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return HashMap::new();
    }

    let min_loss = candidates
        .iter()
        .map(|(_, loss)| *loss)
        .fold(f32::INFINITY, f32::min);
    let mean_loss = candidates.iter().map(|(_, loss)| *loss).sum::<f32>() / candidates.len() as f32;
    let temperature = mean_loss.max(1e-3);

    let mut total = 0.0_f32;
    for (_, loss) in &mut candidates {
        let centered = (*loss - min_loss).max(0.0);
        *loss = (-(centered / temperature)).exp().max(1e-6);
        total += *loss;
    }

    if total <= f32::EPSILON {
        return HashMap::new();
    }

    candidates
        .into_iter()
        .map(|(name, score)| (name, score / total))
        .collect()
}

fn validation_weight_blend_ratio(support: &HashMap<String, f32>, candidate_count: usize) -> f32 {
    if candidate_count == 0 {
        return 0.0;
    }

    let total_support = support
        .values()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f32>();
    if total_support <= f32::EPSILON {
        return 0.0;
    }

    let average_support = total_support / candidate_count as f32;
    (average_support / (average_support + 1.0)).clamp(0.0, 1.0)
}

fn apply_validation_candidate_weights(
    reports: &mut [SwarmCandidateReport],
    losses: &HashMap<String, f32>,
    support: &HashMap<String, f32>,
) {
    let learned = derive_validation_candidate_weights(losses, support);
    if learned.is_empty() {
        apply_candidate_weights(reports);
        return;
    }
    let learned_ratio = validation_weight_blend_ratio(support, reports.len());
    let heuristic_ratio = 1.0 - learned_ratio;

    let heuristic_total = reports
        .iter()
        .map(report_weight)
        .fold(0.0_f32, |acc, weight| acc + weight)
        .max(f32::EPSILON);

    let mut total = 0.0_f32;
    for report in reports.iter_mut() {
        let learned_weight = learned.get(&report.name).copied().unwrap_or(0.0);
        let heuristic_weight = report_weight(report) / heuristic_total;
        let blended = learned_ratio * learned_weight + heuristic_ratio * heuristic_weight;
        report.weight = blended.max(1e-6);
        total += report.weight;
    }

    if total <= f32::EPSILON {
        apply_candidate_weights(reports);
        return;
    }

    normalize_candidate_weights(reports);
}

fn prune_validation_candidates(
    reports: &mut Vec<SwarmCandidateReport>,
    support: &HashMap<String, f32>,
) {
    if reports.len() <= 2 {
        normalize_candidate_weights(reports);
        return;
    }

    let total_support = support
        .values()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f32>();
    if total_support <= f32::EPSILON {
        normalize_candidate_weights(reports);
        return;
    }

    let average_support = total_support / reports.len().max(1) as f32;
    let max_keep = if average_support >= 4.0 {
        4
    } else if average_support >= 2.0 {
        5
    } else {
        6
    };
    let min_weight_floor = if average_support >= 4.0 {
        0.08
    } else if average_support >= 2.0 {
        0.06
    } else {
        0.04
    };

    reports.sort_by(|left, right| right.weight.total_cmp(&left.weight));
    let mut kept = Vec::with_capacity(reports.len().min(max_keep));
    for (idx, report) in reports.iter().cloned().enumerate() {
        if idx < 2 || (idx < max_keep && report.weight >= min_weight_floor) {
            kept.push(report);
        }
    }
    if kept.len() < 2 {
        kept.extend(reports.iter().take(2).cloned());
        kept.dedup_by(|left, right| left.name == right.name);
    }
    *reports = kept;
    normalize_candidate_weights(reports);
}

fn build_candidate_weight_map(reports: &[SwarmCandidateReport]) -> HashMap<String, f32> {
    let total = reports
        .iter()
        .map(|report| report.weight.max(0.0))
        .sum::<f32>();
    reports
        .iter()
        .map(|report| {
            let normalized = if total <= f32::EPSILON {
                1.0 / reports.len().max(1) as f32
            } else {
                report.weight.max(0.0) / total
            };
            (report.name.clone(), normalized)
        })
        .collect()
}

fn write_swarm_artifact_atomically(artifact_path: &Path, payload: &[u8]) -> Result<()> {
    let temp_path = artifact_path.with_extension("tmp");
    let backup_path = artifact_path.with_extension("bak");

    if temp_path.exists() {
        std::fs::remove_file(&temp_path).with_context(|| {
            format!(
                "remove stale swarm forecaster temp artifact {}",
                temp_path.display()
            )
        })?;
    }
    if backup_path.exists() {
        std::fs::remove_file(&backup_path).with_context(|| {
            format!(
                "remove stale swarm forecaster backup artifact {}",
                backup_path.display()
            )
        })?;
    }

    std::fs::write(&temp_path, payload).with_context(|| {
        format!(
            "write swarm forecaster temp artifact {}",
            temp_path.display()
        )
    })?;

    let replaced_existing = if artifact_path.exists() {
        std::fs::rename(artifact_path, &backup_path).with_context(|| {
            format!(
                "rotate previous swarm forecaster artifact {} into {}",
                artifact_path.display(),
                backup_path.display()
            )
        })?;
        true
    } else {
        false
    };

    if let Err(err) = std::fs::rename(&temp_path, artifact_path) {
        if replaced_existing && backup_path.exists() {
            let _ = std::fs::rename(&backup_path, artifact_path);
        }
        let _ = std::fs::remove_file(&temp_path);
        return Err(err).with_context(|| {
            format!(
                "rename swarm forecaster artifact into {}",
                artifact_path.display()
            )
        });
    }

    let _ = std::fs::remove_file(&backup_path);
    Ok(())
}

#[allow(dead_code)]
fn build_candidate_weight_map_from_consensus(
    snapshot: &SwarmForecastSnapshot,
    candidates: &[(String, Vec<f32>)],
) -> HashMap<String, f32> {
    if candidates.is_empty() {
        return HashMap::new();
    }

    let reference = aggregate_average(
        &candidates
            .iter()
            .map(|(_, forecast)| forecast.clone())
            .collect::<Vec<_>>(),
        candidates
            .first()
            .map(|(_, forecast)| forecast.len())
            .expect("non-empty candidate ensemble must provide a forecast horizon"),
        snapshot.last_value,
    );
    let mut reports = candidates
        .iter()
        .map(|(name, forecast)| {
            candidate_report(
                name,
                "local_ensemble",
                forecast,
                "consensus",
                &reference,
                1.0,
                Some(snapshot),
            )
        })
        .collect::<Vec<_>>();
    apply_candidate_weights(&mut reports);
    build_candidate_weight_map(&reports)
}

fn aggregate_weighted(
    candidates: &[(String, Vec<f32>)],
    weights: &HashMap<String, f32>,
    horizon: usize,
    baseline: f32,
) -> Vec<f32> {
    (0..horizon)
        .map(|step| {
            let mut weighted_sum = 0.0_f32;
            let mut total_weight = 0.0_f32;
            for (name, forecast) in candidates {
                if let Some(value) = forecast.get(step).copied()
                    && value.is_finite()
                {
                    let weight = weights
                        .get(name)
                        .copied()
                        .unwrap_or_else(|| 1.0 / candidates.len().max(1) as f32);
                    weighted_sum += value * weight;
                    total_weight += weight;
                }
            }
            if total_weight <= f32::EPSILON {
                baseline
            } else {
                weighted_sum / total_weight
            }
        })
        .collect()
}

fn candidate_effective_models(weights: &HashMap<String, f32>, candidate_count: usize) -> f32 {
    if weights.is_empty() {
        return candidate_count as f32;
    }
    let sum_sq = weights.values().map(|weight| weight * weight).sum::<f32>();
    if sum_sq <= f32::EPSILON {
        candidate_count as f32
    } else {
        (1.0 / sum_sq).min(candidate_count.max(1) as f32)
    }
}

#[cfg(feature = "swarm-forecasting")]
fn build_weighted_reports_external(
    windows: &[(Vec<f32>, Vec<f64>, Vec<f32>)],
    frequency: &str,
    unique_id: &str,
    forecast_horizon: usize,
) -> Result<Vec<SwarmCandidateReport>> {
    let mut accumulators: HashMap<String, SwarmCandidateReport> = HashMap::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut weight_sums: HashMap<String, f32> = HashMap::new();
    let mut validation_loss_sums: HashMap<String, f32> = HashMap::new();
    let mut validation_support: HashMap<String, f32> = HashMap::new();

    for (window_idx, (train_values, train_timestamps, actuals)) in windows.iter().enumerate() {
        let series = build_training_series(train_values, train_timestamps, frequency, unique_id)?;
        let snapshot = infer_snapshot(&series, &TimeSeriesProcessor::new());
        let candidates = candidate_predictions(train_values, &snapshot, actuals.len().max(1));
        let window_weight =
            validation_window_weight(window_idx, windows.len(), train_values.len(), actuals.len());
        let reports = build_candidate_reports(
            &candidates,
            actuals,
            &format!("validation_window_{window_idx}"),
            Some(&snapshot),
        );
        for report in reports {
            let report_name = report.name.clone();
            let entry = accumulators
                .entry(report_name.clone())
                .or_insert_with(|| report.clone());
            if counts.contains_key(&report_name) {
                entry.prediction_mean += report.prediction_mean * window_weight;
                entry.prediction_std += report.prediction_std * window_weight;
                entry.mae += report.mae * window_weight;
                entry.mse += report.mse * window_weight;
                entry.mape += report.mape * window_weight;
                entry.smape += report.smape * window_weight;
                entry.coverage += report.coverage * window_weight;
                entry.bias += report.bias * window_weight;
                entry.directional_accuracy += report.directional_accuracy * window_weight;
                entry.regime_fit += report.regime_fit * window_weight;
                entry.stability_score += report.stability_score * window_weight;
                entry.prediction_length = entry.prediction_length.max(report.prediction_length);
            } else {
                entry.prediction_mean *= window_weight;
                entry.prediction_std *= window_weight;
                entry.mae *= window_weight;
                entry.mse *= window_weight;
                entry.mape *= window_weight;
                entry.smape *= window_weight;
                entry.coverage *= window_weight;
                entry.bias *= window_weight;
                entry.directional_accuracy *= window_weight;
                entry.regime_fit *= window_weight;
                entry.stability_score *= window_weight;
            }
            *counts.entry(report_name.clone()).or_insert(0) += 1;
            *weight_sums.entry(report_name).or_insert(0.0) += window_weight;
            *validation_loss_sums
                .entry(report.name.clone())
                .or_insert(0.0) += report.mse.max(0.0) * window_weight;
            *validation_support.entry(report.name.clone()).or_insert(0.0) += window_weight;
        }
    }

    let mut reports = accumulators
        .into_iter()
        .map(|(name, mut report)| {
            let total_weight = weight_sums
                .get(&name)
                .copied()
                .filter(|weight| *weight > f32::EPSILON)
                .unwrap_or_else(|| counts.get(&name).copied().unwrap_or(1) as f32);
            report.source = "rolling_validation".to_string();
            report.prediction_length = forecast_horizon.max(1);
            report.prediction_mean /= total_weight;
            report.prediction_std /= total_weight;
            report.mae /= total_weight;
            report.mse /= total_weight;
            report.mape /= total_weight;
            report.smape /= total_weight;
            report.coverage /= total_weight;
            report.bias /= total_weight;
            report.directional_accuracy /= total_weight;
            report.regime_fit /= total_weight;
            report.stability_score /= total_weight;
            report
        })
        .collect::<Vec<_>>();
    apply_validation_candidate_weights(&mut reports, &validation_loss_sums, &validation_support);
    prune_validation_candidates(&mut reports, &validation_support);
    Ok(reports)
}

fn build_weighted_reports_local(
    windows: &[(Vec<f32>, Vec<f64>, Vec<f32>)],
    forecast_horizon: usize,
) -> Result<Vec<SwarmCandidateReport>> {
    let mut accumulators: HashMap<String, SwarmCandidateReport> = HashMap::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut weight_sums: HashMap<String, f32> = HashMap::new();
    let mut validation_loss_sums: HashMap<String, f32> = HashMap::new();
    let mut validation_support: HashMap<String, f32> = HashMap::new();

    for (window_idx, (train_values, train_timestamps, actuals)) in windows.iter().enumerate() {
        let snapshot = build_local_snapshot_with_min(train_values, train_timestamps, 8)?;
        let candidates = candidate_forecasts_local(train_values, &snapshot, actuals.len().max(1));
        let window_weight =
            validation_window_weight(window_idx, windows.len(), train_values.len(), actuals.len());
        let reports = candidates
            .iter()
            .map(|(name, forecast)| {
                candidate_report(
                    name,
                    "local_ensemble",
                    forecast,
                    &format!("validation_window_{window_idx}"),
                    actuals,
                    1.0,
                    Some(&snapshot),
                )
            })
            .collect::<Vec<_>>();
        for report in reports {
            let report_name = report.name.clone();
            let entry = accumulators
                .entry(report_name.clone())
                .or_insert_with(|| report.clone());
            if counts.contains_key(&report_name) {
                entry.prediction_mean += report.prediction_mean * window_weight;
                entry.prediction_std += report.prediction_std * window_weight;
                entry.mae += report.mae * window_weight;
                entry.mse += report.mse * window_weight;
                entry.mape += report.mape * window_weight;
                entry.smape += report.smape * window_weight;
                entry.coverage += report.coverage * window_weight;
                entry.bias += report.bias * window_weight;
                entry.directional_accuracy += report.directional_accuracy * window_weight;
                entry.regime_fit += report.regime_fit * window_weight;
                entry.stability_score += report.stability_score * window_weight;
                entry.prediction_length = entry.prediction_length.max(report.prediction_length);
            } else {
                entry.prediction_mean *= window_weight;
                entry.prediction_std *= window_weight;
                entry.mae *= window_weight;
                entry.mse *= window_weight;
                entry.mape *= window_weight;
                entry.smape *= window_weight;
                entry.coverage *= window_weight;
                entry.bias *= window_weight;
                entry.directional_accuracy *= window_weight;
                entry.regime_fit *= window_weight;
                entry.stability_score *= window_weight;
            }
            *counts.entry(report_name.clone()).or_insert(0) += 1;
            *weight_sums.entry(report_name).or_insert(0.0) += window_weight;
            *validation_loss_sums
                .entry(report.name.clone())
                .or_insert(0.0) += report.mse.max(0.0) * window_weight;
            *validation_support.entry(report.name.clone()).or_insert(0.0) += window_weight;
        }
    }

    let mut reports = accumulators
        .into_iter()
        .map(|(name, mut report)| {
            let total_weight = weight_sums
                .get(&name)
                .copied()
                .filter(|weight| *weight > f32::EPSILON)
                .unwrap_or_else(|| counts.get(&name).copied().unwrap_or(1) as f32);
            report.source = "rolling_validation".to_string();
            report.prediction_length = forecast_horizon.max(1);
            report.prediction_mean /= total_weight;
            report.prediction_std /= total_weight;
            report.mae /= total_weight;
            report.mse /= total_weight;
            report.mape /= total_weight;
            report.smape /= total_weight;
            report.coverage /= total_weight;
            report.bias /= total_weight;
            report.directional_accuracy /= total_weight;
            report.regime_fit /= total_weight;
            report.stability_score /= total_weight;
            report
        })
        .collect::<Vec<_>>();
    apply_validation_candidate_weights(&mut reports, &validation_loss_sums, &validation_support);
    prune_validation_candidates(&mut reports, &validation_support);
    Ok(reports)
}

fn validation_window_weight(
    window_idx: usize,
    total_windows: usize,
    training_rows: usize,
    validation_rows: usize,
) -> f32 {
    let recency = if total_windows <= 1 {
        1.0
    } else {
        0.65 + 0.35 * (window_idx as f32 / (total_windows - 1) as f32)
    };
    let sample_depth = ((training_rows + validation_rows).max(1) as f32)
        .ln_1p()
        .max(1.0);
    recency * sample_depth
}

fn build_training_report(
    reports: &[SwarmCandidateReport],
    validation_windows: usize,
    training_rows: usize,
    horizon: usize,
    diversity_score: f32,
    regime_bias: f32,
) -> SwarmTrainingReport {
    let total_weight = reports
        .iter()
        .map(|report| report.weight.max(0.0))
        .sum::<f32>()
        .max(f32::EPSILON);
    let weighted_mean = |selector: fn(&SwarmCandidateReport) -> f32| {
        reports
            .iter()
            .map(|report| selector(report) * report.weight.max(0.0))
            .sum::<f32>()
            / total_weight
    };
    let best_candidate = reports
        .iter()
        .max_by(|left, right| left.weight.total_cmp(&right.weight))
        .map(|report| report.name.clone());
    SwarmTrainingReport {
        training_rows,
        validation_windows,
        fitted_horizon: horizon,
        best_candidate,
        aggregate_mae: weighted_mean(|report| report.mae),
        aggregate_smape: weighted_mean(|report| report.smape),
        aggregate_directional_accuracy: weighted_mean(|report| report.directional_accuracy),
        aggregate_coverage: weighted_mean(|report| report.coverage),
        diversity_score,
        regime_bias,
        updated_at_unix_ms: current_unix_ms(),
    }
}

#[cfg(feature = "swarm-forecasting")]
fn build_candidate_reports(
    candidates: &[(String, ModelType, Vec<f32>)],
    reference: &[f32],
    source: &str,
    snapshot: Option<&SwarmForecastSnapshot>,
) -> Vec<SwarmCandidateReport> {
    let weight = if candidates.is_empty() {
        1.0
    } else {
        1.0 / candidates.len() as f32
    };
    candidates
        .iter()
        .map(|(name, model_type, forecast)| {
            let model_type_label = format!("{model_type:?}");
            candidate_report(
                name,
                &model_type_label,
                forecast,
                source,
                reference,
                weight,
                snapshot,
            )
        })
        .collect::<Vec<_>>()
}

#[cfg(feature = "swarm-forecasting")]
fn model_performance_metrics(prediction: &[f32], reference: &[f32]) -> ModelPerformanceMetrics {
    let band = prediction_stats(prediction).1.max(1e-6);
    ModelPerformanceMetrics {
        mae: mae(prediction, reference),
        mse: mse(prediction, reference),
        mape: mape(prediction, reference),
        smape: smape(prediction, reference),
        coverage: interval_coverage(prediction, reference, band),
    }
}

#[cfg(feature = "swarm-forecasting")]
fn normalize_prediction_intervals(lower: Vec<f32>, upper: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    let mut lower = lower;
    let mut upper = upper;
    for (lower_value, upper_value) in lower.iter_mut().zip(upper.iter_mut()) {
        if *lower_value > *upper_value {
            std::mem::swap(lower_value, upper_value);
        }
    }
    (lower, upper)
}

#[allow(dead_code)]
fn candidate_forecasts_local(
    values: &[f32],
    snapshot: &SwarmForecastSnapshot,
    horizon: usize,
) -> Vec<(String, Vec<f32>)> {
    let trend_weight = snapshot.trend_strength.clamp(0.1, 0.9);
    let mean_reversion = snapshot.mean_reversion_strength.clamp(0.05, 0.9);
    let trend_dominant = swarm_trend_dominant(snapshot);
    let mean_reversion_dominant = swarm_mean_reversion_dominant(snapshot);
    let high_volatility = swarm_high_volatility(snapshot);
    let mut predictions = vec![(
        "persistence".to_string(),
        persistence_forecast(snapshot.last_value, horizon),
    )];

    predictions.push((
        "moving_average_medium".to_string(),
        moving_average_forecast(values, horizon, 8),
    ));

    if mean_reversion_dominant || high_volatility {
        predictions.push((
            "moving_average_short".to_string(),
            moving_average_forecast(values, horizon, 4),
        ));
    }

    if trend_dominant || !high_volatility {
        predictions.push((
            "moving_average_long".to_string(),
            moving_average_forecast(values, horizon, 16),
        ));
    }

    if trend_dominant || snapshot.recent_return.abs() > snapshot.volatility.max(1e-6) * 0.35 {
        predictions.push((
            "ewma_fast".to_string(),
            ewma_forecast(values, horizon, 0.45),
        ));
    }

    if !trend_dominant || snapshot.volatility_ratio < 1.4 {
        predictions.push((
            "ewma_slow".to_string(),
            ewma_forecast(values, horizon, 0.22),
        ));
    }

    if trend_dominant || snapshot.has_trend {
        predictions.push((
            "damped_drift".to_string(),
            damped_drift_forecast(snapshot.last_value, snapshot.drift_slope, horizon, 0.82),
        ));
        predictions.push((
            "drift".to_string(),
            drift_forecast(snapshot.last_value, snapshot.drift_slope, horizon),
        ));
    }

    if mean_reversion_dominant || !snapshot.has_trend || high_volatility {
        predictions.push((
            "mean_reversion".to_string(),
            mean_reversion_forecast(
                snapshot.last_value,
                snapshot.rolling_mean,
                mean_reversion,
                horizon,
            ),
        ));
        predictions.push((
            "regime_anchor".to_string(),
            blend_forecasts(
                &moving_average_forecast(values, horizon, 12),
                &mean_reversion_forecast(
                    snapshot.last_value,
                    snapshot.medium_mean.max(snapshot.rolling_mean),
                    mean_reversion,
                    horizon,
                ),
                0.5 + 0.3 * snapshot.trend_strength,
            ),
        ));
    }

    if trend_dominant && snapshot.recent_return.abs() > 1e-6 {
        predictions.push((
            "momentum_blend".to_string(),
            blend_forecasts(
                &damped_drift_forecast(snapshot.last_value, snapshot.drift_slope, horizon, 0.90),
                &ewma_forecast(values, horizon, 0.35),
                trend_weight,
            ),
        ));
    }

    if snapshot.has_seasonality
        && let Some(seasonal) = seasonal_forecast(values, &snapshot.seasonal_periods, horizon)
    {
        predictions.push(("seasonal".to_string(), seasonal));
        if trend_dominant {
            predictions.push((
                "seasonal_drift".to_string(),
                blend_forecasts(
                    predictions
                        .iter()
                        .find(|(name, _)| name == "drift")
                        .map(|(_, forecast)| forecast.as_slice())
                        .unwrap_or(&[]),
                    predictions
                        .iter()
                        .find(|(name, _)| name == "seasonal")
                        .map(|(_, forecast)| forecast.as_slice())
                        .unwrap_or(&[]),
                    0.55 + 0.25 * snapshot.trend_strength,
                ),
            ));
        }
    }

    prune_local_candidates(snapshot, predictions)
}

fn prune_local_candidates(
    snapshot: &SwarmForecastSnapshot,
    candidates: Vec<(String, Vec<f32>)>,
) -> Vec<(String, Vec<f32>)> {
    if candidates.len() <= 3 {
        return candidates;
    }

    let mut scored = candidates
        .into_iter()
        .map(|candidate| {
            let score = candidate_preselection_score(&candidate.0, &candidate.1, snapshot);
            (score, candidate)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.0.total_cmp(&left.0));

    let duplicate_threshold = if snapshot.volatility_ratio > 1.25 {
        0.05
    } else {
        0.035
    };
    let scale = snapshot
        .volatility
        .max(snapshot.last_value.abs() * 0.005)
        .max(1e-4);
    let max_keep = if snapshot.has_seasonality { 6 } else { 5 };

    let mut kept = Vec::new();
    for (_, candidate) in scored {
        let is_duplicate = kept.iter().any(|existing: &(String, Vec<f32>)| {
            forecast_distance_ratio(&candidate.1, &existing.1, scale) <= duplicate_threshold
        });
        if !is_duplicate || kept.len() < 2 {
            kept.push(candidate);
        }
        if kept.len() >= max_keep {
            break;
        }
    }

    kept
}

#[allow(dead_code)]
fn build_local_snapshot(values: &[f32], timestamps: &[f64]) -> Result<SwarmForecastSnapshot> {
    build_local_snapshot_with_min(values, timestamps, 32)
}

fn snapshot_rebuild_min_observations(observations: usize) -> usize {
    observations.clamp(8, 32)
}

fn build_local_snapshot_with_min(
    values: &[f32],
    timestamps: &[f64],
    min_observations: usize,
) -> Result<SwarmForecastSnapshot> {
    if values.len() != timestamps.len() {
        bail!(
            "swarm forecaster value/timestamp mismatch: {} values vs {} timestamps",
            values.len(),
            timestamps.len()
        );
    }
    if values.len() < min_observations {
        bail!(
            "swarm forecaster requires at least {} observations, received {}",
            min_observations,
            values.len()
        );
    }

    let last_window = values.len().min(32);
    let window = &values[values.len() - last_window..];
    let short_window = values.len().min(8);
    let medium_window = values.len().min(16);
    let long_window = values.len().min(32);
    let rolling_mean = window.iter().copied().sum::<f32>() / window.len() as f32;
    let short_mean = values[values.len() - short_window..]
        .iter()
        .copied()
        .sum::<f32>()
        / short_window as f32;
    let medium_mean = values[values.len() - medium_window..]
        .iter()
        .copied()
        .sum::<f32>()
        / medium_window as f32;
    let long_mean = values[values.len() - long_window..]
        .iter()
        .copied()
        .sum::<f32>()
        / long_window as f32;
    let slope = trend_slope(window);
    let window_volatility = volatility(window);
    let baseline_volatility = volatility(values).max(1e-6);
    let mut seasonal_periods = Vec::new();

    for period in [4_usize, 6, 8, 12, 24, 48] {
        if values.len() >= period * 2 {
            let recent = &values[values.len() - period..];
            let prior = &values[values.len() - 2 * period..values.len() - period];
            let recent_mean = recent.iter().copied().sum::<f32>() / recent.len() as f32;
            let prior_mean = prior.iter().copied().sum::<f32>() / prior.len() as f32;
            let covariance = recent
                .iter()
                .zip(prior.iter())
                .map(|(a, b)| (*a - recent_mean) * (*b - prior_mean))
                .sum::<f32>();
            let recent_var = recent
                .iter()
                .map(|value| (*value - recent_mean).powi(2))
                .sum::<f32>();
            let prior_var = prior
                .iter()
                .map(|value| (*value - prior_mean).powi(2))
                .sum::<f32>();
            let denominator = (recent_var * prior_var).sqrt();
            let correlation = if denominator <= 1e-6 {
                0.0
            } else {
                covariance / denominator
            };
            if correlation.is_finite() && correlation > 0.35 {
                seasonal_periods.push(period);
            }
        }
    }

    Ok(SwarmForecastSnapshot {
        last_value: *values
            .last()
            .expect("local swarm snapshot requires at least one value"),
        rolling_mean,
        drift_slope: slope,
        volatility: window_volatility,
        has_trend: slope.abs() > 1e-3,
        has_seasonality: !seasonal_periods.is_empty(),
        seasonal_periods,
        short_mean,
        medium_mean,
        long_mean,
        recent_return: values
            .last()
            .zip(values.get(values.len().saturating_sub(2)))
            .map(|(last, prev)| last - prev)
            .unwrap_or_default(),
        trend_strength: clamp_unit(
            ((short_mean - long_mean).abs() + slope.abs()) / (window_volatility + 1e-6),
        ),
        mean_reversion_strength: clamp_unit(
            (1.0 - ((short_mean - rolling_mean).abs() / (window_volatility + 1e-6))).max(0.0),
        ),
        volatility_ratio: (window_volatility / baseline_volatility).max(0.0),
    })
}

fn build_validation_windows(
    values: &[f32],
    timestamps: &[f64],
    horizon: usize,
) -> Vec<(Vec<f32>, Vec<f64>, Vec<f32>)> {
    let validation_window = horizon.max(8).min(values.len() / 6).max(4);
    let min_training_rows = horizon.max(16).max(validation_window * 2);
    if values.len() <= min_training_rows + validation_window {
        return Vec::new();
    }

    let mut windows = Vec::new();
    let stride = (validation_window / 2).max(2);
    let mut window_end = values.len();
    while windows.len() < 5 && window_end > min_training_rows + validation_window {
        let train_end = window_end - validation_window;
        if train_end < min_training_rows {
            break;
        }
        windows.push((
            values[..train_end].to_vec(),
            timestamps[..train_end].to_vec(),
            values[train_end..window_end].to_vec(),
        ));
        window_end = window_end.saturating_sub(stride);
    }
    windows.reverse();
    windows
}

fn is_price_like_column_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    [
        "close", "open", "high", "low", "price", "mid", "bid", "ask", "last", "hl2", "hlc3",
        "ohlc4", "typical", "weighted", "wap", "vwap",
    ]
    .iter()
    .any(|needle| name.contains(needle))
}

fn label_series_is_class_like(values: &[f32]) -> bool {
    if values.is_empty() {
        return true;
    }
    let mut unique = values.to_vec();
    unique.sort_by(|left, right| left.total_cmp(right));
    unique.dedup_by(|left, right| (*left - *right).abs() < 1e-6);
    unique.len() <= 4
        || values.iter().all(|value| {
            [-1.0_f32, 0.0, 1.0, 2.0]
                .iter()
                .any(|class| (*value - *class).abs() < 1e-6)
        })
}

fn extract_continuous_label_series(labels: &Series) -> Result<Option<Vec<f32>>> {
    let Ok(series) = labels.cast(&DataType::Float64) else {
        return Ok(None);
    };
    let values = series
        .f64()
        .context("access swarm label series as Float64")?
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            let value = value.with_context(|| {
                format!(
                    "swarm label series contains null at row {idx}; continuous forecasting labels must be fully materialized"
                )
            })?;
            if !value.is_finite() {
                bail!(
                    "swarm label series contains non-finite value {} at row {}",
                    value,
                    idx
                );
            }
            Ok(value as f32)
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() < 32 || label_series_is_class_like(&values) || volatility(&values) <= 1e-8 {
        return Ok(None);
    }
    Ok(Some(values))
}

fn extract_series_from_frame(frame: &DataFrame, labels: &Series) -> Result<Vec<f32>> {
    let preferred_columns = [
        "close",
        "base_close",
        "mid",
        "price",
        "bid",
        "ask",
        "last",
        "target_price",
        "future_close",
        "next_close",
        "close_M1",
        "close_m1",
    ];

    for column_name in preferred_columns {
        if let Ok(column) = frame.column(column_name) {
            let series = column
                .cast(&DataType::Float64)
                .with_context(|| format!("cast swarm source column {column_name} to Float64"))?;
            let values = series
                .f64()
                .context("access swarm source column as Float64")?
                .into_iter()
                .enumerate()
                .map(|(idx, value)| {
                    let value = value.with_context(|| {
                        format!(
                            "swarm source column {column_name} contains null at row {idx}; forecasting requires fully materialized series"
                        )
                    })?;
                    if !value.is_finite() {
                        bail!(
                            "swarm source column {column_name} contains non-finite value {} at row {}",
                            value,
                            idx
                        );
                    }
                    Ok(value as f32)
                })
                .collect::<Result<Vec<_>>>()?;
            if values.len() >= 32 {
                return Ok(values);
            }
        }
    }

    for column in frame.get_columns() {
        if preferred_columns.contains(&column.name().as_str()) {
            continue;
        }
        let name = column.name().as_str().to_ascii_lowercase();
        if name.contains("label") || name.contains("target") || name.contains("signal") {
            continue;
        }
        if !is_price_like_column_name(&name) {
            continue;
        }
        if !is_numeric_dtype(column.dtype()) {
            continue;
        }
        let series = column.cast(&DataType::Float64).with_context(|| {
            format!(
                "cast fallback swarm source column {} to Float64",
                column.name().as_str()
            )
        })?;
        let values = series
            .f64()
            .context("access fallback swarm source column as Float64")?
            .into_iter()
            .enumerate()
            .map(|(idx, value)| {
                let value = value.with_context(|| {
                    format!(
                        "fallback swarm source column {} contains null at row {}; forecasting requires fully materialized series",
                        column.name().as_str(),
                        idx
                    )
                })?;
                if !value.is_finite() {
                    bail!(
                        "fallback swarm source column {} contains non-finite value {} at row {}",
                        column.name().as_str(),
                        value,
                        idx
                    );
                }
                Ok(value as f32)
            })
            .collect::<Result<Vec<_>>>()?;
        if values.len() >= 32 {
            return Ok(values);
        }
    }

    if let Some(values) = extract_continuous_label_series(labels)? {
        return Ok(values);
    }

    let candidate_numeric_columns = frame
        .get_columns()
        .iter()
        .filter(|column| {
            let name = column.name().as_str().to_ascii_lowercase();
            !name.contains("label")
                && !name.contains("target")
                && !name.contains("signal")
                && is_numeric_dtype(column.dtype())
        })
        .map(|column| column.name().as_str().to_string())
        .collect::<Vec<_>>();

    bail!(
        "swarm forecaster could not derive a price-like series from the training frame; price-like columns are required and synthetic row-mean reconstruction is disabled. numeric columns seen: {}",
        if candidate_numeric_columns.is_empty() {
            "<none>".to_string()
        } else {
            candidate_numeric_columns.join(", ")
        }
    )
}

pub struct SwarmForecaster {
    #[cfg(feature = "swarm-forecasting")]
    pub manager: Option<AgentForecastingManager>,
    #[cfg(not(feature = "swarm-forecasting"))]
    pub manager: Option<()>,
    pub config: SwarmForecastConfig,
    runtime_mode: SwarmRuntimeMode,
    runtime_degraded_reason: Option<String>,
    pub fitted: bool,
    pub values: Vec<f32>,
    pub timestamps: Vec<f64>,
    pub unique_id: String,
    pub snapshot: Option<SwarmForecastSnapshot>,
    pub last_result: Option<SwarmForecastResult>,
    pub last_horizon: Option<usize>,
    candidate_reports: Vec<SwarmCandidateReport>,
    pub updated_at_unix_ms: Option<u64>,
    training_report: Option<SwarmTrainingReport>,
}

impl SwarmForecaster {
    pub fn new(memory_limit_mb: f64) -> Self {
        let config = SwarmForecastConfig {
            memory_limit_mb: memory_limit_mb as f32,
            ..SwarmForecastConfig::default()
        };

        #[cfg(feature = "swarm-forecasting")]
        {
            Self {
                manager: Some(AgentForecastingManager::new(config.memory_limit_mb)),
                config,
                runtime_mode: SwarmRuntimeMode::ExternalSwarm,
                runtime_degraded_reason: None,
                fitted: false,
                values: Vec::new(),
                timestamps: Vec::new(),
                unique_id: "swarm_series".to_string(),
                snapshot: None,
                last_result: None,
                last_horizon: None,
                candidate_reports: Vec::new(),
                updated_at_unix_ms: None,
                training_report: None,
            }
        }
        #[cfg(not(feature = "swarm-forecasting"))]
        {
            Self {
                manager: None,
                config,
                runtime_mode: SwarmRuntimeMode::LocalFallback,
                runtime_degraded_reason: None,
                fitted: false,
                values: Vec::new(),
                timestamps: Vec::new(),
                unique_id: "swarm_series".to_string(),
                snapshot: None,
                last_result: None,
                last_horizon: None,
                candidate_reports: Vec::new(),
                updated_at_unix_ms: None,
                training_report: None,
            }
        }
    }

    pub fn with_agent(
        mut self,
        agent_id: impl Into<String>,
        agent_type: impl Into<String>,
    ) -> Self {
        self.config.agent_id = agent_id.into();
        self.config.agent_type = agent_type.into();
        self
    }

    pub fn with_strategy(mut self, strategy: SwarmEnsembleStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn with_frequency(mut self, frequency: impl Into<String>) -> Self {
        self.config.frequency = frequency.into();
        self
    }

    pub fn fit_from_frame(
        &mut self,
        frame: &DataFrame,
        labels: &Series,
        unique_id: impl Into<String>,
    ) -> Result<()> {
        let values = extract_series_from_frame(frame, labels)?;
        let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
        self.fit_series(&values, &timestamps, unique_id)
    }

    #[cfg(feature = "swarm-forecasting")]
    pub fn fit_series(
        &mut self,
        values: &[f32],
        timestamps: &[f64],
        unique_id: impl Into<String>,
    ) -> Result<()> {
        let unique_id = unique_id.into();
        let series = build_training_series(values, timestamps, &self.config.frequency, &unique_id)?;
        let processor = TimeSeriesProcessor::new();
        let snapshot = infer_snapshot(&series, &processor);

        let requirements = ForecastRequirements {
            horizon: self.config.horizon,
            frequency: self.config.frequency.clone(),
            accuracy_target: self.config.accuracy_target,
            latency_requirement_ms: self.config.latency_requirement_ms,
            interpretability_needed: self.config.interpretability_needed,
            online_learning: self.config.online_learning,
        };

        let manager = self
            .manager
            .as_mut()
            .context("swarm forecasting manager missing")?;
        manager
            .assign_model(
                self.config.agent_id.clone(),
                self.config.agent_type.clone(),
                requirements,
            )
            .map_err(|err| anyhow::anyhow!("assign swarm forecasting model: {err}"))?;

        self.values = values.to_vec();
        self.timestamps = timestamps.to_vec();
        self.unique_id = unique_id;
        self.snapshot = Some(snapshot);
        self.last_result = None;
        self.last_horizon = None;
        self.candidate_reports.clear();
        self.updated_at_unix_ms = None;
        self.training_report = None;
        self.runtime_mode = SwarmRuntimeMode::ExternalSwarm;
        self.runtime_degraded_reason = None;
        self.fitted = true;
        Ok(())
    }

    #[cfg(not(feature = "swarm-forecasting"))]
    pub fn fit_series(
        &mut self,
        values: &[f32],
        timestamps: &[f64],
        unique_id: impl Into<String>,
    ) -> Result<()> {
        let unique_id = unique_id.into();
        let snapshot = build_local_snapshot_with_min(values, timestamps, 8)?;

        self.values = values.to_vec();
        self.timestamps = timestamps.to_vec();
        self.unique_id = unique_id;
        self.snapshot = Some(snapshot);
        self.last_result = None;
        self.last_horizon = None;
        self.candidate_reports.clear();
        self.updated_at_unix_ms = None;
        self.training_report = None;
        self.runtime_mode = SwarmRuntimeMode::LocalFallback;
        self.runtime_degraded_reason = Some("swarm_forecasting_feature_disabled".to_string());
        self.fitted = true;
        Ok(())
    }

    #[cfg(feature = "swarm-forecasting")]
    pub fn forecast(&mut self, horizon: usize) -> Result<SwarmForecastResult> {
        if !self.fitted {
            bail!("swarm forecaster has not been fitted");
        }

        let snapshot = self
            .snapshot
            .clone()
            .context("swarm forecaster snapshot missing")?;
        let horizon = horizon.max(1);
        let candidates = candidate_predictions(&self.values, &snapshot, horizon);
        if candidates.len() < 2 {
            bail!("swarm forecaster requires at least two candidate forecasts");
        }

        let point_candidates = candidates
            .iter()
            .map(|(_, _, forecast)| forecast.clone())
            .collect::<Vec<_>>();

        let validation_windows = build_validation_windows(&self.values, &self.timestamps, horizon);
        let weighted_reports = if !validation_windows.is_empty() {
            build_weighted_reports_external(
                &validation_windows,
                &self.config.frequency,
                &self.unique_id,
                horizon,
            )?
        } else {
            let report_reference =
                aggregate_average(&point_candidates, horizon, snapshot.last_value);
            build_candidate_reports(&candidates, &report_reference, "consensus", Some(&snapshot))
        };
        let weight_map = build_candidate_weight_map(&weighted_reports);
        let ordered_weights = candidates
            .iter()
            .map(|(name, _, _)| {
                weight_map
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| 1.0 / candidates.len().max(1) as f32)
            })
            .collect::<Vec<_>>();
        let config = EnsembleConfig {
            strategy: map_strategy(self.config.strategy),
            models: candidates.iter().map(|(name, _, _)| name.clone()).collect(),
            weights: Some(ordered_weights),
            meta_learner: Some("meta_stack".to_string()),
            optimization_metric: OptimizationMetric::CombinedScore,
        };
        let mut ensemble = EnsembleForecaster::new(config).map_err(|err| anyhow::anyhow!(err))?;
        for (name, model_type, forecast) in &candidates {
            ensemble.add_model(EnsembleModel {
                name: name.clone(),
                model_type: *model_type,
                weight: weight_map
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| 1.0 / candidates.len().max(1) as f32),
                performance_metrics: model_performance_metrics(
                    forecast,
                    &aggregate_average(&point_candidates, horizon, snapshot.last_value),
                ),
            });
        }
        self.candidate_reports = weighted_reports.clone();
        let ensemble_result = ensemble
            .ensemble_predict(&point_candidates)
            .map_err(|err| anyhow::anyhow!("swarm ensemble forecast: {err}"))?;
        let ensemble_swarm_result = {
            let (level_80_lower, level_80_upper) = normalize_prediction_intervals(
                ensemble_result.prediction_intervals.level_80.0,
                ensemble_result.prediction_intervals.level_80.1,
            );
            SwarmForecastResult {
                point_forecast: ensemble_result.point_forecast,
                level_80_lower,
                level_80_upper,
                diversity_score: ensemble_result.ensemble_metrics.diversity_score,
                effective_models: ensemble_result.ensemble_metrics.effective_models,
                prediction_variance: ensemble_result.ensemble_metrics.prediction_variance,
                models_used: ensemble_result.models_used,
            }
        };
        let fallback_candidates = candidates
            .iter()
            .map(|(name, _, forecast)| (name.clone(), forecast.clone()))
            .collect::<Vec<_>>();
        let (final_result, runtime_mode, runtime_degraded_reason) =
            select_external_or_fallback_result(
                &snapshot,
                &fallback_candidates,
                &weight_map,
                &self.candidate_reports,
                self.config.strategy,
                horizon,
                ensemble_swarm_result,
            );

        if let Some(manager) = self.manager.as_mut() {
            let confidence = (1.0 - final_result.prediction_variance).clamp(0.0, 1.0);
            let accuracy_proxy = (1.0 - snapshot.volatility.abs()).clamp(0.0, 1.0);
            let _update_result = manager.update_performance(
                &self.config.agent_id,
                self.config.latency_requirement_ms.min(50.0),
                accuracy_proxy,
                confidence,
            );
        }

        self.last_result = Some(final_result.clone());
        self.last_horizon = Some(horizon);
        self.updated_at_unix_ms = current_unix_ms();
        self.runtime_mode = runtime_mode;
        self.runtime_degraded_reason = runtime_degraded_reason;
        self.training_report = Some(build_training_report(
            &self.candidate_reports,
            validation_windows.len(),
            self.values.len(),
            horizon,
            final_result.diversity_score,
            snapshot.trend_strength - snapshot.mean_reversion_strength,
        ));

        Ok(final_result)
    }

    #[cfg(not(feature = "swarm-forecasting"))]
    pub fn forecast(&mut self, horizon: usize) -> Result<SwarmForecastResult> {
        if !self.fitted {
            bail!("swarm forecaster has not been fitted");
        }

        let horizon = horizon.max(1);
        let snapshot = self
            .snapshot
            .clone()
            .context("swarm forecaster snapshot missing")?;
        let candidates = candidate_forecasts_local(&self.values, &snapshot, horizon);
        if candidates.len() < 2 {
            bail!("swarm forecaster requires at least two candidate forecasts");
        }

        let report_reference = aggregate_average(
            &candidates
                .iter()
                .map(|(_, forecast)| forecast.clone())
                .collect::<Vec<_>>(),
            horizon,
            snapshot.last_value,
        );
        let validation_windows = build_validation_windows(&self.values, &self.timestamps, horizon);
        self.candidate_reports = if !validation_windows.is_empty() {
            build_weighted_reports_local(&validation_windows, horizon)?
        } else {
            let mut reports = candidates
                .iter()
                .map(|(name, forecast)| {
                    candidate_report(
                        name,
                        "local_ensemble",
                        forecast,
                        "consensus",
                        &report_reference,
                        1.0 / candidates.len() as f32,
                        Some(&snapshot),
                    )
                })
                .collect::<Vec<_>>();
            apply_candidate_weights(&mut reports);
            reports
        };
        let weight_map = build_candidate_weight_map(&self.candidate_reports);
        let final_result = fallback_forecast_with_strategy(
            &snapshot,
            &candidates,
            &weight_map,
            &self.candidate_reports,
            self.config.strategy,
            horizon,
        );
        self.last_result = Some(final_result.clone());
        self.last_horizon = Some(horizon);
        self.updated_at_unix_ms = current_unix_ms();
        self.runtime_mode = SwarmRuntimeMode::LocalFallback;
        self.runtime_degraded_reason = Some("swarm_forecasting_feature_disabled".to_string());
        self.training_report = Some(build_training_report(
            &self.candidate_reports,
            validation_windows.len(),
            self.values.len(),
            horizon,
            final_result.diversity_score,
            snapshot.trend_strength - snapshot.mean_reversion_strength,
        ));

        Ok(final_result)
    }

    pub fn train(&mut self) -> Result<()> {
        if self.values.is_empty() || self.timestamps.is_empty() {
            bail!("swarm forecaster cannot train without historical values");
        }
        self.fit_series(
            &self.values.clone(),
            &self.timestamps.clone(),
            self.unique_id.clone(),
        )
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if self.fitted && self.snapshot.is_none() {
            bail!("swarm forecaster cannot save a fitted model without a snapshot");
        }
        std::fs::create_dir_all(path)
            .with_context(|| format!("create swarm forecaster directory {}", path.display()))?;
        let artifact = sanitize_forecaster_artifact(SwarmForecasterArtifact {
            config: self.config.clone(),
            runtime_mode: self.runtime_mode,
            runtime_degraded_reason: self.runtime_degraded_reason.clone(),
            fitted: self.fitted,
            values: self.values.clone(),
            timestamps: self.timestamps.clone(),
            unique_id: self.unique_id.clone(),
            snapshot: self.snapshot.clone(),
            last_result: self.last_result.clone(),
            last_horizon: self.last_horizon,
            candidate_reports: self.candidate_reports.clone(),
            updated_at_unix_ms: self.updated_at_unix_ms,
            training_report: self.training_report.clone(),
        })?;
        let payload =
            serde_json::to_vec_pretty(&artifact).context("serialize swarm forecaster artifact")?;
        let artifact_path = path.join(SWARM_ARTIFACT_FILE_NAME);
        write_swarm_artifact_atomically(&artifact_path, &payload)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let payload = std::fs::read(path.join(SWARM_ARTIFACT_FILE_NAME))
            .with_context(|| format!("read swarm forecaster artifact {}", path.display()))?;
        let artifact: SwarmForecasterArtifact =
            serde_json::from_slice(&payload).context("deserialize swarm forecaster artifact")?;
        let artifact = sanitize_forecaster_artifact(artifact)?;

        let SwarmForecasterArtifact {
            config,
            runtime_mode,
            runtime_degraded_reason,
            fitted,
            values,
            timestamps,
            unique_id,
            snapshot,
            last_result,
            last_horizon,
            candidate_reports,
            updated_at_unix_ms,
            training_report,
        } = artifact;

        let mut next_state = Self::new(config.memory_limit_mb as f64);
        next_state.config = config;
        next_state.runtime_mode = runtime_mode;
        next_state.runtime_degraded_reason = runtime_degraded_reason;
        next_state.fitted = fitted;
        next_state.values = values;
        next_state.timestamps = timestamps;
        next_state.unique_id = unique_id;
        next_state.snapshot = snapshot;
        next_state.last_result = last_result.clone();
        next_state.last_horizon = last_horizon;
        next_state.candidate_reports = candidate_reports.clone();
        next_state.updated_at_unix_ms = updated_at_unix_ms;
        next_state.training_report = training_report;

        #[cfg(feature = "swarm-forecasting")]
        {
            next_state.manager = Some(AgentForecastingManager::new(
                next_state.config.memory_limit_mb,
            ));
            if next_state.fitted {
                let manager = next_state
                    .manager
                    .as_mut()
                    .context("swarm forecasting manager missing after load")?;
                let requirements = ForecastRequirements {
                    horizon: next_state.config.horizon,
                    frequency: next_state.config.frequency.clone(),
                    accuracy_target: next_state.config.accuracy_target,
                    latency_requirement_ms: next_state.config.latency_requirement_ms,
                    interpretability_needed: next_state.config.interpretability_needed,
                    online_learning: next_state.config.online_learning,
                };
                manager
                    .assign_model(
                        next_state.config.agent_id.clone(),
                        next_state.config.agent_type.clone(),
                        requirements,
                    )
                    .map_err(|err| anyhow::anyhow!("assign swarm forecasting model: {err}"))?;
                if next_state.snapshot.is_none() {
                    next_state.snapshot = Some(build_local_snapshot_with_min(
                        &next_state.values,
                        &next_state.timestamps,
                        snapshot_rebuild_min_observations(next_state.values.len()),
                    )?);
                }
            }
        }

        #[cfg(not(feature = "swarm-forecasting"))]
        {
            if next_state.fitted && next_state.snapshot.is_none() {
                next_state.snapshot = Some(build_local_snapshot_with_min(
                    &next_state.values,
                    &next_state.timestamps,
                    snapshot_rebuild_min_observations(next_state.values.len()),
                )?);
            }
            if next_state.fitted && next_state.runtime_mode == SwarmRuntimeMode::ExternalSwarm {
                next_state.runtime_mode = SwarmRuntimeMode::LocalFallback;
                next_state.runtime_degraded_reason =
                    Some("swarm_forecasting_feature_disabled".to_string());
            }
        }

        *self = next_state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::NamedFrom;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_artifact_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock moved backwards")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "forex-ai-swarm-forecasting-{name}-{}-{unique}",
            std::process::id()
        ))
    }

    #[test]
    fn extract_series_rejects_class_like_target_column_as_price_source() {
        let frame = DataFrame::new(vec![
            Series::new(
                "feature".into(),
                (0..64).map(|idx| idx as f64).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "target".into(),
                (0..64)
                    .map(|idx| match idx % 3 {
                        0 => -1,
                        1 => 0,
                        _ => 1,
                    })
                    .collect::<Vec<_>>(),
            )
            .into(),
        ])
        .expect("frame");
        let labels = Series::new(
            "labels".into(),
            (0..64)
                .map(|idx| match idx % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                })
                .collect::<Vec<_>>(),
        );

        let err = extract_series_from_frame(&frame, &labels)
            .expect_err("class-like target column must not be forecast as price");
        assert!(err.to_string().contains("price-like series"));
    }

    #[test]
    fn extract_series_uses_continuous_labels_when_price_column_is_absent() {
        let frame = DataFrame::new(vec![
            Series::new(
                "feature".into(),
                (0..64).map(|idx| idx as f64).collect::<Vec<_>>(),
            )
            .into(),
        ])
        .expect("frame");
        let labels = Series::new(
            "continuous_target".into(),
            (0..64)
                .map(|idx| 1.10_f64 + idx as f64 * 0.0001)
                .collect::<Vec<_>>(),
        );

        let values = extract_series_from_frame(&frame, &labels).expect("continuous label series");
        assert_eq!(values.len(), 64);
        assert!(values[63] > values[0]);
    }

    #[test]
    fn load_preserves_saved_local_fallback_artifacts_without_refitting() {
        let dir = test_artifact_dir("load-preserves-local-artifact");
        fs::create_dir_all(&dir).expect("create temp dir");

        let values = vec![1.0, 1.2, 1.1, 1.4, 1.6, 1.5, 1.7, 1.8];
        let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
        let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");
        let last_result = SwarmForecastResult {
            point_forecast: vec![1.9, 2.0, 2.1],
            level_80_lower: vec![1.7, 1.8, 1.9],
            level_80_upper: vec![2.1, 2.2, 2.3],
            diversity_score: 0.12,
            effective_models: 3.0,
            prediction_variance: 0.04,
            models_used: 3,
        };
        let candidate_reports = vec![SwarmCandidateReport {
            name: "persistence".to_string(),
            model_type: "MLP".to_string(),
            source: "local".to_string(),
            weight: 1.0,
            prediction_length: 3,
            prediction_mean: 2.0,
            prediction_std: 0.2,
            mae: 0.1,
            mse: 0.01,
            mape: 5.0,
            smape: 4.0,
            coverage: 0.8,
            bias: 0.0,
            directional_accuracy: 0.75,
            regime_fit: 0.65,
            stability_score: 0.90,
        }];
        let training_report = SwarmTrainingReport {
            training_rows: values.len(),
            validation_windows: 0,
            fitted_horizon: 3,
            best_candidate: Some("persistence".to_string()),
            aggregate_mae: 0.1,
            aggregate_smape: 4.0,
            aggregate_directional_accuracy: 0.75,
            aggregate_coverage: 0.8,
            diversity_score: 0.12,
            regime_bias: 0.2,
            updated_at_unix_ms: Some(456),
        };

        let artifact = SwarmForecasterArtifact {
            config: SwarmForecastConfig::default(),
            runtime_mode: SwarmRuntimeMode::LocalFallback,
            runtime_degraded_reason: Some("swarm_local_fallback_active".to_string()),
            fitted: true,
            values: values.clone(),
            timestamps: timestamps.clone(),
            unique_id: "test-series".to_string(),
            snapshot: Some(snapshot.clone()),
            last_result: Some(last_result.clone()),
            last_horizon: Some(3),
            candidate_reports: candidate_reports.clone(),
            updated_at_unix_ms: Some(123),
            training_report: Some(training_report.clone()),
        };
        let payload = serde_json::to_vec_pretty(&artifact).expect("serialize artifact");
        fs::write(dir.join(SWARM_ARTIFACT_FILE_NAME), payload).expect("write artifact");

        let mut forecaster = SwarmForecaster::new(256.0);
        forecaster.load(&dir).expect("load local fallback artifact");

        assert_eq!(forecaster.runtime_mode, SwarmRuntimeMode::LocalFallback);
        assert!(forecaster.fitted);
        assert_eq!(forecaster.values, values);
        assert_eq!(forecaster.timestamps, timestamps);
        assert_eq!(forecaster.unique_id, "test-series");
        assert_eq!(forecaster.last_horizon, Some(3));
        assert_eq!(forecaster.updated_at_unix_ms, Some(123));
        let loaded_result = forecaster.last_result.expect("last result after load");
        assert_eq!(loaded_result.point_forecast, last_result.point_forecast);
        assert_eq!(loaded_result.level_80_lower, last_result.level_80_lower);
        assert_eq!(loaded_result.level_80_upper, last_result.level_80_upper);
        assert_eq!(loaded_result.diversity_score, last_result.diversity_score);
        assert_eq!(loaded_result.effective_models, last_result.effective_models);
        assert_eq!(
            loaded_result.prediction_variance,
            last_result.prediction_variance
        );
        assert_eq!(loaded_result.models_used, last_result.models_used);

        assert_eq!(forecaster.candidate_reports.len(), candidate_reports.len());
        let loaded_report = &forecaster.candidate_reports[0];
        let expected_report = &candidate_reports[0];
        assert_eq!(loaded_report.name, expected_report.name);
        assert_eq!(loaded_report.model_type, expected_report.model_type);
        assert_eq!(loaded_report.source, expected_report.source);
        assert_eq!(loaded_report.weight, expected_report.weight);
        assert_eq!(
            loaded_report.prediction_length,
            expected_report.prediction_length
        );
        assert_eq!(
            loaded_report.prediction_mean,
            expected_report.prediction_mean
        );
        assert_eq!(loaded_report.prediction_std, expected_report.prediction_std);
        assert_eq!(loaded_report.mae, expected_report.mae);
        assert_eq!(loaded_report.mse, expected_report.mse);
        assert_eq!(loaded_report.mape, expected_report.mape);
        assert_eq!(loaded_report.smape, expected_report.smape);
        assert_eq!(loaded_report.coverage, expected_report.coverage);
        assert_eq!(loaded_report.bias, expected_report.bias);
        assert_eq!(
            loaded_report.directional_accuracy,
            expected_report.directional_accuracy
        );
        assert_eq!(loaded_report.regime_fit, expected_report.regime_fit);
        assert_eq!(
            loaded_report.stability_score,
            expected_report.stability_score
        );

        let loaded_snapshot = forecaster.snapshot.expect("snapshot after load");
        assert_eq!(loaded_snapshot.last_value, snapshot.last_value);
        assert_eq!(loaded_snapshot.rolling_mean, snapshot.rolling_mean);
        assert_eq!(loaded_snapshot.drift_slope, snapshot.drift_slope);
        assert_eq!(loaded_snapshot.volatility, snapshot.volatility);
        assert_eq!(loaded_snapshot.has_trend, snapshot.has_trend);
        assert_eq!(loaded_snapshot.has_seasonality, snapshot.has_seasonality);
        assert_eq!(loaded_snapshot.seasonal_periods, snapshot.seasonal_periods);
        assert_eq!(loaded_snapshot.short_mean, snapshot.short_mean);
        assert_eq!(loaded_snapshot.medium_mean, snapshot.medium_mean);
        assert_eq!(loaded_snapshot.long_mean, snapshot.long_mean);
        assert_eq!(loaded_snapshot.recent_return, snapshot.recent_return);
        assert_eq!(loaded_snapshot.trend_strength, snapshot.trend_strength);
        assert_eq!(
            loaded_snapshot.mean_reversion_strength,
            snapshot.mean_reversion_strength
        );
        assert_eq!(loaded_snapshot.volatility_ratio, snapshot.volatility_ratio);

        let loaded_training_report = forecaster
            .training_report
            .expect("training report after load");
        assert_eq!(
            loaded_training_report.training_rows,
            training_report.training_rows
        );
        assert_eq!(
            loaded_training_report.validation_windows,
            training_report.validation_windows
        );
        assert_eq!(
            loaded_training_report.fitted_horizon,
            training_report.fitted_horizon
        );
        assert_eq!(
            loaded_training_report.best_candidate,
            training_report.best_candidate
        );
        assert_eq!(
            loaded_training_report.aggregate_mae,
            training_report.aggregate_mae
        );
        assert_eq!(
            loaded_training_report.aggregate_smape,
            training_report.aggregate_smape
        );
        assert_eq!(
            loaded_training_report.aggregate_directional_accuracy,
            training_report.aggregate_directional_accuracy
        );
        assert_eq!(
            loaded_training_report.aggregate_coverage,
            training_report.aggregate_coverage
        );
        assert_eq!(
            loaded_training_report.diversity_score,
            training_report.diversity_score
        );
        assert_eq!(
            loaded_training_report.regime_bias,
            training_report.regime_bias
        );
        assert_eq!(
            loaded_training_report.updated_at_unix_ms,
            training_report.updated_at_unix_ms
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rebuilds_stale_fitted_artifact_diagnostics() {
        let dir = test_artifact_dir("rebuilds-stale-artifact");
        fs::create_dir_all(&dir).expect("create temp dir");

        let values = (0..96)
            .map(|idx| 1.0 + idx as f32 * 0.01 + ((idx % 7) as f32 - 3.0) * 0.02)
            .collect::<Vec<_>>();
        let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
        let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");

        let artifact = SwarmForecasterArtifact {
            config: SwarmForecastConfig {
                horizon: 12,
                ..SwarmForecastConfig::default()
            },
            runtime_mode: SwarmRuntimeMode::LocalFallback,
            runtime_degraded_reason: Some("stale_external_state".to_string()),
            fitted: true,
            values: values.clone(),
            timestamps: timestamps.clone(),
            unique_id: "stale-series".to_string(),
            snapshot: Some(snapshot),
            last_result: Some(SwarmForecastResult {
                point_forecast: vec![1.0, 1.1],
                level_80_lower: vec![0.9, 1.0],
                level_80_upper: vec![1.1, 1.2],
                diversity_score: 0.0,
                effective_models: 1.0,
                prediction_variance: 0.0,
                models_used: 1,
            }),
            last_horizon: Some(12),
            candidate_reports: vec![SwarmCandidateReport {
                name: "stale".to_string(),
                model_type: "local".to_string(),
                source: "stale".to_string(),
                weight: 1.0,
                prediction_length: 2,
                prediction_mean: 1.05,
                prediction_std: 0.1,
                mae: 0.2,
                mse: 0.04,
                mape: 10.0,
                smape: 8.0,
                coverage: 0.5,
                bias: 0.0,
                directional_accuracy: 0.5,
                regime_fit: 0.5,
                stability_score: 0.5,
            }],
            updated_at_unix_ms: None,
            training_report: Some(SwarmTrainingReport {
                training_rows: values.len(),
                validation_windows: 0,
                fitted_horizon: 6,
                best_candidate: Some("stale".to_string()),
                aggregate_mae: 0.2,
                aggregate_smape: 8.0,
                aggregate_directional_accuracy: 0.5,
                aggregate_coverage: 0.5,
                diversity_score: 0.0,
                regime_bias: 0.0,
                updated_at_unix_ms: None,
            }),
        };
        let payload = serde_json::to_vec_pretty(&artifact).expect("serialize artifact");
        fs::write(dir.join(SWARM_ARTIFACT_FILE_NAME), payload).expect("write artifact");

        let mut forecaster = SwarmForecaster::new(256.0);
        forecaster.load(&dir).expect("load stale artifact");

        assert!(forecaster.fitted);
        assert_eq!(forecaster.last_horizon, Some(12));
        assert!(forecaster.last_result.is_some());
        assert!(!forecaster.candidate_reports.is_empty());
        assert!(forecaster.training_report.is_some());
        assert!(
            forecaster
                .last_result
                .as_ref()
                .is_some_and(|result| result_is_valid(result, 12))
        );
        assert!(candidate_reports_are_valid(
            &forecaster.candidate_reports,
            12
        ));
        assert!(
            forecaster
                .training_report
                .as_ref()
                .is_some_and(|report| report.fitted_horizon == 12)
        );
        assert_eq!(
            forecaster.runtime_degraded_reason.as_deref(),
            Some("swarm_local_fallback_active")
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(not(feature = "swarm-forecasting"))]
    #[test]
    fn load_downgrades_persisted_external_runtime_when_feature_is_disabled() {
        let dir = test_artifact_dir("load-downgrades-external-runtime");
        fs::create_dir_all(&dir).expect("create temp dir");

        let values = vec![1.0, 1.2, 1.1, 1.4, 1.6, 1.5, 1.7, 1.8];
        let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
        let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");
        let last_result = SwarmForecastResult {
            point_forecast: vec![1.9, 2.0, 2.1],
            level_80_lower: vec![1.7, 1.8, 1.9],
            level_80_upper: vec![2.1, 2.2, 2.3],
            diversity_score: 0.12,
            effective_models: 3.0,
            prediction_variance: 0.04,
            models_used: 3,
        };
        let candidate_reports = vec![SwarmCandidateReport {
            name: "persistence".to_string(),
            model_type: "MLP".to_string(),
            source: "external".to_string(),
            weight: 1.0,
            prediction_length: 3,
            prediction_mean: 2.0,
            prediction_std: 0.2,
            mae: 0.1,
            mse: 0.01,
            mape: 5.0,
            smape: 4.0,
            coverage: 0.8,
            bias: 0.0,
            directional_accuracy: 0.75,
            regime_fit: 0.65,
            stability_score: 0.90,
        }];
        let training_report = SwarmTrainingReport {
            training_rows: values.len(),
            validation_windows: 1,
            fitted_horizon: 3,
            best_candidate: Some("persistence".to_string()),
            aggregate_mae: 0.1,
            aggregate_smape: 4.0,
            aggregate_directional_accuracy: 0.75,
            aggregate_coverage: 0.8,
            diversity_score: 0.12,
            regime_bias: 0.2,
            updated_at_unix_ms: Some(456),
        };

        let artifact = SwarmForecasterArtifact {
            config: SwarmForecastConfig::default(),
            runtime_mode: SwarmRuntimeMode::ExternalSwarm,
            runtime_degraded_reason: None,
            fitted: true,
            values: values.clone(),
            timestamps: timestamps.clone(),
            unique_id: "test-series".to_string(),
            snapshot: Some(snapshot),
            last_result: Some(last_result),
            last_horizon: Some(3),
            candidate_reports,
            updated_at_unix_ms: Some(123),
            training_report: Some(training_report),
        };
        let payload = serde_json::to_vec_pretty(&artifact).expect("serialize artifact");
        fs::write(dir.join(SWARM_ARTIFACT_FILE_NAME), payload).expect("write artifact");

        let mut forecaster = SwarmForecaster::new(256.0);
        forecaster.load(&dir).expect("load external artifact");

        assert_eq!(forecaster.runtime_mode, SwarmRuntimeMode::LocalFallback);
        assert_eq!(
            forecaster.runtime_degraded_reason.as_deref(),
            Some("swarm_forecasting_feature_disabled")
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "swarm-forecasting")]
    #[test]
    fn select_external_or_fallback_result_marks_local_fallback_when_external_result_is_invalid() {
        let snapshot = SwarmForecastSnapshot {
            last_value: 1.0,
            rolling_mean: 1.0,
            drift_slope: 0.02,
            volatility: 0.05,
            has_trend: true,
            has_seasonality: false,
            seasonal_periods: vec![],
            short_mean: 1.01,
            medium_mean: 1.0,
            long_mean: 0.99,
            recent_return: 0.01,
            trend_strength: 0.55,
            mean_reversion_strength: 0.20,
            volatility_ratio: 1.05,
        };
        let candidates = vec![
            ("fast".to_string(), vec![1.01, 1.03]),
            ("slow".to_string(), vec![0.99, 1.01]),
        ];
        let reports = vec![
            SwarmCandidateReport {
                name: "fast".to_string(),
                model_type: "MLP".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.6,
                prediction_length: 2,
                prediction_mean: 1.02,
                prediction_std: 0.01,
                mae: 0.05,
                mse: 0.004,
                mape: 2.0,
                smape: 2.0,
                coverage: 0.8,
                bias: 0.01,
                directional_accuracy: 0.65,
                regime_fit: 0.60,
                stability_score: 0.62,
            },
            SwarmCandidateReport {
                name: "slow".to_string(),
                model_type: "Transformer".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.4,
                prediction_length: 2,
                prediction_mean: 1.0,
                prediction_std: 0.01,
                mae: 0.06,
                mse: 0.005,
                mape: 2.5,
                smape: 2.5,
                coverage: 0.82,
                bias: -0.01,
                directional_accuracy: 0.61,
                regime_fit: 0.58,
                stability_score: 0.59,
            },
        ];
        let weight_map = build_candidate_weight_map(&reports);
        let invalid_external = SwarmForecastResult {
            point_forecast: vec![f32::NAN, 1.02],
            level_80_lower: vec![0.9, 0.95],
            level_80_upper: vec![1.1, 1.08],
            diversity_score: 0.15,
            effective_models: 2.0,
            prediction_variance: 0.04,
            models_used: 2,
        };

        let (final_result, runtime_mode, degraded_reason) = select_external_or_fallback_result(
            &snapshot,
            &candidates,
            &weight_map,
            &reports,
            SwarmEnsembleStrategy::BayesianModelAveraging,
            2,
            invalid_external,
        );

        assert_eq!(runtime_mode, SwarmRuntimeMode::LocalFallback);
        assert_eq!(
            degraded_reason.as_deref(),
            Some("external_swarm_result_invalid")
        );
        assert!(result_is_valid(&final_result, 2));
    }

    #[test]
    fn save_strips_stale_forecast_state_from_unfitted_artifacts() {
        let dir = test_artifact_dir("save-strips-unfitted-state");
        fs::create_dir_all(&dir).expect("create temp dir");

        let values = vec![1.0, 1.2, 1.1, 1.4, 1.6, 1.5, 1.7, 1.8];
        let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
        let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");
        let last_result = SwarmForecastResult {
            point_forecast: vec![1.9, 2.0, 2.1],
            level_80_lower: vec![1.7, 1.8, 1.9],
            level_80_upper: vec![2.1, 2.2, 2.3],
            diversity_score: 0.12,
            effective_models: 3.0,
            prediction_variance: 0.04,
            models_used: 3,
        };

        let mut forecaster = SwarmForecaster::new(256.0);
        forecaster.runtime_mode = SwarmRuntimeMode::ExternalSwarm;
        forecaster.runtime_degraded_reason = Some("stale_external_state".to_string());
        forecaster.values = values.clone();
        forecaster.timestamps = timestamps.clone();
        forecaster.unique_id = "test-series".to_string();
        forecaster.snapshot = Some(snapshot);
        forecaster.last_result = Some(last_result);
        forecaster.last_horizon = Some(3);
        forecaster.candidate_reports = vec![SwarmCandidateReport {
            name: "stale".to_string(),
            model_type: "MLP".to_string(),
            source: "local".to_string(),
            weight: 1.0,
            prediction_length: 3,
            prediction_mean: 2.0,
            prediction_std: 0.2,
            mae: 0.1,
            mse: 0.01,
            mape: 5.0,
            smape: 4.0,
            coverage: 0.8,
            bias: 0.0,
            directional_accuracy: 0.75,
            regime_fit: 0.65,
            stability_score: 0.90,
        }];
        forecaster.updated_at_unix_ms = Some(123);
        forecaster.training_report = Some(SwarmTrainingReport {
            training_rows: values.len(),
            validation_windows: 1,
            fitted_horizon: 3,
            best_candidate: Some("stale".to_string()),
            aggregate_mae: 0.1,
            aggregate_smape: 4.0,
            aggregate_directional_accuracy: 0.75,
            aggregate_coverage: 0.8,
            diversity_score: 0.12,
            regime_bias: 0.2,
            updated_at_unix_ms: Some(456),
        });

        forecaster.save(&dir).expect("save should succeed");
        let payload =
            fs::read(dir.join(SWARM_ARTIFACT_FILE_NAME)).expect("read saved swarm artifact");
        let artifact: SwarmForecasterArtifact =
            serde_json::from_slice(&payload).expect("deserialize saved artifact");

        assert!(!artifact.fitted);
        assert_eq!(artifact.runtime_mode, SwarmRuntimeMode::LocalFallback);
        assert!(artifact.runtime_degraded_reason.is_none());
        assert!(artifact.snapshot.is_none());
        assert!(artifact.last_result.is_none());
        assert!(artifact.last_horizon.is_none());
        assert!(artifact.candidate_reports.is_empty());
        assert!(artifact.updated_at_unix_ms.is_none());
        assert!(artifact.training_report.is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn learned_validation_weights_favor_lower_loss_candidates() {
        let mut losses = HashMap::new();
        losses.insert("best".to_string(), 0.12);
        losses.insert("mid".to_string(), 0.48);
        losses.insert("worst".to_string(), 1.44);

        let mut support = HashMap::new();
        support.insert("best".to_string(), 3.0);
        support.insert("mid".to_string(), 3.0);
        support.insert("worst".to_string(), 3.0);

        let weights = derive_validation_candidate_weights(&losses, &support);

        let best = weights.get("best").copied().expect("best weight");
        let mid = weights.get("mid").copied().expect("mid weight");
        let worst = weights.get("worst").copied().expect("worst weight");

        assert!(
            best > mid,
            "best candidate should receive more weight than mid"
        );
        assert!(
            mid > worst,
            "mid candidate should receive more weight than worst"
        );

        let total = weights.values().copied().sum::<f32>();
        assert!((total - 1.0).abs() < 1e-5, "weights should normalize to 1");
    }

    #[test]
    fn normalize_candidate_weights_preserves_learned_validation_ordering() {
        let mut reports = vec![
            SwarmCandidateReport {
                name: "best".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.62,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.55,
                mse: 0.45,
                mape: 4.0,
                smape: 4.0,
                coverage: 0.72,
                bias: 0.02,
                directional_accuracy: 0.66,
                regime_fit: 0.62,
                stability_score: 0.61,
            },
            SwarmCandidateReport {
                name: "mid".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.28,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.20,
                mse: 0.12,
                mape: 2.0,
                smape: 2.0,
                coverage: 0.84,
                bias: 0.01,
                directional_accuracy: 0.77,
                regime_fit: 0.76,
                stability_score: 0.79,
            },
            SwarmCandidateReport {
                name: "worst".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.10,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.08,
                mse: 0.05,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.92,
                bias: 0.0,
                directional_accuracy: 0.88,
                regime_fit: 0.89,
                stability_score: 0.91,
            },
        ];

        normalize_candidate_weights(&mut reports);

        assert!(
            reports[0].weight > reports[1].weight,
            "existing learned weights should keep best ahead of mid"
        );
        assert!(
            reports[1].weight > reports[2].weight,
            "existing learned weights should keep mid ahead of worst"
        );
        let total = reports.iter().map(|report| report.weight).sum::<f32>();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "normalized learned weights should still sum to 1"
        );
    }

    #[test]
    fn validation_weight_blend_ratio_scales_with_support() {
        let low_support = HashMap::from([("a".to_string(), 1.0_f32), ("b".to_string(), 1.0_f32)]);
        let high_support =
            HashMap::from([("a".to_string(), 24.0_f32), ("b".to_string(), 24.0_f32)]);

        let low_ratio = validation_weight_blend_ratio(&low_support, 2);
        let high_ratio = validation_weight_blend_ratio(&high_support, 2);

        assert!(
            low_ratio < 0.8,
            "low support should preserve heuristic influence"
        );
        assert!(
            high_ratio > 0.95,
            "strong validation support should rely almost entirely on learned weights"
        );
        assert!(high_ratio > low_ratio);
    }

    #[test]
    fn training_report_matches_requires_exact_validation_window_count() {
        let reports = vec![SwarmCandidateReport {
            name: "best".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 1.0,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.8,
            regime_fit: 0.8,
            stability_score: 0.8,
        }];
        let report = SwarmTrainingReport {
            training_rows: 128,
            validation_windows: 2,
            fitted_horizon: 8,
            best_candidate: Some("best".to_string()),
            aggregate_mae: 0.1,
            aggregate_smape: 1.0,
            aggregate_directional_accuracy: 0.8,
            aggregate_coverage: 0.9,
            diversity_score: 0.2,
            regime_bias: 0.1,
            updated_at_unix_ms: Some(1),
        };

        assert!(
            !training_report_matches(&report, &reports, 128, 3, 8),
            "stale validation-window counts should invalidate persisted swarm reports"
        );
    }

    #[test]
    fn rolling_validation_reports_use_forecast_horizon_not_validation_window_length() {
        let train_values = (0..72)
            .map(|idx| 1.0 + idx as f32 * 0.01 + ((idx % 5) as f32 - 2.0) * 0.002)
            .collect::<Vec<_>>();
        let train_timestamps = (0..train_values.len())
            .map(|idx| idx as f64)
            .collect::<Vec<_>>();
        let actuals = vec![1.72, 1.73, 1.74, 1.75];
        let windows = vec![(train_values, train_timestamps, actuals)];

        let reports = build_weighted_reports_local(&windows, 12).expect("rolling reports");

        assert!(!reports.is_empty());
        assert!(reports.iter().all(|report| report.prediction_length == 12));
        assert!(candidate_reports_are_valid(&reports, 12));
        assert!(!candidate_reports_are_valid(&reports, 4));
    }

    #[test]
    fn candidate_reports_are_valid_rejects_duplicate_names() {
        let reports = vec![
            SwarmCandidateReport {
                name: "dup".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.6,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
            SwarmCandidateReport {
                name: "dup".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.4,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
        ];

        assert!(!candidate_reports_are_valid(&reports, 8));
    }

    #[test]
    fn candidate_reports_are_valid_rejects_non_positive_weights() {
        let reports = vec![SwarmCandidateReport {
            name: "a".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 0.0,
            prediction_length: 8,
            prediction_mean: 1.0,
            prediction_std: 0.1,
            mae: 0.1,
            mse: 0.1,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.9,
            bias: 0.0,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        }];

        assert!(!candidate_reports_are_valid(&reports, 8));
    }

    #[test]
    fn prune_validation_candidates_keeps_top_supported_reports() {
        let mut reports = vec![
            SwarmCandidateReport {
                name: "a".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.42,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
            SwarmCandidateReport {
                name: "b".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.28,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
            SwarmCandidateReport {
                name: "c".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.14,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
            SwarmCandidateReport {
                name: "d".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.09,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
            SwarmCandidateReport {
                name: "e".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.04,
                prediction_length: 8,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
        ];
        let support = HashMap::from([
            ("a".to_string(), 8.0_f32),
            ("b".to_string(), 8.0_f32),
            ("c".to_string(), 8.0_f32),
            ("d".to_string(), 8.0_f32),
            ("e".to_string(), 8.0_f32),
        ]);

        prune_validation_candidates(&mut reports, &support);

        assert!(reports.len() < 5);
        assert!(reports.iter().any(|report| report.name == "a"));
        assert!(reports.iter().any(|report| report.name == "b"));
        let total = reports.iter().map(|report| report.weight).sum::<f32>();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn select_active_candidates_uses_positive_weight_reports() {
        let candidates = vec![
            ("a".to_string(), vec![1.0, 1.1]),
            ("b".to_string(), vec![1.0, 1.2]),
            ("c".to_string(), vec![1.0, 0.9]),
        ];
        let reports = vec![
            SwarmCandidateReport {
                name: "a".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.7,
                prediction_length: 2,
                prediction_mean: 1.05,
                prediction_std: 0.05,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
            SwarmCandidateReport {
                name: "b".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.3,
                prediction_length: 2,
                prediction_mean: 1.1,
                prediction_std: 0.05,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
            SwarmCandidateReport {
                name: "c".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.0,
                prediction_length: 2,
                prediction_mean: 0.95,
                prediction_std: 0.05,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
        ];

        let active = select_active_candidates(&candidates, &reports);
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|(name, _)| name == "a" || name == "b"));
    }

    #[test]
    fn calibrated_interval_spread_widens_when_validation_coverage_is_weak() {
        let snapshot = SwarmForecastSnapshot {
            last_value: 1.0,
            rolling_mean: 1.0,
            drift_slope: 0.01,
            volatility: 0.03,
            has_trend: true,
            has_seasonality: false,
            seasonal_periods: vec![],
            short_mean: 1.02,
            medium_mean: 1.01,
            long_mean: 0.99,
            recent_return: 0.01,
            trend_strength: 0.7,
            mean_reversion_strength: 0.2,
            volatility_ratio: 1.0,
        };
        let step_values = vec![1.00, 1.02, 1.03, 1.01];
        let strong = [SwarmCandidateReport {
            name: "a".to_string(),
            model_type: "local".to_string(),
            source: "rolling_validation".to_string(),
            weight: 1.0,
            prediction_length: 4,
            prediction_mean: 1.015,
            prediction_std: 0.01,
            mae: 0.01,
            mse: 0.0002,
            mape: 1.0,
            smape: 1.0,
            coverage: 0.88,
            bias: 0.002,
            directional_accuracy: 0.9,
            regime_fit: 0.9,
            stability_score: 0.9,
        }];
        let weak = [SwarmCandidateReport {
            coverage: 0.45,
            mae: 0.03,
            prediction_std: 0.02,
            bias: 0.01,
            ..strong[0].clone()
        }];
        let strong_refs = strong.iter().collect::<Vec<_>>();
        let weak_refs = weak.iter().collect::<Vec<_>>();

        let strong_spread =
            calibrated_interval_spread(1.015, &step_values, &strong_refs, &snapshot);
        let weak_spread = calibrated_interval_spread(1.015, &step_values, &weak_refs, &snapshot);

        assert!(weak_spread > strong_spread);
    }

    #[test]
    fn active_report_refs_follow_active_candidates() {
        let reports = vec![
            SwarmCandidateReport {
                name: "a".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.6,
                prediction_length: 2,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
            SwarmCandidateReport {
                name: "b".to_string(),
                model_type: "local".to_string(),
                source: "rolling_validation".to_string(),
                weight: 0.4,
                prediction_length: 2,
                prediction_mean: 1.0,
                prediction_std: 0.1,
                mae: 0.1,
                mse: 0.1,
                mape: 1.0,
                smape: 1.0,
                coverage: 0.9,
                bias: 0.0,
                directional_accuracy: 0.9,
                regime_fit: 0.9,
                stability_score: 0.9,
            },
        ];
        let candidates = vec![("b".to_string(), vec![1.0, 1.1])];

        let active = active_report_refs(&reports, &candidates);

        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "b");
    }

    #[test]
    fn prune_local_candidates_deduplicates_near_identical_forecasts() {
        let snapshot = SwarmForecastSnapshot {
            last_value: 1.0,
            rolling_mean: 1.0,
            drift_slope: 0.01,
            volatility: 0.05,
            has_trend: true,
            has_seasonality: false,
            seasonal_periods: vec![],
            short_mean: 1.02,
            medium_mean: 1.01,
            long_mean: 0.99,
            recent_return: 0.01,
            trend_strength: 0.7,
            mean_reversion_strength: 0.2,
            volatility_ratio: 1.0,
        };
        let candidates = vec![
            ("drift".to_string(), vec![1.01, 1.02, 1.03, 1.04]),
            (
                "momentum_blend".to_string(),
                vec![1.0105, 1.0205, 1.0305, 1.0405],
            ),
            ("mean_reversion".to_string(), vec![1.0, 0.995, 0.99, 0.985]),
            ("persistence".to_string(), vec![1.0, 1.0, 1.0, 1.0]),
        ];

        let pruned = prune_local_candidates(&snapshot, candidates);

        assert!(pruned.len() < 4);
        assert!(pruned.iter().any(|(name, _)| name == "drift"));
        assert!(!pruned.iter().any(|(name, _)| name == "momentum_blend"));
    }

    #[test]
    fn candidate_preselection_score_prefers_regime_aligned_family() {
        let trend_snapshot = SwarmForecastSnapshot {
            last_value: 1.0,
            rolling_mean: 0.99,
            drift_slope: 0.02,
            volatility: 0.04,
            has_trend: true,
            has_seasonality: false,
            seasonal_periods: vec![],
            short_mean: 1.03,
            medium_mean: 1.01,
            long_mean: 0.98,
            recent_return: 0.01,
            trend_strength: 0.8,
            mean_reversion_strength: 0.15,
            volatility_ratio: 1.0,
        };
        let drift = vec![1.01, 1.03, 1.05, 1.07];
        let mean_reversion = vec![0.995, 0.992, 0.99, 0.988];

        let drift_score = candidate_preselection_score("drift", &drift, &trend_snapshot);
        let mean_reversion_score =
            candidate_preselection_score("mean_reversion", &mean_reversion, &trend_snapshot);

        assert!(drift_score > mean_reversion_score);
    }

    #[test]
    fn candidate_forecasts_local_reduce_mean_reversion_noise_in_trend_regime() {
        let snapshot = SwarmForecastSnapshot {
            last_value: 1.0,
            rolling_mean: 0.99,
            drift_slope: 0.025,
            volatility: 0.04,
            has_trend: true,
            has_seasonality: false,
            seasonal_periods: vec![],
            short_mean: 1.04,
            medium_mean: 1.02,
            long_mean: 0.98,
            recent_return: 0.012,
            trend_strength: 0.82,
            mean_reversion_strength: 0.18,
            volatility_ratio: 1.0,
        };
        let values = (0..64)
            .map(|idx| 1.0 + idx as f32 * 0.01)
            .collect::<Vec<_>>();

        let candidates = candidate_forecasts_local(&values, &snapshot, 8);

        assert!(candidates.iter().any(|(name, _)| name == "drift"));
        assert!(candidates.iter().any(|(name, _)| name == "damped_drift"));
        assert!(candidates.iter().any(|(name, _)| name == "momentum_blend"));
        assert!(!candidates.iter().any(|(name, _)| name == "mean_reversion"));
        assert!(!candidates.iter().any(|(name, _)| name == "regime_anchor"));
    }

    #[cfg(feature = "swarm-forecasting")]
    #[test]
    fn candidate_predictions_reduce_trend_noise_in_mean_reversion_regime() {
        let snapshot = SwarmForecastSnapshot {
            last_value: 1.0,
            rolling_mean: 1.01,
            drift_slope: -0.002,
            volatility: 0.05,
            has_trend: false,
            has_seasonality: false,
            seasonal_periods: vec![],
            short_mean: 1.0,
            medium_mean: 1.01,
            long_mean: 1.02,
            recent_return: -0.001,
            trend_strength: 0.25,
            mean_reversion_strength: 0.76,
            volatility_ratio: 1.1,
        };
        let values = (0..64)
            .map(|idx| 1.05 - idx as f32 * 0.0015)
            .collect::<Vec<_>>();

        let candidates = candidate_predictions(&values, &snapshot, 8);

        assert!(
            candidates
                .iter()
                .any(|(name, _, _)| name == "mean_reversion")
        );
        assert!(
            candidates
                .iter()
                .any(|(name, _, _)| name == "regime_anchor")
        );
        assert!(
            !candidates
                .iter()
                .any(|(name, _, _)| name == "momentum_blend")
        );
    }

    #[cfg(feature = "swarm-forecasting")]
    #[test]
    fn sanitize_downgrades_stale_external_runtime_to_local_fallback_reason() {
        let values = (0..64)
            .map(|idx| 1.0 + idx as f32 * 0.01 + ((idx % 5) as f32 - 2.0) * 0.002)
            .collect::<Vec<_>>();
        let timestamps = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
        let snapshot = build_local_snapshot_with_min(&values, &timestamps, 8).expect("snapshot");

        let artifact = SwarmForecasterArtifact {
            config: SwarmForecastConfig::default(),
            runtime_mode: SwarmRuntimeMode::ExternalSwarm,
            runtime_degraded_reason: None,
            fitted: true,
            values,
            timestamps,
            unique_id: "repair-external".to_string(),
            snapshot: Some(snapshot),
            last_result: None,
            last_horizon: Some(6),
            candidate_reports: Vec::new(),
            updated_at_unix_ms: Some(1),
            training_report: None,
        };

        let repaired = sanitize_forecaster_artifact(artifact).expect("sanitize");
        assert_eq!(repaired.runtime_mode, SwarmRuntimeMode::LocalFallback);
        assert_eq!(
            repaired.runtime_degraded_reason.as_deref(),
            Some("external_swarm_result_rebuilt_from_local_consensus")
        );
        assert!(repaired.last_result.is_some());
        assert!(!repaired.candidate_reports.is_empty());
    }
}
