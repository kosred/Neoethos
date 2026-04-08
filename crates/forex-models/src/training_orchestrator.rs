use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::base::ExpertModel;
use crate::burn_models::active_burn_backend_name;
use crate::ensemble::{
    CalibrationMethod, ConformalPredictionExpert, MetaBlender, MetaDecisionStack,
    ProbabilityCalibrationExpert,
};
use crate::exit_agent::ExitAgent;
use crate::parallel_trainer::{
    train_models_parallel_with_progress, ModelConfig, ModelTrainingFailure, ModelTrainingProgress,
    ModelType, TrainingPayload,
};
use crate::runtime::capabilities::CapabilityState;
use crate::runtime::dispatch::{build_dispatch_plan, DispatchPlan};
use crate::runtime::exports::{
    write_onnx_export_status, OnnxExportStatus, ONNX_EXPORT_STATUS_FILE_NAME,
};
use crate::runtime::hpo::{
    evaluate_prediction_quality, time_series_holdout_split, write_optimization_report,
    OptimizationReport, OptimizationTrialRecord, OPTIMIZATION_REPORT_FILE_NAME,
};
use crate::runtime::profile::{
    write_training_runtime_profile, TrainingRuntimeProfile, TRAINING_RUNTIME_PROFILE_FILE_NAME,
};
use crate::tree_models::config::ParamValue;
use crate::tree_models::{CatBoostExpert, LightGBMExpert, SklearsTreeExpert, XGBoostExpert};
use crate::{
    BayesianLogitExpert, ElasticNetExpert, GeneticStrategyExpert, IsolationForestExpert, KANExpert,
    MLPExpert, NBeatsExpert, NBeatsxNfExpert, NeatExpert, NeuroEvoExpert, OnlineHoeffdingExpert,
    OnlinePassiveAggressiveExpert, PatchTSTExpert, SwarmForecaster, TabNetExpert, TiDEExpert,
    TiDENfExpert, TimesNetExpert, TradingReinforcementLearner, TransformerExpert,
};
use forex_data::{
    load_symbol_dataset, prepare_multitimeframe_features_with_options, FeatureBuildOptions, Ohlcv,
};
use forex_search::genetic::{ParentSelectionPolicy, SurvivorSelectionPolicy};
use polars::prelude::{BooleanChunked, DataFrame, NamedFrom, NewChunkedArray, Series};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingRunSummary {
    pub planned_models: Vec<String>,
    pub completed_models: Vec<String>,
    pub failed_models: Vec<ModelTrainingFailure>,
}

pub struct TrainingOrchestrator {
    pub settings: forex_core::Settings,
    pub models_dir: PathBuf,
}

impl TrainingOrchestrator {
    pub fn new(settings: forex_core::Settings, models_dir: PathBuf) -> Self {
        Self {
            settings,
            models_dir,
        }
    }

    fn preferred_burn_device_policy(&self) -> String {
        let gpu_pref = self
            .settings
            .system
            .enable_gpu_preference
            .trim()
            .to_ascii_lowercase();
        let system_device = self.settings.system.device.trim().to_ascii_lowercase();
        match gpu_pref.as_str() {
            "false" | "cpu" => "cpu".to_string(),
            "true" | "gpu" => {
                if system_device.is_empty() || system_device == "cpu" {
                    "gpu".to_string()
                } else {
                    system_device
                }
            }
            _ => {
                if system_device.starts_with("cuda:")
                    || system_device.starts_with("gpu:")
                    || system_device.starts_with("wgpu:")
                    || matches!(system_device.as_str(), "gpu" | "cuda" | "wgpu")
                {
                    system_device
                } else {
                    "auto".to_string()
                }
            }
        }
    }

    pub fn train_symbol(&self, symbol: &str, base_tf: &str) -> Result<()> {
        let summary = self.train_symbol_with_progress(symbol, base_tf, |_| {})?;
        if !summary.failed_models.is_empty() {
            anyhow::bail!(
                "Training failed for [{}]; successful models: [{}]",
                summary
                    .failed_models
                    .iter()
                    .map(|failure| failure.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                summary.completed_models.join(", ")
            );
        }

        Ok(())
    }

    pub fn train_symbol_with_progress<R>(
        &self,
        symbol: &str,
        base_tf: &str,
        progress_fn: R,
    ) -> Result<TrainingRunSummary>
    where
        R: Fn(ModelTrainingProgress) + Send + Sync + Clone + 'static,
    {
        info!("Starting Pure-Rust training for symbol: {}", symbol);

        let dispatch_plan = self.create_dispatch_plan()?;
        self.validate_dispatch_plan(&dispatch_plan)?;
        let configs = self.build_training_configs(&dispatch_plan)?;
        let planned_models: Vec<String> =
            configs.iter().map(|config| config.name.clone()).collect();

        let data_root = std::env::var("FOREX_BOT_DATA_ROOT")
            .unwrap_or_else(|_| self.settings.system.data_dir.to_string_lossy().to_string());
        let dataset = load_symbol_dataset(&data_root, symbol)?;

        let opts = FeatureBuildOptions {
            higher_tfs: self.selected_feature_timeframes(base_tf),
            prefix_base_features: self.settings.system.multi_resolution_prefix_base,
            ..FeatureBuildOptions::default()
        };
        let frame = prepare_multitimeframe_features_with_options(&dataset, base_tf, &opts, None)?;
        let base_ohlcv = dataset.frames.get(base_tf).context("base tf missing")?;
        let labels = self.derive_labels(base_ohlcv);
        let raw_payload = TrainingPayload::from_named_dense(frame.data, labels, frame.names)?;
        let filtered_payload = if self.settings.models.filter_to_base_signal {
            let (filtered_frame, filtered_labels) = self.apply_base_signal_filter(
                raw_payload.frame.as_ref(),
                raw_payload.labels.as_ref(),
            )?;
            if filtered_frame.height() == raw_payload.frame.height() {
                raw_payload
            } else {
                TrainingPayload::from_frame(filtered_frame, filtered_labels)?
            }
        } else {
            raw_payload
        };
        let (budgeted_frame, budgeted_labels, row_budget_applied) = self
            .apply_training_row_budget(
                filtered_payload.frame.as_ref(),
                filtered_payload.labels.as_ref(),
            )?;
        let selected_frame =
            self.apply_feature_selection(&budgeted_frame, &budgeted_labels, base_ohlcv)?;
        let payload = if selected_frame.width() == budgeted_frame.width() {
            Arc::new(TrainingPayload::from_frame(
                budgeted_frame,
                budgeted_labels.clone(),
            )?)
        } else {
            Arc::new(TrainingPayload::from_frame(
                selected_frame,
                budgeted_labels.clone(),
            )?)
        };
        let models_dir = self.models_dir.clone();
        let settings = self.settings.clone();
        let symbol = symbol.to_string();
        let base_tf = base_tf.to_string();
        let trained = train_models_parallel_with_progress(
            configs,
            payload,
            progress_fn,
            move |config, payload| {
                info!(
                    "Training model instance: {} ({:?})",
                    config.name, config.model_type
                );
                train_model_dispatch(
                    &models_dir,
                    &settings,
                    &symbol,
                    &base_tf,
                    row_budget_applied,
                    config,
                    payload,
                )
            },
        )?;

        info!(
            "Successfully trained models: {:?}",
            trained.successful_models
        );
        Ok(TrainingRunSummary {
            planned_models,
            completed_models: trained.successful_models,
            failed_models: trained.failed_models,
        })
    }

    fn create_dispatch_plan(&self) -> Result<DispatchPlan> {
        let mut requested_models = self.settings.models.ml_models.clone();
        requested_models.extend(self.settings.models.phase5_core_models.clone());

        if self.settings.models.regime_router_enabled {
            requested_models.extend(self.settings.models.regime_trend_models.clone());
            requested_models.extend(self.settings.models.regime_range_models.clone());
            requested_models.extend(self.settings.models.regime_neutral_models.clone());
        }

        if self.settings.models.phase5_filter_meta_blender {
            requested_models.push("meta_blender".to_string());
        }
        if self.settings.models.calibration_enabled {
            requested_models.push("probability_calibrator".to_string());
            if self.settings.risk.conformal_enabled {
                requested_models.push("conformal_gate".to_string());
            }
            if self.settings.models.phase5_filter_meta_blender {
                requested_models.push("meta_stack".to_string());
            }
        }
        if self.settings.models.use_sac_agent {
            requested_models.push("exit_agent".to_string());
        }
        if self.settings.models.use_rl_agent || self.settings.models.use_rllib_agent {
            requested_models.push("dqn".to_string());
        }
        if self.settings.models.use_neuroevolution {
            requested_models.push("neuro_evo".to_string());
            requested_models.push("neat".to_string());
        }
        if self.settings.models.prop_search_enabled {
            requested_models.push("genetic".to_string());
        }
        requested_models = self.expand_requested_models(requested_models);
        if self.settings.models.regime_router_enabled
            && self.settings.models.regime_router_min_models > 0
        {
            let routed_models = requested_models
                .iter()
                .filter(|name| {
                    configured_contains_model(&self.settings.models.regime_trend_models, name)
                        || configured_contains_model(
                            &self.settings.models.regime_range_models,
                            name,
                        )
                        || configured_contains_model(
                            &self.settings.models.regime_neutral_models,
                            name,
                        )
                })
                .count();
            if routed_models < self.settings.models.regime_router_min_models {
                anyhow::bail!(
                    "regime router requires at least {} routed models but only {} are configured",
                    self.settings.models.regime_router_min_models,
                    routed_models
                );
            }
        }
        build_dispatch_plan(&requested_models)
    }

    fn expand_requested_models(&self, requested_models: Vec<String>) -> Vec<String> {
        let mut expanded = Vec::new();

        for name in requested_models {
            let trimmed = name.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.eq_ignore_ascii_case("transformer") {
                if !self.settings.models.enable_transformer_expert {
                    continue;
                }

                let replica_count = self.settings.models.num_transformers.max(1);
                if replica_count == 1 {
                    expanded.push("transformer".to_string());
                } else {
                    for replica_idx in 1..=replica_count {
                        expanded.push(format!("transformer_{replica_idx:02}"));
                    }
                }
                continue;
            }

            expanded.push(trimmed.to_string());
        }

        expanded
    }

    fn validate_dispatch_plan(&self, dispatch_plan: &DispatchPlan) -> Result<()> {
        let blocked: Vec<String> = dispatch_plan
            .entries
            .iter()
            .filter(|entry| entry.state == CapabilityState::Planned)
            .map(|entry| format!("{} ({}, {})", entry.name, entry.family, entry.state))
            .collect();

        if !blocked.is_empty() {
            anyhow::bail!(
                "configured models resolve to planned capabilities and cannot enter the training dispatch: {}",
                blocked.join(", ")
            );
        }

        Ok(())
    }

    fn build_training_configs(&self, dispatch_plan: &DispatchPlan) -> Result<Vec<ModelConfig>> {
        dispatch_plan
            .entries
            .iter()
            .map(|entry| {
                let mut params = self.default_model_params(&entry.name);
                self.inject_runtime_model_params(&entry.name, &mut params);
                self.apply_model_param_overrides(&entry.name, &mut params);
                Ok(ModelConfig {
                    name: entry.name.clone(),
                    model_type: self.map_model_type(&entry.name)?,
                    capability_family: entry.family,
                    capability_state: entry.state,
                    params,
                })
            })
            .collect()
    }

    fn epochs_from_seconds(seconds: u64, default_epochs: usize) -> usize {
        let derived = (seconds / 30).clamp(16, 400) as usize;
        derived.max(default_epochs)
    }

    fn max_epochs_for_model(&self, name: &str, default_epochs: usize) -> usize {
        if let Some(explicit) = self.settings.models.max_epochs_by_model.get(name) {
            return (*explicit).max(1);
        }

        let seconds = match name {
            "transformer" => self.settings.models.transformer_train_seconds,
            "nbeats" | "nbeatsx_nf" => self.settings.models.nbeats_train_seconds,
            "tide" | "tide_nf" => self.settings.models.tide_train_seconds,
            "tabnet" => self.settings.models.tabnet_train_seconds,
            "kan" => self.settings.models.kan_train_seconds,
            "mlp" => self.settings.models.mlp_train_seconds,
            _ => 0,
        };

        if seconds == 0 {
            default_epochs.max(1)
        } else {
            Self::epochs_from_seconds(seconds, default_epochs)
        }
    }

    fn min_calibration_rows(&self) -> usize {
        self.settings.models.calibration_min_rows.max(32)
    }

    fn effective_label_horizon_bars(&self) -> usize {
        if self.settings.models.label_horizon_bars > 0 {
            self.settings.models.label_horizon_bars
        } else {
            self.settings.risk.meta_label_max_hold_bars.max(1)
        }
    }

    fn holdout_pct(&self) -> f64 {
        let configured = self.settings.models.train_holdout_pct;
        if configured.is_finite() && (0.0..0.5).contains(&configured) {
            configured.max(0.05)
        } else {
            (1.0 - self.settings.models.global_train_ratio).clamp(0.05, 0.40)
        }
    }

    fn hpo_trials_for_model(&self, name: &str) -> usize {
        let canonical = canonical_model_name(name);
        self.settings
            .models
            .hpo_trials_by_model
            .get(canonical)
            .copied()
            .unwrap_or(self.settings.models.hpo_trials)
            .max(1)
    }

    fn inject_runtime_model_params(&self, name: &str, params: &mut HashMap<String, String>) {
        params.insert(
            "__hpo_backend".to_string(),
            self.settings.models.hpo_backend.clone(),
        );
        params.insert(
            "__hpo_trials".to_string(),
            self.hpo_trials_for_model(name).to_string(),
        );
        params.insert(
            "__hpo_max_rows".to_string(),
            self.settings.models.hpo_max_rows.to_string(),
        );
        params.insert(
            "__holdout_pct".to_string(),
            format!("{:.6}", self.holdout_pct()),
        );
        params.insert(
            "__conf_threshold".to_string(),
            format!("{:.6}", self.settings.models.prop_conf_threshold),
        );
        params.insert(
            "__metric_weight".to_string(),
            format!("{:.6}", self.settings.models.prop_metric_weight),
        );
        params.insert(
            "__accuracy_weight".to_string(),
            format!("{:.6}", self.settings.models.prop_accuracy_weight),
        );
        params.insert(
            "__embargo_minutes".to_string(),
            self.settings.models.embargo_minutes.to_string(),
        );
        params.insert(
            "__export_onnx".to_string(),
            self.settings.models.export_onnx.to_string(),
        );
    }

    fn apply_model_param_overrides(&self, name: &str, params: &mut HashMap<String, String>) {
        if let Some(exact) = self.settings.models.model_param_overrides.get(name) {
            params.extend(
                exact
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
        }

        let canonical = canonical_model_name(name);
        if canonical != name {
            if let Some(canonical_overrides) =
                self.settings.models.model_param_overrides.get(canonical)
            {
                for (key, value) in canonical_overrides {
                    params.entry(key.clone()).or_insert_with(|| value.clone());
                }
            }
        }
    }

    fn transformer_hidden_dim(&self) -> usize {
        self.settings
            .models
            .transformer_d_model
            .max(self.settings.models.transformer_hidden_dim)
            .max(16)
    }

    fn transformer_heads(&self) -> usize {
        self.settings
            .models
            .transformer_n_heads
            .max(self.settings.models.transformer_heads)
            .max(1)
    }

    fn transformer_layers(&self) -> usize {
        self.settings
            .models
            .transformer_n_layers
            .max(self.settings.models.transformer_layers)
            .max(1)
    }

    fn effective_training_row_budget(&self) -> Option<usize> {
        [
            self.settings.models.global_max_rows,
            self.settings.models.global_max_rows_per_symbol,
            self.settings.system.max_training_rows_per_tf,
        ]
        .into_iter()
        .filter(|value| *value > 0)
        .min()
    }

    fn apply_training_row_budget(
        &self,
        frame: &DataFrame,
        labels: &[i32],
    ) -> Result<(DataFrame, Vec<i32>, Option<usize>)> {
        if frame.height() != labels.len() {
            anyhow::bail!(
                "training row-budget mismatch: {} rows vs {} labels",
                frame.height(),
                labels.len()
            );
        }

        let Some(cap) = self.effective_training_row_budget() else {
            return Ok((frame.clone(), labels.to_vec(), None));
        };

        if frame.height() <= cap {
            return Ok((frame.clone(), labels.to_vec(), None));
        }

        let start = frame.height().saturating_sub(cap);
        let budgeted_frame = frame.slice(start as i64, cap);
        let budgeted_labels = labels
            .iter()
            .skip(start)
            .take(cap)
            .copied()
            .collect::<Vec<_>>();

        info!(
            "Applied training row budget: kept {} / {} rows",
            budgeted_frame.height(),
            frame.height()
        );
        Ok((budgeted_frame, budgeted_labels, Some(cap)))
    }

    fn selected_feature_timeframes(&self, base_tf: &str) -> Vec<String> {
        let source = if self.settings.system.multi_resolution_enabled
            && !self.settings.system.multi_resolution_timeframes.is_empty()
        {
            &self.settings.system.multi_resolution_timeframes
        } else {
            &self.settings.system.higher_timeframes
        };

        let mut selected = Vec::new();
        for timeframe in source {
            if timeframe.eq_ignore_ascii_case(base_tf) {
                continue;
            }
            if selected
                .iter()
                .any(|existing: &String| existing.eq_ignore_ascii_case(timeframe))
            {
                continue;
            }
            selected.push(timeframe.clone());
        }

        selected
    }

    fn conformal_alpha(&self) -> f32 {
        self.settings.risk.conformal_alpha.clamp(1e-4, 0.99) as f32
    }

    fn conformal_min_prediction_set(&self) -> usize {
        self.settings.risk.conformal_abstain_min_set_size.max(1)
    }

    fn fit_l1_ranked_features(
        &self,
        frame: &DataFrame,
        labels: &[i32],
    ) -> Result<Vec<(String, f32)>> {
        let alpha = 1.0 / self.settings.models.l1_feature_selection_c.max(1e-3);
        let mut selector = ElasticNetExpert::new(alpha, 1.0);
        selector.learning_rate = 0.03;
        selector.epochs = 400;
        selector.fit(frame, &labels_to_series(labels))?;
        selector.ranked_feature_importance()
    }

    fn recent_sample(
        &self,
        frame: &DataFrame,
        labels: &[i32],
    ) -> Result<(DataFrame, Vec<i32>, usize)> {
        let sample_limit = self.settings.models.l1_feature_selection_sample_limit.max(
            self.settings
                .models
                .l1_feature_selection_max_features
                .max(1),
        );
        let total_rows = frame.height();
        let sample_rows = total_rows.min(sample_limit.max(1));
        let start = total_rows.saturating_sub(sample_rows);
        let sampled = frame.slice(start as i64, sample_rows);
        let sampled_labels = labels
            .iter()
            .skip(start)
            .take(sample_rows)
            .copied()
            .collect::<Vec<_>>();
        Ok((sampled, sampled_labels, start))
    }

    fn take_row_subset(
        &self,
        frame: &DataFrame,
        labels: &[i32],
        indices: &[usize],
    ) -> Result<(DataFrame, Vec<i32>)> {
        if indices.is_empty() {
            anyhow::bail!("row subset indices must not be empty");
        }

        let mut mask = vec![false; frame.height()];
        let mut subset_labels = Vec::with_capacity(indices.len());
        for idx in indices.iter().copied().filter(|idx| *idx < frame.height()) {
            mask[idx] = true;
            subset_labels.push(labels[idx]);
        }

        if subset_labels.is_empty() {
            anyhow::bail!("row subset produced no labels");
        }

        let mask = BooleanChunked::from_slice("l1_selection_mask".into(), &mask);
        let subset = frame.filter(&mask)?;
        Ok((subset, subset_labels))
    }

    fn derive_regime_buckets(
        &self,
        base_ohlcv: &Ohlcv,
        sample_start: usize,
        sample_len: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut trend = Vec::new();
        let mut range = Vec::new();
        let mut neutral = Vec::new();

        for local_idx in 0..sample_len {
            let global_idx = sample_start + local_idx;
            if global_idx >= base_ohlcv.close.len()
                || global_idx >= base_ohlcv.open.len()
                || global_idx >= base_ohlcv.high.len()
                || global_idx >= base_ohlcv.low.len()
            {
                break;
            }

            let open = base_ohlcv.open[global_idx];
            let close = base_ohlcv.close[global_idx];
            let high = base_ohlcv.high[global_idx];
            let low = base_ohlcv.low[global_idx];
            let range_span = (high - low).abs().max(1e-9);
            let body_ratio = ((close - open).abs() / range_span) as f32;

            let lookback = global_idx.saturating_sub(16);
            let anchor = base_ohlcv.close[lookback];
            let drift_ratio = ((close - anchor).abs() / close.abs().max(1e-9)) as f32;

            if body_ratio >= 0.60 || drift_ratio >= 0.0035 {
                trend.push(local_idx);
            } else if body_ratio <= 0.25 && drift_ratio <= 0.0012 {
                range.push(local_idx);
            } else {
                neutral.push(local_idx);
            }
        }

        (trend, range, neutral)
    }

    fn apply_feature_selection(
        &self,
        frame: &DataFrame,
        labels: &[i32],
        base_ohlcv: &Ohlcv,
    ) -> Result<DataFrame> {
        if !self.settings.models.l1_feature_selection_enabled {
            return Ok(frame.clone());
        }

        let min_features = self
            .settings
            .models
            .l1_feature_selection_min_features
            .max(1);
        let max_features = self
            .settings
            .models
            .l1_feature_selection_max_features
            .max(min_features);
        if frame.width() <= min_features {
            return Ok(frame.clone());
        }

        let (sampled_frame, sampled_labels, sample_start) = self.recent_sample(frame, labels)?;
        let mut score_map = HashMap::<String, f32>::new();

        for (name, score) in self.fit_l1_ranked_features(&sampled_frame, &sampled_labels)? {
            *score_map.entry(name).or_insert(0.0) += score.max(0.0);
        }

        if self.settings.models.l1_feature_selection_per_regime {
            let (trend_rows, range_rows, neutral_rows) =
                self.derive_regime_buckets(base_ohlcv, sample_start, sampled_frame.height());

            for rows in [trend_rows, range_rows, neutral_rows] {
                if rows.len() < min_features.max(32) {
                    continue;
                }
                let (subset_frame, subset_labels) =
                    self.take_row_subset(&sampled_frame, &sampled_labels, &rows)?;
                for (name, score) in self.fit_l1_ranked_features(&subset_frame, &subset_labels)? {
                    *score_map.entry(name).or_insert(0.0) += score.max(0.0) * 0.75;
                }
            }
        }

        let mut ranked = score_map.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));

        let positive = ranked
            .iter()
            .filter(|(_, score)| *score > 1e-6)
            .count()
            .max(min_features)
            .min(max_features)
            .min(frame.width());

        let selected_names = ranked
            .into_iter()
            .take(positive)
            .map(|(name, _)| name)
            .collect::<Vec<_>>();

        if selected_names.is_empty() || selected_names.len() >= frame.width() {
            return Ok(frame.clone());
        }

        let selected_columns = selected_names
            .iter()
            .map(|name| {
                frame
                    .column(name)
                    .with_context(|| format!("selected feature column `{name}` missing from frame"))
                    .cloned()
            })
            .collect::<Result<Vec<_>>>()?;

        info!(
            "Applied L1 feature selection: kept {} / {} features",
            selected_columns.len(),
            frame.width()
        );
        DataFrame::new(selected_columns).context("rebuild feature-selected dataframe")
    }

    fn apply_base_signal_filter(
        &self,
        frame: &DataFrame,
        labels: &[i32],
    ) -> Result<(DataFrame, Vec<i32>)> {
        if frame.height() != labels.len() {
            anyhow::bail!(
                "base-signal filter row/label mismatch: {} rows vs {} labels",
                frame.height(),
                labels.len()
            );
        }

        let active_rows = labels.iter().filter(|label| **label != 0).count();
        if active_rows < 64 || active_rows == labels.len() {
            return Ok((frame.clone(), labels.to_vec()));
        }

        let mask = BooleanChunked::from_slice(
            "base_signal_filter".into(),
            &labels.iter().map(|label| *label != 0).collect::<Vec<_>>(),
        );
        let filtered_frame = frame.filter(&mask)?;
        let filtered_labels = labels
            .iter()
            .copied()
            .filter(|label| *label != 0)
            .collect::<Vec<_>>();

        if filtered_frame.height() != filtered_labels.len() || filtered_labels.is_empty() {
            anyhow::bail!(
                "base-signal filter produced inconsistent payload: {} rows vs {} labels",
                filtered_frame.height(),
                filtered_labels.len()
            );
        }

        info!(
            "Applied base-signal directional filter: kept {} / {} rows",
            filtered_frame.height(),
            frame.height()
        );
        Ok((filtered_frame, filtered_labels))
    }

    fn default_model_params(&self, name: &str) -> HashMap<String, String> {
        let canonical = canonical_model_name(name);
        match canonical {
            "xgboost_rf" => HashMap::from([
                ("variant".to_string(), "rf".to_string()),
                ("num_parallel_tree".to_string(), "64".to_string()),
                ("subsample".to_string(), "0.8".to_string()),
                ("colsample_bynode".to_string(), "0.8".to_string()),
                (
                    "device".to_string(),
                    self.settings.models.tree_device_preference.clone(),
                ),
            ]),
            "xgboost_dart" => HashMap::from([
                ("variant".to_string(), "dart".to_string()),
                ("rate_drop".to_string(), "0.1".to_string()),
                ("skip_drop".to_string(), "0.5".to_string()),
                (
                    "device".to_string(),
                    self.settings.models.tree_device_preference.clone(),
                ),
            ]),
            "catboost_alt" => HashMap::from([
                ("variant".to_string(), "alt".to_string()),
                ("depth".to_string(), "10".to_string()),
                ("l2_leaf_reg".to_string(), "5.0".to_string()),
                (
                    "device".to_string(),
                    self.settings.models.tree_device_preference.clone(),
                ),
            ]),
            "lightgbm" => HashMap::from([
                (
                    "device".to_string(),
                    self.settings.models.tree_device_preference.clone(),
                ),
                ("num_iterations".to_string(), "400".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
                ("max_depth".to_string(), "8".to_string()),
                ("num_leaves".to_string(), "31".to_string()),
            ]),
            "xgboost" => HashMap::from([
                (
                    "device".to_string(),
                    self.settings.models.tree_device_preference.clone(),
                ),
                ("n_estimators".to_string(), "800".to_string()),
                ("max_depth".to_string(), "8".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
            ]),
            "catboost" => HashMap::from([
                (
                    "device".to_string(),
                    self.settings.models.tree_device_preference.clone(),
                ),
                ("iterations".to_string(), "500".to_string()),
                ("depth".to_string(), "8".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
            ]),
            "mlp" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.nf_hidden_dim.to_string(),
                ),
                ("n_layers".to_string(), "3".to_string()),
                ("dropout".to_string(), "0.10".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("mlp", 100).to_string(),
                ),
            ]),
            "transformer" => {
                let replica_idx = transformer_replica_index(name).unwrap_or(1);
                let replica_offset = replica_idx.saturating_sub(1);
                let dropout = (self.settings.models.transformer_dropout
                    + replica_offset as f64 * 0.01)
                    .clamp(0.0, 0.35);
                let hidden_dim = self.transformer_hidden_dim();
                let n_heads = self.transformer_heads();
                let n_layers = self.transformer_layers();
                HashMap::from([
                    ("hidden_dim".to_string(), hidden_dim.to_string()),
                    ("n_heads".to_string(), n_heads.to_string()),
                    ("n_layers".to_string(), n_layers.to_string()),
                    ("dim_ff".to_string(), (hidden_dim * 2).to_string()),
                    ("dropout".to_string(), format!("{dropout:.4}")),
                    (
                        "batch_size".to_string(),
                        self.settings.models.train_batch_size.max(8).to_string(),
                    ),
                    (
                        "max_epochs".to_string(),
                        self.max_epochs_for_model("transformer", 120).to_string(),
                    ),
                    (
                        "seed".to_string(),
                        (42_u64 + replica_offset as u64 * 97).to_string(),
                    ),
                    ("ensemble_member".to_string(), replica_idx.to_string()),
                    ("device".to_string(), self.preferred_burn_device_policy()),
                ])
            }
            "nbeats" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.nbeats_hidden_dim.to_string(),
                ),
                ("n_blocks".to_string(), "4".to_string()),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("nbeats", 100).to_string(),
                ),
            ]),
            "tide" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.tide_hidden_dim.to_string(),
                ),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("tide", 100).to_string(),
                ),
            ]),
            "tabnet" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.tabnet_hidden_dim.to_string(),
                ),
                ("n_steps".to_string(), "5".to_string()),
                ("relaxation_factor".to_string(), "1.5".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("tabnet", 120).to_string(),
                ),
            ]),
            "kan" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.kan_hidden_dim.to_string(),
                ),
                ("n_layers".to_string(), "3".to_string()),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("kan", 100).to_string(),
                ),
            ]),
            "patchtst" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.transformer_hidden_dim().to_string(),
                ),
                ("n_heads".to_string(), self.transformer_heads().to_string()),
                (
                    "n_layers".to_string(),
                    self.transformer_layers().to_string(),
                ),
                (
                    "dim_ff".to_string(),
                    (self.transformer_hidden_dim() * 2).to_string(),
                ),
                ("dropout".to_string(), "0.10".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("patchtst", 120).to_string(),
                ),
            ]),
            "timesnet" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.nf_hidden_dim.to_string(),
                ),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("timesnet", 100).to_string(),
                ),
            ]),
            "nbeatsx_nf" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.nbeats_hidden_dim.to_string(),
                ),
                ("n_blocks".to_string(), "4".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("nbeatsx_nf", 100).to_string(),
                ),
            ]),
            "tide_nf" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.tide_hidden_dim.to_string(),
                ),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("tide_nf", 100).to_string(),
                ),
            ]),
            "meta_blender" => HashMap::from([("backend".to_string(), "xgboost".to_string())]),
            "probability_calibrator" => HashMap::from([
                (
                    "method".to_string(),
                    self.settings.models.calibration_method.clone(),
                ),
                (
                    "min_rows".to_string(),
                    self.min_calibration_rows().to_string(),
                ),
            ]),
            "conformal_gate" => HashMap::from([
                (
                    "method".to_string(),
                    self.settings.models.calibration_method.clone(),
                ),
                (
                    "alpha".to_string(),
                    format!("{:.6}", self.conformal_alpha()),
                ),
                (
                    "min_prediction_set".to_string(),
                    self.conformal_min_prediction_set().to_string(),
                ),
                (
                    "min_rows".to_string(),
                    self.min_calibration_rows().to_string(),
                ),
            ]),
            "meta_stack" => HashMap::from([
                (
                    "method".to_string(),
                    self.settings.models.calibration_method.clone(),
                ),
                (
                    "alpha".to_string(),
                    format!("{:.6}", self.conformal_alpha()),
                ),
                (
                    "min_prediction_set".to_string(),
                    self.conformal_min_prediction_set().to_string(),
                ),
                (
                    "min_rows".to_string(),
                    self.min_calibration_rows().to_string(),
                ),
            ]),
            "exit_agent" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.exit_agent_hidden_dim.to_string(),
                ),
                (
                    "gamma".to_string(),
                    format!("{:.6}", self.settings.models.exit_agent_gamma),
                ),
                (
                    "epsilon".to_string(),
                    format!("{:.6}", self.settings.models.exit_agent_epsilon),
                ),
                (
                    "epsilon_min".to_string(),
                    format!("{:.6}", self.settings.models.exit_agent_epsilon_min),
                ),
                (
                    "epsilon_decay".to_string(),
                    format!("{:.6}", self.settings.models.exit_agent_epsilon_decay),
                ),
                (
                    "memory_capacity".to_string(),
                    self.settings.models.exit_agent_memory_capacity.to_string(),
                ),
                (
                    "reward_horizon".to_string(),
                    if self.settings.models.exit_agent_reward_horizon == 0 {
                        self.settings
                            .risk
                            .triple_barrier_max_bars
                            .clamp(6, 64)
                            .to_string()
                    } else {
                        self.settings.models.exit_agent_reward_horizon.to_string()
                    },
                ),
                (
                    "warmup_steps".to_string(),
                    if self.settings.models.exit_agent_warmup_steps == 0 {
                        Self::epochs_from_seconds(self.settings.models.rl_train_seconds, 96)
                            .to_string()
                    } else {
                        self.settings.models.exit_agent_warmup_steps.to_string()
                    },
                ),
            ]),
            "online_pa" => HashMap::from([
                ("c".to_string(), "1.0".to_string()),
                (
                    "epochs".to_string(),
                    Self::epochs_from_seconds(self.settings.models.rl_train_seconds / 4, 8)
                        .to_string(),
                ),
            ]),
            "online_hoeffding" => HashMap::from([
                ("n_steps".to_string(), "24".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
                ("feature_subsample_rate".to_string(), "0.8".to_string()),
                ("max_depth".to_string(), "5".to_string()),
                ("n_bins".to_string(), "32".to_string()),
                ("grace_period".to_string(), "32".to_string()),
                (
                    "drift_detector".to_string(),
                    if self.settings.risk.feature_drift_threshold <= 0.20 {
                        "adwin".to_string()
                    } else {
                        "page_hinkley".to_string()
                    },
                ),
                (
                    "drift_delta".to_string(),
                    format!(
                        "{:.6}",
                        (self.settings.risk.feature_drift_threshold / 100.0).clamp(0.0005, 0.02)
                    ),
                ),
            ]),
            "isolation_forest" => HashMap::from([
                ("n_trees".to_string(), "128".to_string()),
                ("sample_size".to_string(), "256".to_string()),
            ]),
            "genetic" => HashMap::from([
                (
                    "population".to_string(),
                    self.settings
                        .models
                        .prop_search_population
                        .max(16)
                        .to_string(),
                ),
                (
                    "generations".to_string(),
                    self.settings
                        .models
                        .prop_search_generations
                        .max(1)
                        .to_string(),
                ),
                (
                    "max_indicators".to_string(),
                    if self.settings.models.prop_search_max_indicators == 0 {
                        "64".to_string()
                    } else {
                        self.settings.models.prop_search_max_indicators.to_string()
                    },
                ),
                (
                    "portfolio_size".to_string(),
                    self.settings
                        .models
                        .prop_search_portfolio_size
                        .clamp(4, self.settings.models.prop_search_population.max(4))
                        .to_string(),
                ),
                (
                    "parent_selection".to_string(),
                    self.settings.models.prop_search_parent_selection.clone(),
                ),
                (
                    "survivor_selection".to_string(),
                    self.settings.models.prop_search_survivor_selection.clone(),
                ),
                (
                    "survivor_fraction".to_string(),
                    format!("{:.6}", self.settings.models.prop_search_survivor_fraction),
                ),
                (
                    "immigrant_fraction".to_string(),
                    format!("{:.6}", self.settings.models.prop_search_immigrant_fraction),
                ),
                (
                    "selection_temperature".to_string(),
                    format!(
                        "{:.6}",
                        self.settings.models.prop_search_selection_temperature
                    ),
                ),
                (
                    "train_years".to_string(),
                    self.settings.models.prop_search_train_years.to_string(),
                ),
                (
                    "val_years".to_string(),
                    self.settings.models.prop_search_val_years.to_string(),
                ),
                (
                    "device".to_string(),
                    self.settings.models.prop_search_device.clone(),
                ),
                (
                    "checkpoint".to_string(),
                    self.settings
                        .models
                        .prop_search_checkpoint
                        .to_string_lossy()
                        .to_string(),
                ),
                (
                    "async".to_string(),
                    self.settings.models.prop_search_async.to_string(),
                ),
                (
                    "async_wait".to_string(),
                    self.settings.models.prop_search_async_wait.to_string(),
                ),
                (
                    "tournament_size".to_string(),
                    if self.settings.models.prop_search_tournament_size == 0 {
                        (self.settings.models.prop_search_population.max(16) / 10)
                            .max(3)
                            .to_string()
                    } else {
                        self.settings
                            .models
                            .prop_search_tournament_size
                            .max(3)
                            .to_string()
                    },
                ),
            ]),
            "dqn" => HashMap::from([
                (
                    "backend".to_string(),
                    if self.settings.models.use_rllib_agent
                        || (self.settings.models.auto_enable_rllib
                            && self.settings.system.enable_gpu
                            && self.settings.models.ray_tune_max_concurrency > 0)
                    {
                        "rllib".to_string()
                    } else {
                        "native".to_string()
                    },
                ),
                (
                    "auto_rllib".to_string(),
                    (self.settings.models.auto_enable_rllib
                        && !self.settings.models.use_rllib_agent)
                        .to_string(),
                ),
                (
                    "epochs".to_string(),
                    Self::epochs_from_seconds(self.settings.models.rl_train_seconds, 48)
                        .to_string(),
                ),
                (
                    "max_steps".to_string(),
                    (self.settings.models.rl_timesteps / 10_000)
                        .clamp(128, 4096)
                        .to_string(),
                ),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(32).to_string(),
                ),
                (
                    "state_bins".to_string(),
                    self.settings.models.rl_state_bins.to_string(),
                ),
                (
                    "state_encoding".to_string(),
                    self.settings.models.rl_state_encoding.clone(),
                ),
                (
                    "update_interval".to_string(),
                    if self.settings.models.rl_update_interval == 0 {
                        self.settings
                            .models
                            .rl_parallel_envs
                            .clamp(8, 128)
                            .to_string()
                    } else {
                        self.settings.models.rl_update_interval.max(1).to_string()
                    },
                ),
                (
                    "update_freq".to_string(),
                    if self.settings.models.rl_update_freq == 0 {
                        self.settings
                            .models
                            .rl_eval_episodes
                            .clamp(1, 16)
                            .to_string()
                    } else {
                        self.settings.models.rl_update_freq.max(1).to_string()
                    },
                ),
                (
                    "learning_rate".to_string(),
                    format!("{:.6}", self.settings.models.rl_learning_rate),
                ),
                (
                    "gamma".to_string(),
                    format!("{:.6}", self.settings.models.rl_gamma),
                ),
                (
                    "epsilon_start".to_string(),
                    format!("{:.6}", self.settings.models.rl_epsilon_start),
                ),
                (
                    "epsilon_end".to_string(),
                    format!("{:.6}", self.settings.models.rl_epsilon_end),
                ),
                (
                    "epsilon_decay".to_string(),
                    format!("{:.6}", self.settings.models.rl_epsilon_decay),
                ),
                (
                    "buffer_capacity".to_string(),
                    if self.settings.models.rl_buffer_capacity == 0 {
                        (self.settings.models.train_batch_size.max(32) * 1024).to_string()
                    } else {
                        self.settings.models.rl_buffer_capacity.to_string()
                    },
                ),
                (
                    "parallel_envs".to_string(),
                    self.settings.models.rl_parallel_envs.max(1).to_string(),
                ),
                (
                    "eval_episodes".to_string(),
                    self.settings.models.rl_eval_episodes.max(1).to_string(),
                ),
                (
                    "reward_horizon".to_string(),
                    if self.settings.models.rl_reward_horizon == 0 {
                        self.settings
                            .risk
                            .triple_barrier_max_bars
                            .clamp(6, 64)
                            .to_string()
                    } else {
                        self.settings.models.rl_reward_horizon.to_string()
                    },
                ),
                (
                    "episode_len".to_string(),
                    if self.settings.models.rl_episode_len == 0 {
                        self.settings
                            .models
                            .transformer_seq_len
                            .clamp(24, 256)
                            .to_string()
                    } else {
                        self.settings.models.rl_episode_len.to_string()
                    },
                ),
                (
                    "rllib_num_workers".to_string(),
                    self.settings.models.rllib_num_workers.to_string(),
                ),
                (
                    "ray_tune_max_concurrency".to_string(),
                    self.settings
                        .models
                        .ray_tune_max_concurrency
                        .max(1)
                        .to_string(),
                ),
                ("device".to_string(), self.settings.system.device.clone()),
                (
                    "hidden_dims".to_string(),
                    self.settings
                        .models
                        .rl_network_arch
                        .iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ]),
            "swarm_forecaster" => HashMap::from([
                (
                    "memory_limit_mb".to_string(),
                    format!("{:.2}", self.settings.models.swarm_memory_limit_mb),
                ),
                (
                    "horizon".to_string(),
                    if self.settings.models.swarm_horizon == 0 {
                        self.settings
                            .risk
                            .triple_barrier_max_bars
                            .clamp(8, 64)
                            .to_string()
                    } else {
                        self.settings.models.swarm_horizon.to_string()
                    },
                ),
                (
                    "frequency".to_string(),
                    self.settings.models.swarm_frequency.clone(),
                ),
                (
                    "strategy".to_string(),
                    self.settings.models.swarm_strategy.clone(),
                ),
                (
                    "accuracy_target".to_string(),
                    format!(
                        "{:.4}",
                        self.settings.models.prop_conf_threshold.clamp(0.5, 0.99)
                    ),
                ),
                (
                    "latency_ms".to_string(),
                    if self.settings.models.swarm_latency_ms == 0 {
                        self.settings
                            .system
                            .poll_interval_seconds
                            .saturating_mul(1000)
                            .to_string()
                    } else {
                        self.settings.models.swarm_latency_ms.to_string()
                    },
                ),
                (
                    "online_learning".to_string(),
                    self.settings.models.swarm_online_learning.to_string(),
                ),
                (
                    "interpretability_needed".to_string(),
                    self.settings
                        .models
                        .swarm_interpretability_needed
                        .to_string(),
                ),
            ]),
            "neuro_evo" => HashMap::from([
                (
                    "hidden_dim".to_string(),
                    self.settings.models.evo_hidden_size.to_string(),
                ),
                (
                    "sigma".to_string(),
                    format!("{:.6}", self.settings.models.evo_sigma),
                ),
                (
                    "generations".to_string(),
                    Self::epochs_from_seconds(self.settings.models.evo_train_seconds, 24)
                        .to_string(),
                ),
                (
                    "population".to_string(),
                    self.settings.models.evo_population.max(16).to_string(),
                ),
                (
                    "islands".to_string(),
                    self.settings.models.evo_islands.max(1).to_string(),
                ),
            ]),
            "neat" => HashMap::from([
                (
                    "population".to_string(),
                    self.settings.models.evo_population.max(48).to_string(),
                ),
                (
                    "generations".to_string(),
                    Self::epochs_from_seconds(self.settings.models.evo_train_seconds, 48)
                        .to_string(),
                ),
                (
                    "mutation_rate".to_string(),
                    format!(
                        "{:.6}",
                        (self.settings.models.evo_sigma * 2.4).clamp(0.2, 1.2)
                    ),
                ),
                (
                    "species_elitism".to_string(),
                    self.settings.models.evo_islands.clamp(1, 3).to_string(),
                ),
                (
                    "compatibility_threshold".to_string(),
                    format!(
                        "{:.6}",
                        (1.5 + self.settings.models.evo_sigma * 4.0).clamp(1.5, 4.0)
                    ),
                ),
                ("immigrant_fraction".to_string(), "0.100000".to_string()),
                ("seed".to_string(), "42".to_string()),
            ]),
            _ => HashMap::new(),
        }
    }

    fn map_model_type(&self, name: &str) -> Result<ModelType> {
        match canonical_model_name(name) {
            "lightgbm" => Ok(ModelType::LightGBM),
            "xgboost" | "xgboost_rf" | "xgboost_dart" => Ok(ModelType::XGBoost),
            "catboost" | "catboost_alt" => Ok(ModelType::CatBoost),
            "sklears_tree" => Ok(ModelType::SklearsTree),
            "mlp" => Ok(ModelType::MLP),
            "nbeats" => Ok(ModelType::NBeats),
            "nbeatsx_nf" => Ok(ModelType::NBeatsxNf),
            "tide" => Ok(ModelType::TiDE),
            "tide_nf" => Ok(ModelType::TiDENf),
            "tabnet" => Ok(ModelType::TabNet),
            "kan" => Ok(ModelType::KAN),
            "transformer" => Ok(ModelType::Transformer),
            "patchtst" => Ok(ModelType::PatchTST),
            "timesnet" => Ok(ModelType::TimesNet),
            "elasticnet" => Ok(ModelType::ElasticNet),
            "bayes_logit" => Ok(ModelType::BayesianLogit),
            "meta_blender" => Ok(ModelType::MetaBlender),
            "probability_calibrator" => Ok(ModelType::ProbabilityCalibrator),
            "conformal_gate" => Ok(ModelType::ConformalGate),
            "meta_stack" => Ok(ModelType::MetaStack),
            "exit_agent" => Ok(ModelType::ExitAgent),
            "online_pa" => Ok(ModelType::OnlinePassiveAggressive),
            "online_hoeffding" => Ok(ModelType::OnlineHoeffding),
            "isolation_forest" => Ok(ModelType::IsolationForest),
            "dqn" => Ok(ModelType::Dqn),
            "swarm_forecaster" => Ok(ModelType::SwarmForecaster),
            "genetic" => Ok(ModelType::Genetic),
            "neuro_evo" => Ok(ModelType::NeuroEvo),
            "neat" => Ok(ModelType::Neat),
            other => anyhow::bail!(
                "Model `{other}` does not have a concrete ModelType mapping in the orchestrator"
            ),
        }
    }

    fn derive_labels(&self, ohlcv: &Ohlcv) -> Vec<i32> {
        let n = ohlcv.close.len();
        if n == 0 {
            return Vec::new();
        }
        assert_eq!(
            ohlcv.open.len(),
            n,
            "derive_labels requires aligned OHLCV series: open={} close={}",
            ohlcv.open.len(),
            n
        );
        assert_eq!(
            ohlcv.high.len(),
            n,
            "derive_labels requires aligned OHLCV series: high={} close={}",
            ohlcv.high.len(),
            n
        );
        assert_eq!(
            ohlcv.low.len(),
            n,
            "derive_labels requires aligned OHLCV series: low={} close={}",
            ohlcv.low.len(),
            n
        );

        let mut labels = vec![0; n];
        let hold_bars = self.effective_label_horizon_bars();
        let atr_period = self.settings.risk.atr_period.max(2);
        let min_distance = self.settings.risk.meta_label_min_dist.max(1e-6);
        let atr_stop_multiplier = self
            .settings
            .models
            .label_stop_atr_multiplier
            .max(self.settings.risk.atr_stop_multiplier)
            .max(0.1);
        let base_rr = self
            .settings
            .models
            .label_take_profit_rr
            .max(self.settings.risk.min_risk_reward)
            .max(1.0);
        let fixed_sl = self.settings.risk.meta_label_fixed_sl.max(min_distance);
        let fixed_tp = self.settings.risk.meta_label_fixed_tp.max(min_distance);
        let true_ranges = compute_true_ranges(ohlcv);
        let use_triple_barrier = self.settings.models.label_use_triple_barrier;
        let neutral_band_fraction = self
            .settings
            .models
            .label_neutral_band_atr_fraction
            .clamp(0.05, 1.0);

        for (i, slot) in labels.iter_mut().enumerate() {
            let entry = ohlcv.close[i];
            if !entry.is_finite() || entry.abs() <= f64::EPSILON || i + 1 >= n {
                continue;
            }

            let atr = trailing_average(&true_ranges, i, atr_period).max(min_distance);
            let stop_distance = fixed_sl.max(atr * atr_stop_multiplier).max(min_distance);
            let take_profit_distance = fixed_tp.max(stop_distance * base_rr).max(min_distance);
            let horizon_end = (i + hold_bars).min(n - 1);
            if use_triple_barrier {
                let upper_barrier = entry + take_profit_distance;
                let lower_barrier = entry - stop_distance;

                for forward_idx in (i + 1)..=horizon_end {
                    let high = ohlcv.high[forward_idx];
                    let low = ohlcv.low[forward_idx];
                    let close = ohlcv.close[forward_idx];
                    let tp_hit = high.is_finite() && high >= upper_barrier;
                    let sl_hit = low.is_finite() && low <= lower_barrier;

                    if tp_hit && sl_hit {
                        let upper_dist = (close - upper_barrier).abs();
                        let lower_dist = (close - lower_barrier).abs();
                        *slot = if upper_dist < lower_dist {
                            1
                        } else if lower_dist < upper_dist {
                            -1
                        } else if close >= entry {
                            1
                        } else {
                            -1
                        };
                        break;
                    }

                    if tp_hit {
                        *slot = 1;
                        break;
                    }

                    if sl_hit {
                        *slot = -1;
                        break;
                    }
                }
            }

            if *slot != 0 {
                continue;
            }

            let terminal_close = ohlcv.close[horizon_end];
            let terminal_move = terminal_close - entry;
            let neutral_band = min_distance.max(atr * neutral_band_fraction);
            if terminal_move >= neutral_band {
                *slot = 1;
            } else if terminal_move <= -neutral_band {
                *slot = -1;
            }
        }

        labels
    }
}

fn compute_true_ranges(ohlcv: &Ohlcv) -> Vec<f64> {
    let mut true_ranges = Vec::with_capacity(ohlcv.close.len());
    let mut previous_close: Option<f64> = None;

    for idx in 0..ohlcv.close.len() {
        let high = ohlcv
            .high
            .get(idx)
            .copied()
            .unwrap_or_else(|| ohlcv.close[idx]);
        let low = ohlcv
            .low
            .get(idx)
            .copied()
            .unwrap_or_else(|| ohlcv.close[idx]);
        let close = ohlcv.close[idx];
        let range = match previous_close {
            Some(prev_close) => {
                let intrabar = (high - low).abs();
                let gap_high = (high - prev_close).abs();
                let gap_low = (low - prev_close).abs();
                intrabar.max(gap_high).max(gap_low)
            }
            None => (high - low).abs(),
        };
        true_ranges.push(if range.is_finite() { range } else { 0.0 });
        previous_close = Some(close);
    }

    true_ranges
}

fn trailing_average(values: &[f64], end_idx: usize, window: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let window = window.max(1);
    let start_idx = end_idx.saturating_add(1).saturating_sub(window);
    let slice = &values[start_idx..=end_idx.min(values.len() - 1)];
    if slice.is_empty() {
        0.0
    } else {
        slice.iter().copied().sum::<f64>() / slice.len() as f64
    }
}

fn labels_to_series(labels: &[i32]) -> Series {
    Series::new("label".into(), labels.to_vec())
}

fn canonical_model_name(name: &str) -> &str {
    if name
        .strip_prefix("transformer_")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()))
    {
        "transformer"
    } else {
        name
    }
}

fn transformer_replica_index(name: &str) -> Option<usize> {
    name.strip_prefix("transformer_")
        .and_then(|suffix| suffix.parse::<usize>().ok())
        .filter(|idx| *idx > 0)
}

fn configured_contains_model(configured: &[String], candidate: &str) -> bool {
    let canonical = canonical_model_name(candidate);
    configured
        .iter()
        .any(|name| name == candidate || canonical_model_name(name) == canonical)
}

fn parse_tree_params(params: &HashMap<String, String>) -> HashMap<String, ParamValue> {
    params
        .iter()
        .map(|(key, value)| {
            let parsed = match value.trim().to_ascii_lowercase().as_str() {
                "true" => ParamValue::Bool(true),
                "false" => ParamValue::Bool(false),
                _ => {
                    if let Ok(parsed) = value.parse::<i32>() {
                        ParamValue::Int(parsed)
                    } else if let Ok(parsed) = value.parse::<f64>() {
                        ParamValue::Float(parsed)
                    } else {
                        ParamValue::String(value.clone())
                    }
                }
            };
            (key.clone(), parsed)
        })
        .collect()
}

fn parse_f32_param(params: &HashMap<String, String>, key: &str, default: f32) -> f32 {
    params
        .get(key)
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn parse_f64_param(params: &HashMap<String, String>, key: &str, default: f64) -> f64 {
    params
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn parse_usize_param(params: &HashMap<String, String>, key: &str, default: usize) -> usize {
    params
        .get(key)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_u64_param(params: &HashMap<String, String>, key: &str, default: u64) -> u64 {
    params
        .get(key)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_bool_param(params: &HashMap<String, String>, key: &str, default: bool) -> bool {
    params
        .get(key)
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => true,
            "false" | "0" | "no" | "off" => false,
            _ => default,
        })
        .unwrap_or(default)
}

fn parse_string_param(params: &HashMap<String, String>, key: &str) -> Option<String> {
    params
        .get(key)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn model_params_only(params: &HashMap<String, String>) -> HashMap<String, String> {
    params
        .iter()
        .filter(|(key, _)| !key.starts_with("__"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn hpo_backend_from_params(params: &HashMap<String, String>) -> String {
    params
        .get("__hpo_backend")
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "grid".to_string())
}

fn hpo_trials_from_params(params: &HashMap<String, String>) -> usize {
    parse_usize_param(params, "__hpo_trials", 1).max(1)
}

fn hpo_max_rows_from_params(params: &HashMap<String, String>) -> usize {
    parse_usize_param(params, "__hpo_max_rows", 0)
}

fn embargo_minutes_from_params(params: &HashMap<String, String>) -> usize {
    parse_usize_param(params, "__embargo_minutes", 0)
}

fn holdout_pct_from_params(params: &HashMap<String, String>) -> f64 {
    parse_f64_param(params, "__holdout_pct", 0.20).clamp(0.05, 0.45)
}

fn confidence_threshold_from_params(params: &HashMap<String, String>) -> f32 {
    parse_f32_param(params, "__conf_threshold", 0.55).clamp(0.0, 1.0)
}

fn metric_weight_from_params(params: &HashMap<String, String>) -> f64 {
    parse_f64_param(params, "__metric_weight", 1.0).max(0.0)
}

fn accuracy_weight_from_params(params: &HashMap<String, String>) -> f64 {
    parse_f64_param(params, "__accuracy_weight", 0.10).clamp(0.0, 1.0)
}

fn export_onnx_requested(params: &HashMap<String, String>) -> bool {
    parse_bool_param(params, "__export_onnx", false)
}

fn parse_parent_selection_policy(params: &HashMap<String, String>) -> ParentSelectionPolicy {
    ParentSelectionPolicy::parse(
        parse_string_param(params, "parent_selection")
            .as_deref()
            .unwrap_or("rank"),
    )
}

fn parse_survivor_selection_policy(params: &HashMap<String, String>) -> SurvivorSelectionPolicy {
    SurvivorSelectionPolicy::parse(
        parse_string_param(params, "survivor_selection")
            .as_deref()
            .unwrap_or("rank"),
    )
}

fn training_runtime_profile(
    settings: &forex_core::Settings,
    config: &ModelConfig,
    symbol: &str,
    base_tf: &str,
    payload: &TrainingPayload,
    row_budget_applied: Option<usize>,
    higher_timeframes: Vec<String>,
) -> TrainingRuntimeProfile {
    let effective_label_horizon_bars = if settings.models.label_horizon_bars > 0 {
        settings.models.label_horizon_bars
    } else {
        settings.risk.meta_label_max_hold_bars.max(1)
    };
    let requested_backend = parse_string_param(&config.params, "backend");
    let requested_device = parse_string_param(&config.params, "device");
    let rllib_requested = requested_backend
        .as_deref()
        .is_some_and(|backend| backend.eq_ignore_ascii_case("rllib"));
    let mut notes = Vec::new();
    if rllib_requested {
        notes.push(
            "rllib backend is recorded as a requested runtime hint; current DQN training path remains native rlkit".to_string(),
        );
        if parse_bool_param(&config.params, "auto_rllib", false) {
            notes.push(
                "rllib backend was auto-requested from config because GPU preference and RLlib auto-enable were both active".to_string(),
            );
        }
    }
    if settings.models.enable_ddp || settings.models.enable_fsdp {
        notes.push(
            format!(
                "distributed deep-learning flags are recorded for runtime traceability; active burn backend in this build is `{}` and training still executes as a single-process local run",
                active_burn_backend_name()
            ),
        );
    }
    if parse_bool_param(&config.params, "async", false) {
        notes.push(
            "async discovery/search hint recorded; current model-training dispatch still executes synchronously inside the orchestrator".to_string(),
        );
    }
    if parse_string_param(&config.params, "checkpoint").is_some() {
        notes.push(
            "checkpoint path recorded for future search persistence; current model artifact path remains the primary persisted output".to_string(),
        );
    }

    TrainingRuntimeProfile {
        model_name: config.name.clone(),
        capability_family: config.capability_family,
        capability_state: config.capability_state,
        symbol: symbol.to_string(),
        base_timeframe: base_tf.to_string(),
        feature_count: payload.frame.width(),
        dataset_rows: payload.frame.height(),
        row_budget_applied,
        label_horizon_bars: settings.models.label_horizon_bars,
        effective_label_horizon_bars,
        meta_label_max_hold_bars: settings.risk.meta_label_max_hold_bars.max(1),
        label_use_triple_barrier: settings.models.label_use_triple_barrier,
        higher_timeframes,
        multi_resolution_enabled: settings.system.multi_resolution_enabled,
        base_features_prefixed: settings.system.multi_resolution_prefix_base,
        base_signal_filter_enabled: settings.models.filter_to_base_signal,
        l1_feature_selection_enabled: settings.models.l1_feature_selection_enabled,
        requested_backend,
        requested_device,
        checkpoint_path: parse_string_param(&config.params, "checkpoint").map(PathBuf::from),
        async_requested: parse_bool_param(&config.params, "async", false),
        async_wait_requested: parse_bool_param(&config.params, "async_wait", false),
        train_years: parse_usize_param(&config.params, "train_years", 0),
        val_years: parse_usize_param(&config.params, "val_years", 0),
        requested_hpo_backend: hpo_backend_from_params(&config.params),
        requested_hpo_trials: hpo_trials_from_params(&config.params),
        holdout_pct: holdout_pct_from_params(&config.params),
        embargo_minutes: embargo_minutes_from_params(&config.params),
        export_onnx_requested: export_onnx_requested(&config.params),
        rllib_requested,
        rllib_num_workers: parse_usize_param(
            &config.params,
            "rllib_num_workers",
            settings.models.rllib_num_workers,
        ),
        ray_tune_max_concurrency: parse_usize_param(
            &config.params,
            "ray_tune_max_concurrency",
            settings.models.ray_tune_max_concurrency.max(1),
        ),
        ddp_enabled: settings.models.enable_ddp,
        fsdp_enabled: settings.models.enable_fsdp,
        ddp_world_size: settings.models.ddp_world_size.max(1),
        symbol_hash_buckets: settings.models.symbol_hash_buckets.max(1),
        notes,
    }
}

fn write_training_profile_sidecar(
    artifact_dir: &std::path::Path,
    settings: &forex_core::Settings,
    config: &ModelConfig,
    symbol: &str,
    base_tf: &str,
    payload: &TrainingPayload,
    row_budget_applied: Option<usize>,
) -> Result<()> {
    let higher_timeframes = if settings.system.multi_resolution_enabled
        && !settings.system.multi_resolution_timeframes.is_empty()
    {
        settings
            .system
            .multi_resolution_timeframes
            .iter()
            .filter(|tf| !tf.eq_ignore_ascii_case(base_tf))
            .cloned()
            .collect::<Vec<_>>()
    } else {
        settings
            .system
            .higher_timeframes
            .iter()
            .filter(|tf| !tf.eq_ignore_ascii_case(base_tf))
            .cloned()
            .collect::<Vec<_>>()
    };

    let profile = training_runtime_profile(
        settings,
        config,
        symbol,
        base_tf,
        payload,
        row_budget_applied,
        higher_timeframes,
    );
    write_training_runtime_profile(
        &artifact_dir.join(TRAINING_RUNTIME_PROFILE_FILE_NAME),
        &profile,
    )
}

fn timeframe_to_minutes(base_tf: &str) -> usize {
    let upper = base_tf.trim().to_ascii_uppercase();
    if let Some(minutes) = upper
        .strip_prefix('M')
        .and_then(|value| value.parse::<usize>().ok())
    {
        return minutes.max(1);
    }
    if let Some(hours) = upper
        .strip_prefix('H')
        .and_then(|value| value.parse::<usize>().ok())
    {
        return (hours.max(1)) * 60;
    }
    if let Some(days) = upper
        .strip_prefix('D')
        .and_then(|value| value.parse::<usize>().ok())
    {
        return (days.max(1)) * 1_440;
    }
    if let Some(weeks) = upper
        .strip_prefix('W')
        .and_then(|value| value.parse::<usize>().ok())
    {
        return (weeks.max(1)) * 10_080;
    }
    if let Some(months) = upper
        .strip_prefix("MN")
        .and_then(|value| value.parse::<usize>().ok())
    {
        return (months.max(1)) * 43_200;
    }
    1
}

fn embargo_rows_for_timeframe(base_tf: &str, embargo_minutes: usize) -> usize {
    let tf_minutes = timeframe_to_minutes(base_tf).max(1);
    if embargo_minutes == 0 {
        0
    } else {
        ((embargo_minutes as f64) / (tf_minutes as f64)).ceil() as usize
    }
}

fn halton(mut index: usize, base: usize) -> f64 {
    let mut fraction = 1.0_f64;
    let mut result = 0.0_f64;
    let base = base.max(2);
    while index > 0 {
        fraction /= base as f64;
        result += fraction * (index % base) as f64;
        index /= base;
    }
    result
}

fn sample_choice(
    values: &[&str],
    trial_idx: usize,
    trials: usize,
    backend: &str,
    base: usize,
) -> String {
    if values.is_empty() {
        return String::new();
    }
    let index = if backend == "grid" {
        if trials <= 1 {
            0
        } else {
            trial_idx % values.len()
        }
    } else {
        let frac = halton(trial_idx + 1, base);
        ((frac * values.len() as f64).floor() as usize).min(values.len().saturating_sub(1))
    };
    values[index].to_string()
}

fn sample_f64(
    min: f64,
    max: f64,
    trial_idx: usize,
    trials: usize,
    backend: &str,
    base: usize,
) -> f64 {
    if (max - min).abs() < f64::EPSILON {
        return min;
    }
    let fraction = if backend == "grid" {
        if trials <= 1 {
            0.5
        } else {
            trial_idx as f64 / (trials.saturating_sub(1).max(1)) as f64
        }
    } else {
        halton(trial_idx + 1, base)
    };
    min + ((max - min) * fraction.clamp(0.0, 1.0))
}

fn sample_usize(
    values: &[usize],
    trial_idx: usize,
    trials: usize,
    backend: &str,
    base: usize,
) -> usize {
    if values.is_empty() {
        return 0;
    }
    let index = if backend == "grid" {
        if trials <= 1 {
            0
        } else {
            trial_idx % values.len()
        }
    } else {
        let frac = halton(trial_idx + 1, base);
        ((frac * values.len() as f64).floor() as usize).min(values.len().saturating_sub(1))
    };
    values[index]
}

fn calibration_method_from_params(params: &HashMap<String, String>) -> CalibrationMethod {
    match params
        .get("method")
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("identity") => CalibrationMethod::Identity,
        Some("temperature") => CalibrationMethod::Temperature,
        _ => CalibrationMethod::Platt,
    }
}

fn model_artifact_dir(
    models_dir: &std::path::Path,
    symbol: &str,
    base_tf: &str,
    model_name: &str,
) -> PathBuf {
    models_dir.join(symbol).join(base_tf).join(model_name)
}

fn supports_hpo(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::LightGBM
            | ModelType::XGBoost
            | ModelType::CatBoost
            | ModelType::MLP
            | ModelType::NBeats
            | ModelType::NBeatsxNf
            | ModelType::TiDE
            | ModelType::TiDENf
            | ModelType::TabNet
            | ModelType::KAN
            | ModelType::Transformer
            | ModelType::PatchTST
            | ModelType::TimesNet
            | ModelType::ElasticNet
            | ModelType::OnlinePassiveAggressive
            | ModelType::OnlineHoeffding
            | ModelType::IsolationForest
            | ModelType::NeuroEvo
            | ModelType::Neat
    )
}

fn uses_shared_expert_dispatch(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::LightGBM
            | ModelType::XGBoost
            | ModelType::CatBoost
            | ModelType::MLP
            | ModelType::NBeats
            | ModelType::NBeatsxNf
            | ModelType::TiDE
            | ModelType::TiDENf
            | ModelType::TabNet
            | ModelType::KAN
            | ModelType::Transformer
            | ModelType::PatchTST
            | ModelType::TimesNet
            | ModelType::ElasticNet
            | ModelType::BayesianLogit
            | ModelType::MetaBlender
            | ModelType::ProbabilityCalibrator
            | ModelType::ConformalGate
            | ModelType::MetaStack
            | ModelType::OnlinePassiveAggressive
            | ModelType::OnlineHoeffding
            | ModelType::IsolationForest
            | ModelType::NeuroEvo
            | ModelType::Neat
    )
}

fn build_expert_model(
    config: &ModelConfig,
    input_dim: usize,
    params: &HashMap<String, String>,
) -> Result<Box<dyn ExpertModel>> {
    match config.model_type {
        ModelType::LightGBM => Ok(Box::new(LightGBMExpert::new(
            0,
            Some(parse_tree_params(params)),
        ))),
        ModelType::XGBoost => Ok(Box::new(XGBoostExpert::new(
            0,
            Some(parse_tree_params(params)),
        ))),
        ModelType::CatBoost => {
            let mut model = CatBoostExpert::new(0);
            model.config.params = parse_tree_params(params);
            Ok(Box::new(model))
        }
        ModelType::ElasticNet => {
            let mut model = ElasticNetExpert::new(
                parse_f64_param(params, "alpha", 0.1),
                parse_f64_param(params, "l1_ratio", 0.5),
            );
            model.learning_rate = parse_f32_param(params, "lr", model.learning_rate);
            model.epochs = parse_usize_param(params, "epochs", model.epochs);
            Ok(Box::new(model))
        }
        ModelType::BayesianLogit => Ok(Box::new(BayesianLogitExpert::new())),
        ModelType::MetaBlender => Ok(Box::new(MetaBlender::new())),
        ModelType::ProbabilityCalibrator => {
            let mut model =
                ProbabilityCalibrationExpert::new(calibration_method_from_params(params));
            model.min_fit_rows = parse_usize_param(params, "min_rows", model.min_fit_rows);
            Ok(Box::new(model))
        }
        ModelType::ConformalGate => {
            let mut model = ConformalPredictionExpert::new(
                calibration_method_from_params(params),
                parse_f32_param(params, "alpha", 0.10),
            );
            model.min_prediction_set =
                parse_usize_param(params, "min_prediction_set", model.min_prediction_set);
            model.min_fit_rows = parse_usize_param(params, "min_rows", model.min_fit_rows);
            Ok(Box::new(model))
        }
        ModelType::MetaStack => {
            let mut model = MetaDecisionStack::new(
                calibration_method_from_params(params),
                parse_f32_param(params, "alpha", 0.10),
            );
            model.min_prediction_set =
                parse_usize_param(params, "min_prediction_set", model.min_prediction_set);
            model.min_fit_rows = parse_usize_param(params, "min_rows", model.min_fit_rows);
            Ok(Box::new(model))
        }
        ModelType::OnlinePassiveAggressive => Ok(Box::new(OnlinePassiveAggressiveExpert::new(
            parse_f32_param(params, "c", 1.0),
            parse_usize_param(params, "epochs", 4),
        ))),
        ModelType::OnlineHoeffding => {
            Ok(Box::new(OnlineHoeffdingExpert::new(Some(params.clone()))))
        }
        ModelType::IsolationForest => {
            let mut model = IsolationForestExpert::new(
                parse_usize_param(params, "n_trees", 128),
                parse_usize_param(params, "sample_size", 256),
            );
            if let Some(extension_level) = params
                .get("extension_level")
                .and_then(|value| value.parse::<usize>().ok())
            {
                model.extension_level = extension_level;
            }
            if let Some(max_tree_depth) = params
                .get("max_tree_depth")
                .and_then(|value| value.parse::<usize>().ok())
            {
                model.max_tree_depth = Some(max_tree_depth);
            }
            Ok(Box::new(model))
        }
        ModelType::NeuroEvo => Ok(Box::new(
            NeuroEvoExpert::with_config(
                input_dim.max(1),
                parse_usize_param(params, "hidden_dim", 32),
                parse_f64_param(params, "sigma", 0.25),
                parse_usize_param(params, "generations", 24),
            )
            .with_search_topology(
                parse_usize_param(params, "population", 16),
                parse_usize_param(params, "islands", 1),
            ),
        )),
        ModelType::Neat => Ok(Box::new(
            NeatExpert::with_config(
                input_dim.max(1),
                parse_usize_param(params, "population", 96),
                parse_usize_param(params, "generations", 48),
            )
            .with_search_params(
                parse_f32_param(params, "mutation_rate", 0.85),
                parse_usize_param(params, "species_elitism", 1),
                parse_f32_param(params, "compatibility_threshold", 2.5),
                parse_f32_param(params, "immigrant_fraction", 0.1),
                parse_u64_param(params, "seed", 42),
            ),
        )),
        ModelType::MLP => Ok(Box::new(MLPExpert::new(42, Some(params.clone())))),
        ModelType::NBeats => Ok(Box::new(NBeatsExpert::new(42, Some(params.clone())))),
        ModelType::NBeatsxNf => Ok(Box::new(NBeatsxNfExpert::new(42, Some(params.clone())))),
        ModelType::TiDE => Ok(Box::new(TiDEExpert::new(42, Some(params.clone())))),
        ModelType::TiDENf => Ok(Box::new(TiDENfExpert::new(42, Some(params.clone())))),
        ModelType::TabNet => Ok(Box::new(TabNetExpert::new(42, Some(params.clone())))),
        ModelType::KAN => Ok(Box::new(KANExpert::new(42, Some(params.clone())))),
        ModelType::Transformer => Ok(Box::new(TransformerExpert::new(42, Some(params.clone())))),
        ModelType::PatchTST => Ok(Box::new(PatchTSTExpert::new(42, Some(params.clone())))),
        ModelType::TimesNet => Ok(Box::new(TimesNetExpert::new(42, Some(params.clone())))),
        other => anyhow::bail!(
            "model `{}` ({:?}) does not support the shared ExpertModel training path",
            config.name,
            other
        ),
    }
}

fn generate_hpo_candidate_params(
    config: &ModelConfig,
    base_params: &HashMap<String, String>,
    trial_idx: usize,
    trials: usize,
    backend: &str,
) -> HashMap<String, String> {
    let mut params = base_params.clone();
    let canonical = canonical_model_name(&config.name);

    match canonical {
        "lightgbm" => {
            params.insert(
                "learning_rate".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.02, 0.12, trial_idx, trials, backend, 2)
                ),
            );
            params.insert(
                "num_iterations".to_string(),
                sample_usize(&[240, 400, 560, 720, 960], trial_idx, trials, backend, 3).to_string(),
            );
            params.insert(
                "max_depth".to_string(),
                sample_choice(&["4", "6", "8", "10", "12"], trial_idx, trials, backend, 5),
            );
            params.insert(
                "num_leaves".to_string(),
                sample_choice(&["15", "31", "63", "127"], trial_idx, trials, backend, 7),
            );
            params.insert(
                "min_data_in_leaf".to_string(),
                sample_choice(&["16", "32", "64", "128"], trial_idx, trials, backend, 11),
            );
            params.insert(
                "feature_fraction".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.60, 1.00, trial_idx, trials, backend, 13)
                ),
            );
            params.insert(
                "bagging_fraction".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.60, 1.00, trial_idx, trials, backend, 17)
                ),
            );
            params.insert(
                "lambda_l2".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.0, 8.0, trial_idx, trials, backend, 19)
                ),
            );
        }
        "xgboost" => {
            params.insert(
                "learning_rate".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.02, 0.12, trial_idx, trials, backend, 2)
                ),
            );
            params.insert(
                "n_estimators".to_string(),
                sample_usize(&[320, 480, 640, 800, 960], trial_idx, trials, backend, 3).to_string(),
            );
            params.insert(
                "max_depth".to_string(),
                sample_choice(&["4", "6", "8", "10"], trial_idx, trials, backend, 5),
            );
            params.insert(
                "min_child_weight".to_string(),
                sample_choice(&["1", "2", "4", "8"], trial_idx, trials, backend, 7),
            );
            params.insert(
                "subsample".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.60, 1.00, trial_idx, trials, backend, 11)
                ),
            );
            params.insert(
                "colsample_bytree".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.60, 1.00, trial_idx, trials, backend, 13)
                ),
            );
            params.insert(
                "reg_lambda".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.5, 8.0, trial_idx, trials, backend, 17)
                ),
            );
            if canonical_model_name(&config.name) == "xgboost" && config.name.contains("dart") {
                params.insert(
                    "rate_drop".to_string(),
                    format!(
                        "{:.4}",
                        sample_f64(0.05, 0.25, trial_idx, trials, backend, 19)
                    ),
                );
                params.insert(
                    "skip_drop".to_string(),
                    format!(
                        "{:.4}",
                        sample_f64(0.10, 0.70, trial_idx, trials, backend, 23)
                    ),
                );
            }
            if config.name.contains("rf") {
                params.insert(
                    "num_parallel_tree".to_string(),
                    sample_usize(&[32, 64, 96, 128], trial_idx, trials, backend, 29).to_string(),
                );
            }
        }
        "catboost" => {
            params.insert(
                "iterations".to_string(),
                sample_usize(&[320, 480, 640, 800, 960], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "depth".to_string(),
                sample_choice(&["4", "6", "8", "10"], trial_idx, trials, backend, 3),
            );
            params.insert(
                "learning_rate".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.02, 0.12, trial_idx, trials, backend, 5)
                ),
            );
            params.insert(
                "l2_leaf_reg".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(1.0, 10.0, trial_idx, trials, backend, 7)
                ),
            );
            params.insert(
                "random_strength".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.0, 3.0, trial_idx, trials, backend, 11)
                ),
            );
            params.insert(
                "bagging_temperature".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.0, 2.0, trial_idx, trials, backend, 13)
                ),
            );
        }
        "mlp" | "transformer" | "patchtst" | "timesnet" | "nbeats" | "nbeatsx_nf" | "tide"
        | "tide_nf" | "tabnet" | "kan" => {
            params.insert(
                "hidden_dim".to_string(),
                sample_usize(&[128, 192, 256, 384, 512], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "dropout".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.02, 0.25, trial_idx, trials, backend, 3)
                ),
            );
            params.insert(
                "lr".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.0001, 0.0030, trial_idx, trials, backend, 5)
                ),
            );
            params.insert(
                "batch_size".to_string(),
                sample_usize(&[16, 32, 64, 128], trial_idx, trials, backend, 7).to_string(),
            );
            params.insert(
                "max_epochs".to_string(),
                sample_usize(&[48, 72, 96, 128, 160], trial_idx, trials, backend, 11).to_string(),
            );
            if matches!(canonical, "transformer" | "patchtst") {
                params.insert(
                    "n_heads".to_string(),
                    sample_choice(&["4", "6", "8", "12"], trial_idx, trials, backend, 13),
                );
                params.insert(
                    "n_layers".to_string(),
                    sample_choice(&["2", "3", "4", "6"], trial_idx, trials, backend, 17),
                );
            }
            if matches!(canonical, "nbeats" | "nbeatsx_nf") {
                params.insert(
                    "n_blocks".to_string(),
                    sample_choice(&["2", "3", "4", "5"], trial_idx, trials, backend, 19),
                );
            }
            if canonical == "tabnet" {
                params.insert(
                    "n_steps".to_string(),
                    sample_choice(&["3", "4", "5", "6"], trial_idx, trials, backend, 23),
                );
                params.insert(
                    "relaxation_factor".to_string(),
                    format!(
                        "{:.4}",
                        sample_f64(1.2, 2.0, trial_idx, trials, backend, 29)
                    ),
                );
            }
            if canonical == "kan" {
                params.insert(
                    "n_layers".to_string(),
                    sample_choice(&["2", "3", "4"], trial_idx, trials, backend, 31),
                );
            }
        }
        "elasticnet" => {
            params.insert(
                "alpha".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.0005, 0.5, trial_idx, trials, backend, 2)
                ),
            );
            params.insert(
                "l1_ratio".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.05, 0.95, trial_idx, trials, backend, 3)
                ),
            );
            params.insert(
                "lr".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.001, 0.05, trial_idx, trials, backend, 5)
                ),
            );
            params.insert(
                "epochs".to_string(),
                sample_usize(&[200, 400, 800, 1200], trial_idx, trials, backend, 7).to_string(),
            );
        }
        "online_pa" => {
            params.insert(
                "c".to_string(),
                format!("{:.5}", sample_f64(0.1, 5.0, trial_idx, trials, backend, 2)),
            );
            params.insert(
                "epochs".to_string(),
                sample_usize(&[2, 4, 6, 8, 12], trial_idx, trials, backend, 3).to_string(),
            );
        }
        "online_hoeffding" => {
            params.insert(
                "learning_rate".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.005, 0.20, trial_idx, trials, backend, 2)
                ),
            );
            params.insert(
                "max_depth".to_string(),
                sample_choice(&["3", "5", "7", "9"], trial_idx, trials, backend, 3),
            );
            params.insert(
                "n_bins".to_string(),
                sample_choice(&["16", "32", "48", "64"], trial_idx, trials, backend, 5),
            );
            params.insert(
                "grace_period".to_string(),
                sample_choice(&["16", "32", "48", "64"], trial_idx, trials, backend, 7),
            );
            params.insert(
                "feature_subsample_rate".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.50, 1.00, trial_idx, trials, backend, 11)
                ),
            );
        }
        "isolation_forest" => {
            params.insert(
                "n_trees".to_string(),
                sample_usize(&[64, 96, 128, 192, 256], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "sample_size".to_string(),
                sample_usize(&[128, 192, 256, 384, 512], trial_idx, trials, backend, 3).to_string(),
            );
            params.insert(
                "extension_level".to_string(),
                sample_choice(&["0", "1", "2"], trial_idx, trials, backend, 5),
            );
            params.insert(
                "max_tree_depth".to_string(),
                sample_choice(&["8", "12", "16", "24"], trial_idx, trials, backend, 7),
            );
        }
        "neuro_evo" => {
            params.insert(
                "hidden_dim".to_string(),
                sample_usize(&[16, 32, 64, 96, 128], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "sigma".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.05, 0.45, trial_idx, trials, backend, 3)
                ),
            );
            params.insert(
                "generations".to_string(),
                sample_usize(&[16, 24, 32, 48, 64], trial_idx, trials, backend, 5).to_string(),
            );
        }
        "neat" => {
            params.insert(
                "population".to_string(),
                sample_usize(&[48, 64, 96, 128, 160], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "generations".to_string(),
                sample_usize(&[24, 36, 48, 64, 96], trial_idx, trials, backend, 3).to_string(),
            );
            params.insert(
                "mutation_rate".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.35, 1.05, trial_idx, trials, backend, 5)
                ),
            );
            params.insert(
                "compatibility_threshold".to_string(),
                format!("{:.5}", sample_f64(1.5, 4.0, trial_idx, trials, backend, 7)),
            );
        }
        _ => {}
    }

    params
}

fn select_hpo_dataset(
    payload: &TrainingPayload,
    max_rows: usize,
) -> Result<(DataFrame, Vec<i32>, Option<usize>)> {
    if payload.frame.height() != payload.labels.len() {
        anyhow::bail!(
            "HPO payload mismatch: {} rows vs {} labels",
            payload.frame.height(),
            payload.labels.len()
        );
    }
    if max_rows == 0 || payload.frame.height() <= max_rows {
        return Ok((
            payload.frame.as_ref().clone(),
            payload.labels.as_ref().clone(),
            None,
        ));
    }

    let start = payload.frame.height().saturating_sub(max_rows);
    let frame = payload.frame.slice(start as i64, max_rows);
    let labels = payload
        .labels
        .iter()
        .skip(start)
        .take(max_rows)
        .copied()
        .collect::<Vec<_>>();
    Ok((frame, labels, Some(max_rows)))
}

fn optimize_model_config(
    config: &ModelConfig,
    payload: &TrainingPayload,
    base_tf: &str,
    row_budget_applied: Option<usize>,
) -> Result<(HashMap<String, String>, OptimizationReport)> {
    let backend = hpo_backend_from_params(&config.params);
    let trials_requested = if supports_hpo(config.model_type) {
        hpo_trials_from_params(&config.params)
    } else {
        1
    };
    let base_params = model_params_only(&config.params);
    let holdout_pct = holdout_pct_from_params(&config.params);
    let confidence_threshold = confidence_threshold_from_params(&config.params);
    let metric_weight = metric_weight_from_params(&config.params);
    let accuracy_weight = accuracy_weight_from_params(&config.params);
    let embargo_rows =
        embargo_rows_for_timeframe(base_tf, embargo_minutes_from_params(&config.params));
    let (hpo_frame, hpo_labels, hpo_rows_applied) =
        select_hpo_dataset(payload, hpo_max_rows_from_params(&config.params))?;

    let Some((train_frame, train_labels, val_frame, val_labels)) =
        time_series_holdout_split(&hpo_frame, &hpo_labels, holdout_pct, embargo_rows, 256, 64)?
    else {
        let report = OptimizationReport {
            model_name: config.name.clone(),
            capability_family: config.capability_family,
            capability_state: config.capability_state,
            backend,
            trials_requested,
            trials_completed: 0,
            holdout_pct,
            train_rows: hpo_frame.height(),
            val_rows: 0,
            selected_trial_index: 0,
            selected_params: base_params.clone(),
            selected_metrics: None,
            row_budget_applied,
            hpo_rows_applied,
            notes: vec!["dataset too small for holdout-driven HPO; using base params".to_string()],
            trials: vec![],
        };
        return Ok((base_params, report));
    };

    let mut best_params = base_params.clone();
    let mut best_metrics = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_trial_index = 0_usize;
    let mut trials = Vec::new();

    for trial_idx in 0..trials_requested {
        let candidate_params = if trial_idx == 0 {
            base_params.clone()
        } else {
            generate_hpo_candidate_params(
                config,
                &base_params,
                trial_idx,
                trials_requested,
                &backend,
            )
        };

        let mut model = match build_expert_model(config, payload.frame.width(), &candidate_params) {
            Ok(model) => model,
            Err(error) => {
                trials.push(OptimizationTrialRecord {
                    index: trial_idx,
                    backend: backend.clone(),
                    params: candidate_params,
                    metrics: None,
                    error: Some(error.to_string()),
                    selected: false,
                });
                continue;
            }
        };

        let train_labels_series = labels_to_series(&train_labels);
        match model
            .fit(&train_frame, &train_labels_series)
            .and_then(|_| model.predict_proba(&val_frame))
        {
            Ok(probabilities) => {
                let metrics = evaluate_prediction_quality(
                    &probabilities,
                    &val_labels,
                    confidence_threshold,
                    metric_weight,
                    accuracy_weight,
                )?;
                let selected = metrics.objective_score > best_score;
                if selected {
                    best_score = metrics.objective_score;
                    best_metrics = Some(metrics.clone());
                    best_params = candidate_params.clone();
                    best_trial_index = trial_idx;
                }
                trials.push(OptimizationTrialRecord {
                    index: trial_idx,
                    backend: backend.clone(),
                    params: candidate_params,
                    metrics: Some(metrics),
                    error: None,
                    selected,
                });
            }
            Err(error) => {
                trials.push(OptimizationTrialRecord {
                    index: trial_idx,
                    backend: backend.clone(),
                    params: candidate_params,
                    metrics: None,
                    error: Some(error.to_string()),
                    selected: false,
                });
            }
        }
    }

    if let Some(selected_trial) = trials.get_mut(best_trial_index) {
        selected_trial.selected = true;
    }

    let trials_completed = trials
        .iter()
        .filter(|trial| trial.metrics.is_some())
        .count();
    let notes = if trials_completed == 0 {
        vec!["all HPO trials failed; falling back to base params".to_string()]
    } else {
        Vec::new()
    };

    let report = OptimizationReport {
        model_name: config.name.clone(),
        capability_family: config.capability_family,
        capability_state: config.capability_state,
        backend,
        trials_requested,
        trials_completed,
        holdout_pct,
        train_rows: train_frame.height(),
        val_rows: val_frame.height(),
        selected_trial_index: best_trial_index,
        selected_params: best_params.clone(),
        selected_metrics: best_metrics,
        row_budget_applied,
        hpo_rows_applied,
        notes,
        trials,
    };

    Ok((best_params, report))
}

fn write_onnx_status_sidecar(
    artifact_dir: &std::path::Path,
    config: &ModelConfig,
    payload: &TrainingPayload,
) -> Result<()> {
    if !export_onnx_requested(&config.params) {
        return Ok(());
    }

    let exporter = if cfg!(feature = "python-onnx-export") {
        "python-onnx-export"
    } else {
        "none"
    };
    let reason = if cfg!(feature = "python-onnx-export") {
        "runtime artifact to ONNX export has not been wired for the current Rust training path yet"
    } else {
        "python-onnx-export feature is not enabled for this build"
    };
    let status = OnnxExportStatus::skipped(
        config.name.clone(),
        config.capability_family,
        config.capability_state,
        exporter,
        artifact_dir.to_path_buf(),
        payload.frame.width(),
        payload.frame.height().min(512),
        reason,
    );
    write_onnx_export_status(&artifact_dir.join(ONNX_EXPORT_STATUS_FILE_NAME), &status)
}

fn train_model_dispatch(
    models_dir: &std::path::Path,
    settings: &forex_core::Settings,
    symbol: &str,
    base_tf: &str,
    row_budget_applied: Option<usize>,
    config: &ModelConfig,
    payload: &TrainingPayload,
) -> Result<()> {
    let artifact_dir = model_artifact_dir(models_dir, symbol, base_tf, &config.name);

    if uses_shared_expert_dispatch(config.model_type) {
        let (selected_params, optimization_report) =
            optimize_model_config(config, payload, base_tf, row_budget_applied)?;
        let labels = labels_to_series(payload.labels.as_ref());
        let mut model = build_expert_model(config, payload.frame.width(), &selected_params)?;
        model.fit(payload.frame.as_ref(), &labels)?;
        model.save(&artifact_dir)?;
        write_optimization_report(
            &artifact_dir.join(OPTIMIZATION_REPORT_FILE_NAME),
            &optimization_report,
        )?;
        write_onnx_status_sidecar(&artifact_dir, config, payload)?;
        write_training_profile_sidecar(
            &artifact_dir,
            settings,
            config,
            symbol,
            base_tf,
            payload,
            row_budget_applied,
        )?;
        return Ok(());
    }

    let labels = labels_to_series(payload.labels.as_ref());

    match config.model_type {
        ModelType::SklearsTree => {
            let mut model = SklearsTreeExpert::new();
            model.fit(payload.frame.as_ref(), &labels)?;
            model.save(&artifact_dir)?;
            write_onnx_status_sidecar(&artifact_dir, config, payload)?;
            write_training_profile_sidecar(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
            )?;
            Ok(())
        }
        ModelType::ExitAgent => {
            let mut model = ExitAgent::with_hidden_dim(
                payload.frame.width().max(1),
                parse_usize_param(&config.params, "hidden_dim", 64),
            )
            .with_gamma(parse_f32_param(&config.params, "gamma", 0.99))
            .with_epsilon(parse_f32_param(&config.params, "epsilon", 0.20))
            .with_exploration_schedule(
                parse_f32_param(&config.params, "epsilon_min", 0.05),
                parse_f32_param(&config.params, "epsilon_decay", 0.999),
            )
            .with_memory_capacity(parse_usize_param(&config.params, "memory_capacity", 10_000))
            .with_reward_horizon(parse_usize_param(&config.params, "reward_horizon", 0))
            .with_warmup_steps(parse_usize_param(&config.params, "warmup_steps", 0));
            model.fit_from_frame(payload.frame.as_ref(), &labels)?;
            model.save(&artifact_dir)?;
            write_onnx_status_sidecar(&artifact_dir, config, payload)?;
            write_training_profile_sidecar(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
            )?;
            Ok(())
        }
        ModelType::Dqn => {
            let mut model = TradingReinforcementLearner::new()
                .with_state_bins(parse_usize_param(&config.params, "state_bins", 255) as u16)
                .with_encoding_name(
                    parse_string_param(&config.params, "state_encoding")
                        .as_deref()
                        .unwrap_or("normalized"),
                )
                .with_train_schedule(
                    parse_usize_param(&config.params, "epochs", 48),
                    parse_usize_param(&config.params, "max_steps", 512),
                    parse_usize_param(&config.params, "batch_size", 64),
                )
                .with_update_schedule(
                    parse_usize_param(&config.params, "update_interval", 32),
                    parse_usize_param(&config.params, "update_freq", 4),
                )
                .with_optimizer(
                    parse_f64_param(&config.params, "learning_rate", 1e-3),
                    parse_f32_param(&config.params, "gamma", 0.99),
                )
                .with_exploration_schedule(
                    parse_f32_param(&config.params, "epsilon_start", 1.0),
                    parse_f32_param(&config.params, "epsilon_end", 0.02),
                    parse_f32_param(&config.params, "epsilon_decay", 0.995),
                )
                .with_buffer_capacity(parse_usize_param(&config.params, "buffer_capacity", 50_000))
                .with_runtime_hints(
                    parse_string_param(&config.params, "backend")
                        .unwrap_or_else(|| "rlkit".to_string()),
                    parse_string_param(&config.params, "device")
                        .unwrap_or_else(|| "auto".to_string()),
                    parse_usize_param(&config.params, "parallel_envs", 1),
                    parse_usize_param(&config.params, "eval_episodes", 8),
                    parse_usize_param(&config.params, "rllib_num_workers", 0),
                    parse_usize_param(&config.params, "ray_tune_max_concurrency", 1),
                )
                .with_episode_layout(
                    parse_usize_param(&config.params, "reward_horizon", 0),
                    parse_usize_param(&config.params, "episode_len", 0),
                );
            if let Some(hidden_dims) = config.params.get("hidden_dims") {
                let parsed = hidden_dims
                    .split(',')
                    .filter_map(|value| value.trim().parse::<usize>().ok())
                    .filter(|value| *value > 0)
                    .collect::<Vec<_>>();
                if !parsed.is_empty() {
                    model = model.with_hidden_dims(parsed);
                }
            }
            model.train_on_frame(payload.frame.as_ref(), &labels)?;
            model.save(&artifact_dir)?;
            write_onnx_status_sidecar(&artifact_dir, config, payload)?;
            write_training_profile_sidecar(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
            )?;
            Ok(())
        }
        ModelType::SwarmForecaster => {
            let mut model =
                SwarmForecaster::new(parse_f64_param(&config.params, "memory_limit_mb", 256.0));
            if let Some(horizon) = config
                .params
                .get("horizon")
                .and_then(|value| value.parse::<usize>().ok())
            {
                model.config.horizon = horizon.max(1);
            }
            if let Some(frequency) = parse_string_param(&config.params, "frequency") {
                model.config.frequency = frequency;
            }
            if let Some(strategy) = parse_string_param(&config.params, "strategy") {
                model.config.strategy = match strategy.trim().to_ascii_lowercase().as_str() {
                    "simple" | "simple_average" => {
                        crate::forecasting::swarm_impl::SwarmEnsembleStrategy::SimpleAverage
                    }
                    "weighted" | "weighted_average" => {
                        crate::forecasting::swarm_impl::SwarmEnsembleStrategy::WeightedAverage
                    }
                    "median" => crate::forecasting::swarm_impl::SwarmEnsembleStrategy::Median,
                    "trimmed" | "trimmed_mean" => {
                        crate::forecasting::swarm_impl::SwarmEnsembleStrategy::TrimmedMean
                    }
                    _ => {
                        crate::forecasting::swarm_impl::SwarmEnsembleStrategy::BayesianModelAveraging
                    }
                };
            }
            model.config.accuracy_target = parse_f32_param(
                &config.params,
                "accuracy_target",
                model.config.accuracy_target,
            );
            model.config.latency_requirement_ms = parse_f32_param(
                &config.params,
                "latency_ms",
                model.config.latency_requirement_ms,
            );
            model.config.online_learning = parse_bool_param(
                &config.params,
                "online_learning",
                model.config.online_learning,
            );
            model.config.interpretability_needed = parse_bool_param(
                &config.params,
                "interpretability_needed",
                model.config.interpretability_needed,
            );
            model.fit_from_frame(payload.frame.as_ref(), &labels, symbol)?;
            model.save(&artifact_dir)?;
            write_onnx_status_sidecar(&artifact_dir, config, payload)?;
            write_training_profile_sidecar(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
            )?;
            Ok(())
        }
        ModelType::Genetic => {
            let mut model = GeneticStrategyExpert::new(
                parse_usize_param(&config.params, "population", 64),
                parse_usize_param(&config.params, "generations", 12),
                parse_usize_param(&config.params, "max_indicators", 8),
            )?
            .with_portfolio_size(parse_usize_param(&config.params, "portfolio_size", 12))
            .with_history_window(
                parse_usize_param(&config.params, "train_years", 0),
                parse_usize_param(&config.params, "val_years", 0),
            )
            .with_search_policy(
                parse_parent_selection_policy(&config.params),
                parse_survivor_selection_policy(&config.params),
                parse_f64_param(&config.params, "survivor_fraction", 0.10),
                parse_f64_param(&config.params, "immigrant_fraction", 0.18),
                parse_f64_param(&config.params, "selection_temperature", 0.75),
                parse_usize_param(&config.params, "tournament_size", 3),
            );
            model.fit(payload.frame.as_ref(), &labels, None, Some(symbol))?;
            model.save(&artifact_dir)?;
            write_onnx_status_sidecar(&artifact_dir, config, payload)?;
            write_training_profile_sidecar(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
            )?;
            Ok(())
        }
        other => anyhow::bail!(
            "training dispatch contract drift for model `{}` ({:?}); registry/orchestrator mapping is inconsistent",
            config.name,
            other
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::get_model_capability;

    fn orchestrator_with_models(models: &[&str]) -> TrainingOrchestrator {
        let mut settings = forex_core::Settings::default();
        settings.models.ml_models = models.iter().map(|name| (*name).to_string()).collect();
        settings.models.phase5_core_models.clear();
        settings.models.phase5_filter_meta_blender = false;
        settings.models.calibration_enabled = false;
        settings.models.use_rl_agent = false;
        settings.models.use_sac_agent = false;
        settings.models.use_rllib_agent = false;
        settings.models.use_neuroevolution = false;
        settings.risk.conformal_enabled = false;
        TrainingOrchestrator::new(settings, PathBuf::from("models"))
    }

    #[test]
    fn create_dispatch_plan_rejects_empty_model_config() {
        let orchestrator = orchestrator_with_models(&[]);
        let err = orchestrator
            .create_dispatch_plan()
            .expect_err("expected empty-config error");
        assert!(err.to_string().contains("no model names"));
    }

    #[test]
    fn build_dispatch_plan_orders_and_deduplicates_enabled_models() {
        let orchestrator = orchestrator_with_models(&["mlp", "lightgbm", "patchtst", "lightgbm"]);
        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");

        let names: Vec<_> = plan
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(names, vec!["lightgbm", "mlp", "patchtst"]);
    }

    #[test]
    fn default_configured_inventory_resolves_to_capabilities_and_deterministic_plan() {
        let settings = forex_core::Settings::default();
        let orchestrator = TrainingOrchestrator::new(settings.clone(), PathBuf::from("models"));

        for name in &settings.models.ml_models {
            let capability = get_model_capability(name)
                .unwrap_or_else(|| panic!("missing capability for configured model {name}"));
            assert_eq!(capability.name, *name);
        }

        let first_plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build from default settings");
        let second_plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should be deterministic");

        let first_names: Vec<_> = first_plan
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        let second_names: Vec<_> = second_plan
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();

        assert_eq!(first_names, second_names);
        assert_eq!(first_plan.entries.len(), second_plan.entries.len());
        assert!(
            !first_plan.entries.is_empty(),
            "default dispatch plan should not be empty"
        );

        for entry in &first_plan.entries {
            let capability = get_model_capability(&entry.name)
                .unwrap_or_else(|| panic!("missing capability for dispatch entry {}", entry.name));
            assert_eq!(entry.family, capability.family);
            assert_eq!(entry.state, capability.state);
        }
    }

    #[test]
    fn validate_dispatch_plan_accepts_implemented_capabilities() {
        let orchestrator = orchestrator_with_models(&["xgboost", "mlp"]);
        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");

        orchestrator
            .validate_dispatch_plan(&plan)
            .expect("implemented capabilities should be runnable");
    }

    #[test]
    fn build_training_configs_maps_tree_variants_to_base_trainers_with_params() {
        let orchestrator = orchestrator_with_models(&["catboost_alt"]);
        let mut plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");
        for entry in &mut plan.entries {
            entry.state = CapabilityState::Verified;
        }

        let configs = orchestrator
            .build_training_configs(&plan)
            .expect("tree variants should map to concrete base trainers");
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].model_type, ModelType::CatBoost);
        assert_eq!(configs[0].params.get("variant"), Some(&"alt".to_string()));
    }

    #[test]
    fn build_training_configs_maps_configured_models_to_concrete_model_types() {
        let orchestrator =
            orchestrator_with_models(&["lightgbm", "online_pa", "patchtst", "genetic"]);
        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");

        let configs = orchestrator
            .build_training_configs(&plan)
            .expect("configured models should map to concrete model types");
        let pairs = configs
            .iter()
            .map(|config| (config.name.as_str(), config.model_type))
            .collect::<Vec<_>>();

        assert_eq!(
            pairs,
            vec![
                ("genetic", ModelType::Genetic),
                ("lightgbm", ModelType::LightGBM),
                ("online_pa", ModelType::OnlinePassiveAggressive),
                ("patchtst", ModelType::PatchTST),
            ]
        );
    }

    #[test]
    fn build_training_configs_maps_neat_to_concrete_model_type() {
        let orchestrator = orchestrator_with_models(&["neat"]);
        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");

        let configs = orchestrator
            .build_training_configs(&plan)
            .expect("neat should map to a concrete model type");
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "neat");
        assert_eq!(configs[0].model_type, ModelType::Neat);
    }

    #[test]
    fn derive_labels_uses_meta_label_max_hold_bars_when_model_horizon_is_zero() {
        let mut orchestrator = orchestrator_with_models(&["lightgbm"]);
        orchestrator.settings.models.label_use_triple_barrier = false;
        orchestrator.settings.models.label_horizon_bars = 0;
        orchestrator.settings.risk.meta_label_max_hold_bars = 3;
        orchestrator.settings.risk.triple_barrier_max_bars = 1;
        orchestrator.settings.risk.vol_horizon_bars = 1;
        orchestrator.settings.risk.meta_label_min_dist = 0.2;

        let ohlcv = Ohlcv {
            timestamp: Some(vec![0, 1, 2, 3]),
            open: vec![100.0, 100.1, 100.2, 100.3],
            high: vec![100.0, 100.1, 100.2, 100.3],
            low: vec![100.0, 100.1, 100.2, 100.3],
            close: vec![100.0, 100.1, 100.2, 100.3],
            volume: None,
        };

        let labels = orchestrator.derive_labels(&ohlcv);
        assert_eq!(labels[0], 1);
    }

    #[test]
    fn training_runtime_profile_records_capability_metadata_and_effective_label_horizon() {
        let mut orchestrator = orchestrator_with_models(&["lightgbm"]);
        orchestrator.settings.models.label_horizon_bars = 0;
        orchestrator.settings.risk.meta_label_max_hold_bars = 3;
        orchestrator.settings.models.label_use_triple_barrier = false;

        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");
        let configs = orchestrator
            .build_training_configs(&plan)
            .expect("training configs should build");
        let config = &configs[0];
        let payload =
            TrainingPayload::from_dense(ndarray::Array2::<f32>::zeros((4, 1)), vec![0, 0, 0, 0])
                .expect("build payload");

        let profile = training_runtime_profile(
            &orchestrator.settings,
            config,
            "EURUSD",
            "M1",
            &payload,
            Some(4),
            vec!["H1".to_string()],
        );

        assert_eq!(
            profile.capability_family,
            crate::runtime::capabilities::ModelFamily::Tree
        );
        assert_eq!(profile.capability_state, CapabilityState::Implemented);
        assert_eq!(profile.label_horizon_bars, 0);
        assert_eq!(profile.effective_label_horizon_bars, 3);
        assert_eq!(profile.meta_label_max_hold_bars, 3);
        assert!(!profile.label_use_triple_barrier);
    }
}
