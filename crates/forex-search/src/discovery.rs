use crate::artifact_io::{stable_json_hash, write_json_atomic};
use crate::eval::{BacktestMetrics, fast_evaluate_strategy_core, simulate_trades_core};
use crate::genetic::strategy_gene::EvaluationConfig;
use crate::genetic::{
    Gene, evolve_search_with_progress_and_limits, month_day_indices, signals_for_gene_full,
};
use crate::quality::{StrategyMetrics, StrategyQualityAnalyzer, Trade};
use crate::validation::{
    CanonicalBacktestArtifactFile, CanonicalBacktestScope, CombinatorialPurgedCV, ForwardTestInput,
    ForwardTestValidationArtifactFile, ForwardTestValidationScope, PropFirmRiskInput,
    PropFirmRiskRules, PropFirmRiskValidationArtifactFile, PropFirmRiskValidationScope,
    WalkforwardBacktestInput, WalkforwardSummary, WalkforwardValidationArtifactFile,
    WalkforwardValidationScope, compute_forward_test_summary, compute_prop_firm_risk_summary,
    embargoed_walkforward_backtest, write_canonical_backtest_artifact_atomic,
    write_forward_test_validation_artifact_atomic, write_prop_firm_risk_validation_artifact_atomic,
    write_walkforward_validation_artifact_atomic,
};
use anyhow::{Context, Result};
use chrono::{Datelike, TimeZone, Utc};
use forex_core::contracts::{
    DeterminismPolicy, LiveValidationEvidence, TemporalFeatureContract, ValidationEvidenceManifest,
};
use forex_data::{FeatureFrame, Ohlcv};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Typed runtime knobs that previously lived only in `FOREX_BOT_*` env vars.
///
/// These values change *production* discovery semantics (which features are
/// kept, how much data the stage-1 funnel sees, what counts as in-sample for
/// the prefilter), so they belong in typed config rather than ambient env
/// state. Callers that still want to honour the env vars must opt in via
/// [`DiscoveryRuntimeOverrides::from_env`] or
/// [`DiscoveryConfig::with_env_runtime_overrides`] — the discovery cycle
/// itself no longer reads the environment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscoveryRuntimeOverrides {
    /// Maximum number of features to keep after the in-sample correlation
    /// prefilter. `0` disables the prefilter entirely.
    pub prefilter_top_k: usize,
    /// Fraction of rows treated as in-sample when ranking features. Must be
    /// strictly positive and at most `1.0`.
    pub prefilter_insample_frac: f64,
    /// Fraction of recent rows fed to the multi-stage funnel's first stage.
    /// Clamped to `[0.01, 1.0]` at use time.
    pub funnel_stage1_pct: f64,
}

impl Default for DiscoveryRuntimeOverrides {
    fn default() -> Self {
        Self {
            prefilter_top_k: 50,
            prefilter_insample_frac: 0.70,
            funnel_stage1_pct: 0.25,
        }
    }
}

impl DiscoveryRuntimeOverrides {
    /// One-shot read of the legacy `FOREX_BOT_*` env vars. This is the only
    /// place in `forex-search` that consults the environment for these
    /// knobs; production callers should prefer constructing the struct from
    /// typed config.
    pub fn from_env() -> Self {
        let mut overrides = Self::default();
        if let Some(top_k) = std::env::var("FOREX_BOT_PREFILTER_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            overrides.prefilter_top_k = top_k;
        }
        if let Some(insample) = std::env::var("FOREX_BOT_PREFILTER_INSAMPLE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
        {
            overrides.prefilter_insample_frac = insample;
        }
        if let Some(stage1) = std::env::var("FOREX_BOT_FUNNEL_STAGE1_PCT")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite())
        {
            overrides.funnel_stage1_pct = stage1.clamp(0.01, 1.0);
        }
        overrides
    }

    fn resolved_funnel_stage1_pct(&self) -> f64 {
        if self.funnel_stage1_pct.is_finite() {
            self.funnel_stage1_pct.clamp(0.01, 1.0)
        } else {
            0.25
        }
    }

    fn resolved_prefilter_insample_frac(&self) -> f64 {
        if self.prefilter_insample_frac.is_finite()
            && self.prefilter_insample_frac > 0.0
            && self.prefilter_insample_frac <= 1.0
        {
            self.prefilter_insample_frac
        } else {
            0.70
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub timeframe_label: String,
    pub evaluation_symbol: String,
    pub evaluation_account_currency: String,
    pub evaluation_spread_pips: f64,
    pub evaluation_commission_per_trade: f64,
    pub population: usize,
    pub generations: usize,
    pub max_indicators: usize,
    pub candidate_count: usize,
    pub portfolio_size: usize,
    pub max_rows: usize,
    pub max_rows_by_timeframe: HashMap<String, usize>,
    pub max_hours: f64,
    pub corr_threshold: f64,
    pub min_trades_per_day: f64,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    pub enable_cpcv: bool,
    pub cpcv_n_splits: usize,
    pub cpcv_n_test_groups: usize,
    pub cpcv_embargo_pct: f64,
    pub cpcv_purge_pct: f64,
    pub cpcv_min_phi: f64,
    pub cpcv_max_rows: usize,
    pub filtering: crate::genetic::FilteringConfig,
    /// Starting account balance used for PnL%, DD%, and regime loss limits.
    pub initial_balance: f64,
    /// Reject a gene if any regime-specific PnL drops below
    /// `-initial_balance * max_regime_loss_pct / 100`.
    pub max_regime_loss_pct: f64,
    /// Higher timeframes to include in multitimeframe feature preparation.
    pub higher_timeframes: Vec<String>,
    /// Typed replacements for the legacy `FOREX_BOT_PREFILTER_*` /
    /// `FOREX_BOT_FUNNEL_STAGE1_PCT` env vars.
    pub runtime_overrides: DiscoveryRuntimeOverrides,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            timeframe_label: "M1".to_string(),
            evaluation_symbol: "EURUSD".to_string(),
            evaluation_account_currency: "USD".to_string(),
            evaluation_spread_pips: 1.5,
            evaluation_commission_per_trade: 0.0,
            population: 1000,
            generations: 10,
            max_indicators: 5,
            candidate_count: 5000,
            portfolio_size: 2000,
            max_rows: 0,
            max_rows_by_timeframe: HashMap::new(),
            max_hours: 0.0,
            corr_threshold: 0.85,
            min_trades_per_day: 0.2,
            walkforward_splits: 20,
            embargo_minutes: 120,
            enable_cpcv: true,
            cpcv_n_splits: 5,
            cpcv_n_test_groups: 2,
            cpcv_embargo_pct: 0.01,
            cpcv_purge_pct: 0.02,
            cpcv_min_phi: 0.80,
            cpcv_max_rows: 0,
            filtering: crate::genetic::FilteringConfig::default(),
            initial_balance: 100_000.0,
            max_regime_loss_pct: 3.0,
            higher_timeframes: Vec::new(),
            runtime_overrides: DiscoveryRuntimeOverrides::default(),
        }
    }
}

impl DiscoveryConfig {
    pub fn from_settings(settings: &forex_core::Settings) -> Self {
        let model_settings = &settings.models;
        let filtering = crate::genetic::FilteringConfig {
            min_trades: model_settings.prop_min_trades.max(1) as f64,
            anomaly_guard: true,
            min_positive_months: model_settings.prop_search_val_min_positive_months,
            min_trades_per_month: model_settings.prop_search_val_min_trades_per_month as f64,
            min_monthly_return_pct: model_settings.prop_search_val_min_monthly_profit_pct / 100.0,
            log_trades: model_settings.prop_search_val_log_trades,
            trade_log_max: model_settings.prop_search_val_trade_log_max.max(1),
            opportunistic_enabled: model_settings.prop_search_opportunistic_enabled,
            use_opportunistic_candidates: model_settings.prop_search_use_opportunistic,
            opportunistic_min_positive_months: model_settings
                .prop_search_opportunistic_min_positive_months,
            opportunistic_min_trades_per_month: model_settings
                .prop_search_opportunistic_min_trades_per_month
                as f64,
            opportunistic_min_trade_return_pct: model_settings
                .prop_search_opportunistic_min_trade_return_pct,
            opportunistic_max_dd: model_settings.prop_search_opportunistic_max_dd.max(0.0),
            ..Default::default()
        };

        let candidate_count = if model_settings.prop_search_val_candidates == 0 {
            model_settings.prop_search_population.max(50)
        } else {
            model_settings.prop_search_val_candidates.max(1)
        };

        Self {
            timeframe_label: settings.system.base_timeframe.clone(),
            evaluation_symbol: settings.system.symbol.clone(),
            evaluation_account_currency: "USD".to_string(),
            evaluation_spread_pips: settings.risk.backtest_spread_pips.max(0.0),
            evaluation_commission_per_trade: settings.risk.commission_per_lot.max(0.0),
            population: model_settings.prop_search_population.max(10),
            generations: model_settings.prop_search_generations.max(1),
            max_indicators: if model_settings.prop_search_max_indicators == 0 {
                5
            } else {
                model_settings.prop_search_max_indicators.max(1)
            },
            candidate_count,
            portfolio_size: model_settings.prop_search_portfolio_size.max(1),
            max_rows: model_settings.prop_search_max_rows,
            max_rows_by_timeframe: model_settings.prop_search_max_rows_by_tf.clone(),
            max_hours: model_settings.prop_search_max_hours.max(0.0),
            corr_threshold: 0.85,
            min_trades_per_day: model_settings.prop_search_val_min_trades_per_day.max(0.2),
            walkforward_splits: model_settings.walkforward_splits.max(2),
            embargo_minutes: model_settings.embargo_minutes,
            enable_cpcv: model_settings.enable_cpcv,
            cpcv_n_splits: model_settings.cpcv_n_splits.max(2),
            cpcv_n_test_groups: model_settings.cpcv_n_test_groups.max(1),
            cpcv_embargo_pct: model_settings.cpcv_embargo_pct.max(0.0),
            cpcv_purge_pct: model_settings.cpcv_purge_pct.max(0.0),
            cpcv_min_phi: model_settings.cpcv_min_phi.max(0.0),
            cpcv_max_rows: model_settings.cpcv_max_rows,
            filtering,
            initial_balance: settings.risk.initial_balance.max(1.0),
            max_regime_loss_pct: 3.0,
            higher_timeframes: settings.system.higher_timeframes.clone(),
            runtime_overrides: DiscoveryRuntimeOverrides::default(),
        }
    }

    /// Opt-in helper that resolves legacy `FOREX_BOT_*` discovery env vars
    /// into the typed `runtime_overrides` field. Production callers should
    /// prefer setting `runtime_overrides` explicitly.
    pub fn with_env_runtime_overrides(mut self) -> Self {
        self.runtime_overrides = DiscoveryRuntimeOverrides::from_env();
        self
    }

    pub fn evaluation_config(&self, price_hint: Option<f64>) -> EvaluationConfig {
        EvaluationConfig::for_symbol(
            &self.evaluation_symbol,
            &self.evaluation_account_currency,
            price_hint,
            Some(self.evaluation_spread_pips),
            Some(self.evaluation_commission_per_trade),
        )
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    pub portfolio: Vec<Gene>,
    pub candidates: Vec<Gene>,
    pub quality_metrics: Vec<StrategyMetrics>,
    pub logged_trades: Vec<LoggedStrategyTrades>,
    /// Feature names as they existed *after* prefiltering inside discovery.
    /// Gene indices refer to columns in this list, not the caller's original names.
    pub effective_feature_names: Vec<String>,
    pub validation_gates: DiscoveryValidationGates,
    pub canonical_backtest_artifacts: Vec<CanonicalBacktestArtifactFile>,
    pub walkforward_validation_artifacts: Vec<WalkforwardValidationArtifactFile>,
    /// Forward-test artifacts produced by replaying the portfolio on a
    /// held-out tail. Empty until the caller invokes
    /// [`compute_discovery_forward_test_artifacts`] with a tail dataset.
    pub forward_test_validation_artifacts: Vec<ForwardTestValidationArtifactFile>,
    /// Prop-firm risk validation artifacts produced by replaying the
    /// portfolio on a held-out tail and applying typed
    /// [`PropFirmRiskRules`]. Empty until the caller invokes
    /// [`compute_discovery_prop_firm_artifacts`] with a tail dataset and
    /// a rule set.
    pub prop_firm_validation_artifacts: Vec<PropFirmRiskValidationArtifactFile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoggedStrategyTrades {
    pub strategy_id: String,
    pub opportunistic: bool,
    pub trades: Vec<Trade>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryFilterProfile {
    pub max_dd: f64,
    pub min_profit: f64,
    pub min_trades: f64,
    pub min_sharpe: f64,
    pub min_win_rate: f64,
    pub min_profit_factor: f64,
    pub min_positive_months: usize,
    pub min_trades_per_month: f64,
    pub min_monthly_return_pct: f64,
    pub opportunistic_enabled: bool,
    pub opportunistic_min_positive_months: usize,
    pub opportunistic_min_trades_per_month: f64,
    pub opportunistic_min_trade_return_pct: f64,
    pub opportunistic_max_dd: f64,
    pub log_trades: bool,
    pub trade_log_max: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryValidationGates {
    pub walkforward_passed: bool,
    pub cpcv_passed: bool,
    pub canonical_backtest_artifacts: usize,
    pub walkforward_validation_artifacts: usize,
    pub cpcv_fold_count: usize,
    pub cpcv_profitable_fold_ratio: f64,
    pub temporal_contract_hash: Option<String>,
}

impl DiscoveryValidationGates {
    pub fn pending() -> Self {
        Self {
            walkforward_passed: false,
            cpcv_passed: false,
            canonical_backtest_artifacts: 0,
            walkforward_validation_artifacts: 0,
            cpcv_fold_count: 0,
            cpcv_profitable_fold_ratio: 0.0,
            temporal_contract_hash: None,
        }
    }

    pub fn is_portfolio_export_ready(&self) -> bool {
        self.walkforward_passed && self.cpcv_passed
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryRunProfile {
    pub timeframe_label: String,
    pub population: usize,
    pub generations: usize,
    pub max_indicators: usize,
    pub candidate_count_target: usize,
    pub portfolio_size_target: usize,
    pub max_rows: usize,
    pub max_runtime_hours: f64,
    pub corr_threshold: f64,
    pub min_trades_per_day: f64,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    pub enable_cpcv: bool,
    pub cpcv_n_splits: usize,
    pub cpcv_n_test_groups: usize,
    pub cpcv_embargo_pct: f64,
    pub cpcv_purge_pct: f64,
    pub cpcv_min_phi: f64,
    pub filters: DiscoveryFilterProfile,
    pub candidates_observed: usize,
    pub portfolio_observed: usize,
    pub quality_metrics_observed: usize,
    pub logged_trade_sets: usize,
    pub walkforward_passed: bool,
    pub cpcv_passed: bool,
    pub canonical_backtest_artifacts_observed: usize,
    pub walkforward_validation_artifacts_observed: usize,
    pub forward_test_validation_artifacts_observed: usize,
    pub prop_firm_validation_artifacts_observed: usize,
    pub cpcv_fold_count: usize,
    pub cpcv_profitable_fold_ratio: f64,
    pub validation_temporal_contract_hash: Option<String>,
    pub prefilter_top_k: usize,
    pub prefilter_insample_frac: f64,
    pub funnel_stage1_pct: f64,
    /// Per-kind validation-evidence hashes ready for the typed
    /// [`forex_core::contracts::ValidationEvidenceManifest`]. `None`
    /// per field indicates that artifact kind was not produced for
    /// this run.
    pub validation_evidence_hashes: DiscoveryPerKindEvidenceHashes,
    pub validation_evidence_complete: bool,
    pub validation_evidence_missing_kinds: Vec<String>,
    /// Resolved determinism policy under which the genetic search ran.
    /// `Deterministic { seed }` means the run is reproducible; the two
    /// non-deterministic variants surface in the persisted profile so
    /// `LivePromotionGate::PromotionRejectedDeterminism` failures can
    /// be diagnosed without re-running.
    pub determinism_policy: DeterminismPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiscoveryProgress {
    SearchStarted {
        population: usize,
        generations: usize,
        max_indicators: usize,
    },
    GenerationCompleted {
        generation: usize,
        total_generations: usize,
        best_fitness: f64,
        stagnant_generations: usize,
        archived_profitable: usize,
    },
    CandidatesRanked {
        candidate_count: usize,
        truncated_to: usize,
    },
    CandidatesFiltered {
        passed_filters: usize,
        evaluated_candidates: usize,
        min_trades_required: usize,
    },
    QualityScreened {
        strict_passed: usize,
        opportunistic_passed: usize,
        evaluated_candidates: usize,
        logged_trade_sets: usize,
    },
    PortfolioSelected {
        portfolio_size: usize,
        rejected_by_correlation: usize,
        target_portfolio: usize,
    },
    Completed {
        candidate_count: usize,
        filtered_count: usize,
        portfolio_size: usize,
    },
}

pub fn ensure_non_empty_portfolio(result: &DiscoveryResult, context: &str) -> Result<()> {
    if result.portfolio.is_empty() {
        anyhow::bail!(
            "Discovery produced an empty portfolio for {} (candidates={})",
            context,
            result.candidates.len()
        );
    }
    Ok(())
}

fn row_cap_for_config(config: &DiscoveryConfig) -> usize {
    let tf_cap = config
        .max_rows_by_timeframe
        .get(&config.timeframe_label)
        .copied()
        .unwrap_or(0);
    match (config.max_rows, tf_cap) {
        (0, 0) => 0,
        (0, tf) => tf,
        (global, 0) => global,
        (global, tf) => global.min(tf),
    }
}

fn trim_recent_history(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<(FeatureFrame, Ohlcv, Option<usize>)> {
    let frame_rows = features.data.nrows();
    let ohlcv_rows = ohlcv.close.len();
    let available_rows = frame_rows.min(ohlcv_rows);
    if available_rows == 0 {
        anyhow::bail!("cannot run discovery on empty history");
    }

    let mut start_idx = 0usize;
    let row_cap = row_cap_for_config(config);
    if row_cap > 0 && row_cap < available_rows {
        start_idx = available_rows - row_cap;
    }

    let trimmed_rows = available_rows.saturating_sub(start_idx);
    let row_budget_applied = if start_idx > 0 {
        Some(trimmed_rows)
    } else {
        None
    };

    let trimmed_features = FeatureFrame {
        timestamps: features.timestamps[start_idx..available_rows].to_vec(),
        names: features.names.clone(),
        data: features
            .data
            .slice(ndarray::s![start_idx..available_rows, ..])
            .to_owned(),
    };
    let trimmed_ohlcv = slice_ohlcv(ohlcv, start_idx, available_rows);
    Ok((trimmed_features, trimmed_ohlcv, row_budget_applied))
}

fn slice_ohlcv(ohlcv: &Ohlcv, start_idx: usize, end_idx: usize) -> Ohlcv {
    Ohlcv {
        timestamp: ohlcv
            .timestamp
            .as_ref()
            .map(|ts| ts[start_idx..end_idx].to_vec()),
        open: ohlcv.open[start_idx..end_idx].to_vec(),
        high: ohlcv.high[start_idx..end_idx].to_vec(),
        low: ohlcv.low[start_idx..end_idx].to_vec(),
        close: ohlcv.close[start_idx..end_idx].to_vec(),
        volume: ohlcv
            .volume
            .as_ref()
            .map(|vol| vol[start_idx..end_idx].to_vec()),
    }
}

fn quality_analyzer_for_config(config: &DiscoveryConfig) -> StrategyQualityAnalyzer {
    StrategyQualityAnalyzer {
        min_sharpe: config.filtering.min_sharpe.max(0.0),
        min_sortino: config.filtering.min_sharpe.max(0.0),
        min_calmar: 0.0,
        min_profit_factor: config.filtering.min_profit_factor.max(0.0),
        min_win_rate: config.filtering.min_win_rate.clamp(0.0, 1.0),
        min_trades: config.filtering.min_trades.max(0.0) as usize,
        max_dd_acceptable: config.filtering.max_dd.max(0.0),
        min_monthly_return_pct: config.filtering.min_monthly_return_pct.max(0.0),
        edge_significance_pvalue: 0.05,
    }
}

fn discovery_backtest_settings(
    config: &DiscoveryConfig,
    gene: &Gene,
    price_hint: Option<f64>,
) -> crate::eval::BacktestSettings {
    let evaluation = config.evaluation_config(price_hint);
    crate::eval::BacktestSettings {
        sl_pips: if gene.sl_pips.is_finite() && gene.sl_pips > 0.0 {
            gene.sl_pips
        } else {
            20.0
        },
        tp_pips: if gene.tp_pips.is_finite() && gene.tp_pips > 0.0 {
            gene.tp_pips
        } else {
            40.0
        },
        max_hold_bars: evaluation.max_hold_bars,
        trailing_enabled: evaluation.trailing_enabled,
        trailing_atr_multiplier: evaluation.trailing_atr_multiplier,
        trailing_be_trigger_r: evaluation.trailing_be_trigger_r,
        pip_value: evaluation.pip_value,
        spread_pips: evaluation.spread_pips,
        commission_per_trade: evaluation.commission_per_trade,
        pip_value_per_lot: evaluation.pip_value_per_lot,
        kill_zones_enabled: true,
        ..crate::eval::BacktestSettings::default()
    }
}

fn passes_strict_quality(metrics: &StrategyMetrics, cfg: &crate::genetic::FilteringConfig) -> bool {
    if cfg.min_positive_months > 0 && metrics.positive_months < cfg.min_positive_months {
        return false;
    }
    if cfg.min_trades_per_month > 0.0 && metrics.trades_per_month < cfg.min_trades_per_month {
        return false;
    }
    if cfg.min_monthly_return_pct > 0.0
        && metrics.avg_monthly_return_pct < cfg.min_monthly_return_pct
    {
        return false;
    }
    true
}

fn passes_opportunistic_quality(
    metrics: &StrategyMetrics,
    cfg: &crate::genetic::FilteringConfig,
) -> bool {
    if !cfg.opportunistic_enabled || !cfg.use_opportunistic_candidates {
        return false;
    }
    if cfg.opportunistic_min_positive_months > 0
        && metrics.positive_months < cfg.opportunistic_min_positive_months
    {
        return false;
    }
    if cfg.opportunistic_min_trades_per_month > 0.0
        && metrics.trades_per_month < cfg.opportunistic_min_trades_per_month
    {
        return false;
    }
    let avg_trade_return_pct = metrics.avg_win_pct.abs() * 100.0;
    if cfg.opportunistic_min_trade_return_pct > 0.0
        && avg_trade_return_pct < cfg.opportunistic_min_trade_return_pct
    {
        return false;
    }
    if cfg.opportunistic_max_dd > 0.0 && metrics.max_drawdown_pct > cfg.opportunistic_max_dd {
        return false;
    }
    true
}

#[derive(Debug, Serialize)]
struct DiscoveryDatasetFingerprint<'a> {
    row_count: usize,
    first_timestamp: Option<i64>,
    last_timestamp: Option<i64>,
    feature_names: &'a [String],
    close_rows: usize,
    first_close: Option<f64>,
    last_close: Option<f64>,
}

#[derive(Debug, Serialize)]
struct DiscoveryTemporalPolicy<'a> {
    timeframe_label: &'a str,
    higher_timeframes: &'a [String],
    feature_names: &'a [String],
}

#[derive(Debug, Serialize)]
struct DiscoveryWalkforwardPolicy {
    train_ratio: f64,
    walkforward_splits: usize,
    embargo_minutes: usize,
    enable_cpcv: bool,
    cpcv_n_splits: usize,
    cpcv_n_test_groups: usize,
    cpcv_embargo_pct: f64,
    cpcv_purge_pct: f64,
    cpcv_min_phi: f64,
}

#[derive(Debug, Serialize)]
struct DiscoveryLiveReadinessPolicy {
    portfolio_size_target: usize,
    max_regime_loss_pct: f64,
    filtering: crate::genetic::FilteringConfig,
}

#[derive(Debug, Serialize)]
struct DiscoveryBacktestPolicy {
    symbol: String,
    account_currency: String,
    timeframe_label: String,
    sl_pips: f64,
    tp_pips: f64,
    max_hold_bars: usize,
    min_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
    kill_zones_enabled: bool,
}

fn discovery_temporal_contract(
    config: &DiscoveryConfig,
    feature_names: &[String],
) -> Result<TemporalFeatureContract> {
    let feature_policy_hash = stable_json_hash(&DiscoveryTemporalPolicy {
        timeframe_label: &config.timeframe_label,
        higher_timeframes: &config.higher_timeframes,
        feature_names,
    })?;
    let label_policy_hash = stable_json_hash(&(
        "strategy-search-signal-v1",
        "prior-bar-signal-next-bar-fill",
        &config.timeframe_label,
    ))?;
    let walk_forward_policy_hash = stable_json_hash(&DiscoveryWalkforwardPolicy {
        train_ratio: 0.70,
        walkforward_splits: config.walkforward_splits,
        embargo_minutes: config.embargo_minutes,
        enable_cpcv: config.enable_cpcv,
        cpcv_n_splits: config.cpcv_n_splits,
        cpcv_n_test_groups: config.cpcv_n_test_groups,
        cpcv_embargo_pct: config.cpcv_embargo_pct,
        cpcv_purge_pct: config.cpcv_purge_pct,
        cpcv_min_phi: config.cpcv_min_phi,
    })?;
    let live_readiness_policy_hash = stable_json_hash(&DiscoveryLiveReadinessPolicy {
        portfolio_size_target: config.portfolio_size,
        max_regime_loss_pct: config.max_regime_loss_pct,
        filtering: config.filtering,
    })?;

    Ok(TemporalFeatureContract::strict_live(
        "UTC",
        feature_policy_hash,
        label_policy_hash,
        walk_forward_policy_hash,
        live_readiness_policy_hash,
    )?)
}

fn validation_row_count(features: &FeatureFrame, ohlcv: &Ohlcv) -> Result<usize> {
    let n = features.data.nrows();
    if n == 0
        || features.timestamps.len() != n
        || ohlcv.close.len() != n
        || ohlcv.high.len() != n
        || ohlcv.low.len() != n
    {
        anyhow::bail!(
            "discovery validation requires aligned non-empty features/OHLCV rows (features={}, timestamps={}, close={}, high={}, low={})",
            n,
            features.timestamps.len(),
            ohlcv.close.len(),
            ohlcv.high.len(),
            ohlcv.low.len()
        );
    }
    Ok(n)
}

fn discovery_dataset_hash(features: &FeatureFrame, ohlcv: &Ohlcv) -> Result<String> {
    stable_json_hash(&DiscoveryDatasetFingerprint {
        row_count: features.data.nrows(),
        first_timestamp: features.timestamps.first().copied(),
        last_timestamp: features.timestamps.last().copied(),
        feature_names: &features.names,
        close_rows: ohlcv.close.len(),
        first_close: ohlcv.close.first().copied(),
        last_close: ohlcv.close.last().copied(),
    })
}

fn discovery_backtest_policy_hash(
    config: &DiscoveryConfig,
    gene: &Gene,
    settings: &crate::eval::BacktestSettings,
) -> Result<String> {
    stable_json_hash(&DiscoveryBacktestPolicy {
        symbol: config.evaluation_symbol.clone(),
        account_currency: config.evaluation_account_currency.clone(),
        timeframe_label: config.timeframe_label.clone(),
        sl_pips: settings.sl_pips,
        tp_pips: settings.tp_pips,
        max_hold_bars: settings.max_hold_bars,
        min_hold_bars: settings.min_hold_bars,
        trailing_enabled: settings.trailing_enabled,
        trailing_atr_multiplier: settings.trailing_atr_multiplier,
        trailing_be_trigger_r: settings.trailing_be_trigger_r,
        pip_value: settings.pip_value,
        spread_pips: settings.spread_pips,
        commission_per_trade: settings.commission_per_trade,
        pip_value_per_lot: settings.pip_value_per_lot,
        kill_zones_enabled: settings.kill_zones_enabled,
    })
    .with_context(|| format!("hashing backtest policy for {}", gene.strategy_id))
}

fn embargo_bars_from_timestamps(timestamps: &[i64], embargo_minutes: usize) -> usize {
    if embargo_minutes == 0 || timestamps.len() < 2 {
        return 0;
    }
    let step_ms = timestamps
        .windows(2)
        .filter_map(|window| {
            let step = window[1].saturating_sub(window[0]);
            (step > 0).then_some(step)
        })
        .min()
        .unwrap_or(60_000);
    let embargo_ms = (embargo_minutes as i64).saturating_mul(60_000);
    ((embargo_ms + step_ms - 1) / step_ms).max(0) as usize
}

fn walkforward_summary_passed(summary: &WalkforwardSummary) -> bool {
    summary.walk_forward_splits > 0
        && summary.avg_pnl > 0.0
        && !summary.any_daily_loss_breach
        && !summary.any_consistency_violation
        && !summary.any_trade_limit_violation
        && summary.all_min_trading_days_ok
}

fn evaluate_cpcv_gate(
    portfolio: &[Gene],
    portfolio_signals: &[Vec<i8>],
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    months: &[i64],
    days: &[i64],
) -> Result<(bool, usize, f64)> {
    if portfolio.is_empty() {
        return Ok((false, 0, 0.0));
    }
    if !config.enable_cpcv {
        return Ok((true, 0, 1.0));
    }

    let n = ohlcv.close.len();
    let capped_n = if config.cpcv_max_rows > 0 {
        config.cpcv_max_rows.min(n)
    } else {
        n
    };
    let offset = n.saturating_sub(capped_n);
    let cv = CombinatorialPurgedCV::new(
        config.cpcv_n_splits,
        config.cpcv_n_test_groups,
        config.cpcv_embargo_pct,
        config.cpcv_purge_pct,
    );
    let splits = cv.split(capped_n);
    if splits.is_empty() {
        return Ok((false, 0, 0.0));
    }

    let mut fold_count = 0usize;
    let mut profitable_folds = 0usize;
    for (gene, signals) in portfolio.iter().zip(portfolio_signals) {
        let settings = discovery_backtest_settings(config, gene, ohlcv.close.last().copied());
        for (_, test_idx) in &splits {
            if test_idx.is_empty() {
                continue;
            }
            let absolute_idx: Vec<usize> = test_idx.iter().map(|idx| offset + *idx).collect();
            let close: Vec<f64> = absolute_idx.iter().map(|idx| ohlcv.close[*idx]).collect();
            let high: Vec<f64> = absolute_idx.iter().map(|idx| ohlcv.high[*idx]).collect();
            let low: Vec<f64> = absolute_idx.iter().map(|idx| ohlcv.low[*idx]).collect();
            let sig: Vec<i8> = absolute_idx.iter().map(|idx| signals[*idx]).collect();
            let fold_months: Vec<i64> = absolute_idx.iter().map(|idx| months[*idx]).collect();
            let fold_days: Vec<i64> = absolute_idx.iter().map(|idx| days[*idx]).collect();
            let metrics = BacktestMetrics::from_metric_array(fast_evaluate_strategy_core(
                &close,
                &high,
                &low,
                &sig,
                &fold_months,
                &fold_days,
                &[],
                &settings,
            ));
            fold_count += 1;
            let drawdown_ok =
                config.filtering.max_dd <= 0.0 || metrics.max_drawdown <= config.filtering.max_dd;
            if metrics.trade_count > 0 && metrics.net_profit > 0.0 && drawdown_ok {
                profitable_folds += 1;
            }
        }
    }

    if fold_count == 0 {
        return Ok((false, 0, 0.0));
    }
    let ratio = profitable_folds as f64 / fold_count as f64;
    Ok((
        ratio >= config.cpcv_min_phi.clamp(0.0, 1.0),
        fold_count,
        ratio,
    ))
}

fn build_discovery_validation_artifacts(
    portfolio: &[Gene],
    portfolio_signals: &[Vec<i8>],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<(
    DiscoveryValidationGates,
    Vec<CanonicalBacktestArtifactFile>,
    Vec<WalkforwardValidationArtifactFile>,
)> {
    if portfolio.is_empty() {
        return Ok((DiscoveryValidationGates::pending(), Vec::new(), Vec::new()));
    }
    let n = validation_row_count(features, ohlcv)?;
    if portfolio_signals.iter().any(|signals| signals.len() != n) {
        anyhow::bail!("discovery validation requires portfolio signals aligned to feature rows");
    }

    let temporal_contract = discovery_temporal_contract(config, &features.names)?;
    let temporal_contract_hash = temporal_contract.temporal_contract_hash();
    let dataset_hash = discovery_dataset_hash(features, ohlcv)?;
    let (months, days) = month_day_indices(&features.timestamps);
    let timestamps = &features.timestamps[..n];
    let embargo_bars = embargo_bars_from_timestamps(timestamps, config.embargo_minutes);

    let mut canonical_backtest_artifacts = Vec::with_capacity(portfolio.len());
    let mut walkforward_validation_artifacts = Vec::with_capacity(portfolio.len());
    let mut walkforward_passed = true;

    for (gene, signals) in portfolio.iter().zip(portfolio_signals) {
        let settings = discovery_backtest_settings(config, gene, ohlcv.close.last().copied());
        let strategy_hash = stable_json_hash(gene)?;
        let evaluation_config_hash = discovery_backtest_policy_hash(config, gene, &settings)?;
        let metrics = BacktestMetrics::from_metric_array(fast_evaluate_strategy_core(
            &ohlcv.close,
            &ohlcv.high,
            &ohlcv.low,
            signals,
            &months,
            &days,
            timestamps,
            &settings,
        ));
        canonical_backtest_artifacts.push(CanonicalBacktestArtifactFile::new(
            CanonicalBacktestScope::new(
                dataset_hash.clone(),
                evaluation_config_hash.clone(),
                strategy_hash.clone(),
                &temporal_contract,
            ),
            metrics,
        ));

        let walkforward_summary = embargoed_walkforward_backtest(WalkforwardBacktestInput {
            close: &ohlcv.close,
            high: &ohlcv.high,
            low: &ohlcv.low,
            signals,
            months: &months,
            days: &days,
            timestamps,
            train_ratio: 0.70,
            n_splits: config.walkforward_splits.max(1),
            embargo_bars,
            settings: &settings,
            max_daily_loss_pct: config.max_regime_loss_pct,
            max_daily_profit_pct: 0.0,
            min_trading_days: 0,
            max_trades_per_day: 0,
            initial_balance: config.initial_balance,
        })?;
        walkforward_passed &= walkforward_summary_passed(&walkforward_summary);
        walkforward_validation_artifacts.push(WalkforwardValidationArtifactFile::new(
            WalkforwardValidationScope::for_strategy(
                dataset_hash.clone(),
                evaluation_config_hash,
                strategy_hash,
                &temporal_contract,
            ),
            walkforward_summary,
        ));
    }

    let (cpcv_passed, cpcv_fold_count, cpcv_profitable_fold_ratio) =
        evaluate_cpcv_gate(portfolio, portfolio_signals, ohlcv, config, &months, &days)?;

    let validation_gates = DiscoveryValidationGates {
        walkforward_passed,
        cpcv_passed,
        canonical_backtest_artifacts: canonical_backtest_artifacts.len(),
        walkforward_validation_artifacts: walkforward_validation_artifacts.len(),
        cpcv_fold_count,
        cpcv_profitable_fold_ratio,
        temporal_contract_hash: Some(temporal_contract_hash),
    };

    Ok((
        validation_gates,
        canonical_backtest_artifacts,
        walkforward_validation_artifacts,
    ))
}

/// Replay each portfolio gene on a held-out tail window and produce one
/// [`ForwardTestValidationArtifactFile`] per strategy. The caller passes
/// the *raw* tail (with the same `feature_names` ordering it had before
/// discovery) and `effective_feature_names` produced by discovery; the
/// helper aligns the tail's columns to the post-prefilter set so the
/// gene indices match.
///
/// Returns `Err` when any name in `effective_feature_names` is missing
/// from the tail's columns — this indicates the tail comes from a
/// different feature pipeline than the discovery run that produced the
/// portfolio, and a forward-test on it would be meaningless.
pub fn compute_discovery_forward_test_artifacts(
    portfolio: &[Gene],
    effective_feature_names: &[String],
    tail_features: &FeatureFrame,
    tail_ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<Vec<ForwardTestValidationArtifactFile>> {
    if portfolio.is_empty() {
        return Ok(Vec::new());
    }

    // Project the tail's columns onto the post-prefilter set used by the
    // portfolio. When the tail already matches, this is a cheap clone of
    // the underlying ndarray; when it does not, we slice column-by-column.
    let tail_features = if tail_features.names == effective_feature_names {
        std::borrow::Cow::Borrowed(tail_features)
    } else {
        let mut keep_indices = Vec::with_capacity(effective_feature_names.len());
        for name in effective_feature_names {
            let idx = tail_features
                .names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "forward-test tail is missing feature '{}' from the discovery effective \
                         feature set; the tail must come from the same feature pipeline as the \
                         in-sample discovery run",
                        name
                    )
                })?;
            keep_indices.push(idx);
        }
        let n_rows = tail_features.data.nrows();
        let mut projected = ndarray::Array2::<f32>::zeros((n_rows, keep_indices.len()));
        for (new_idx, &orig_idx) in keep_indices.iter().enumerate() {
            projected
                .column_mut(new_idx)
                .assign(&tail_features.data.column(orig_idx));
        }
        std::borrow::Cow::Owned(FeatureFrame {
            timestamps: tail_features.timestamps.clone(),
            names: effective_feature_names.to_vec(),
            data: projected,
        })
    };
    let tail_features = tail_features.as_ref();

    let n = validation_row_count(tail_features, tail_ohlcv)?;
    if n == 0 {
        anyhow::bail!("forward-test tail must contain at least one bar");
    }

    let temporal_contract = discovery_temporal_contract(config, &tail_features.names)?;
    let tail_dataset_hash = discovery_dataset_hash(tail_features, tail_ohlcv)?;
    let (months, days) = month_day_indices(&tail_features.timestamps);
    let timestamps = &tail_features.timestamps[..n];

    let mut artifacts = Vec::with_capacity(portfolio.len());
    for gene in portfolio {
        let settings = discovery_backtest_settings(config, gene, tail_ohlcv.close.last().copied());
        let strategy_hash = stable_json_hash(gene)?;
        let evaluation_config_hash = discovery_backtest_policy_hash(config, gene, &settings)?;
        let evaluation_config = config.evaluation_config(tail_ohlcv.close.last().copied());
        let signals = signals_for_gene_full(tail_features, tail_ohlcv, gene, &evaluation_config);
        if signals.len() != n {
            anyhow::bail!(
                "forward-test signals length {} does not match validation row count {}",
                signals.len(),
                n
            );
        }
        let summary = compute_forward_test_summary(ForwardTestInput {
            close: &tail_ohlcv.close[..n],
            high: &tail_ohlcv.high[..n],
            low: &tail_ohlcv.low[..n],
            signals: &signals[..n],
            months: &months[..n],
            days: &days[..n],
            timestamps,
            settings: &settings,
        })?;
        artifacts.push(ForwardTestValidationArtifactFile::new(
            ForwardTestValidationScope::new(
                tail_dataset_hash.clone(),
                evaluation_config_hash,
                strategy_hash,
                &temporal_contract,
            ),
            summary,
        ));
    }
    Ok(artifacts)
}

/// Replay each portfolio gene on a held-out tail window, simulate trades
/// under the canonical backtest core, and aggregate them through
/// [`compute_prop_firm_risk_summary`] to produce one
/// [`PropFirmRiskValidationArtifactFile`] per strategy. The signature
/// mirrors [`compute_discovery_forward_test_artifacts`]: the caller
/// passes the tail with its original `feature_names` ordering, and the
/// helper aligns it to `effective_feature_names` before running the
/// simulation.
///
/// Returns `Err` when the tail is missing any effective feature, when
/// the tail is empty, or when the simulator produces a signal vector of
/// the wrong length — each path indicates the tail comes from a
/// different feature pipeline than the discovery run that produced the
/// portfolio.
pub fn compute_discovery_prop_firm_artifacts(
    portfolio: &[Gene],
    effective_feature_names: &[String],
    tail_features: &FeatureFrame,
    tail_ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    rules: PropFirmRiskRules,
) -> Result<Vec<PropFirmRiskValidationArtifactFile>> {
    if portfolio.is_empty() {
        return Ok(Vec::new());
    }

    let tail_features = if tail_features.names == effective_feature_names {
        std::borrow::Cow::Borrowed(tail_features)
    } else {
        let mut keep_indices = Vec::with_capacity(effective_feature_names.len());
        for name in effective_feature_names {
            let idx = tail_features
                .names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "prop-firm tail is missing feature '{}' from the discovery effective \
                         feature set; the tail must come from the same feature pipeline as the \
                         in-sample discovery run",
                        name
                    )
                })?;
            keep_indices.push(idx);
        }
        let n_rows = tail_features.data.nrows();
        let mut projected = ndarray::Array2::<f32>::zeros((n_rows, keep_indices.len()));
        for (new_idx, &orig_idx) in keep_indices.iter().enumerate() {
            projected
                .column_mut(new_idx)
                .assign(&tail_features.data.column(orig_idx));
        }
        std::borrow::Cow::Owned(FeatureFrame {
            timestamps: tail_features.timestamps.clone(),
            names: effective_feature_names.to_vec(),
            data: projected,
        })
    };
    let tail_features = tail_features.as_ref();

    let n = validation_row_count(tail_features, tail_ohlcv)?;
    if n == 0 {
        anyhow::bail!("prop-firm tail must contain at least one bar");
    }
    let temporal_contract = discovery_temporal_contract(config, &tail_features.names)?;
    let tail_dataset_hash = discovery_dataset_hash(tail_features, tail_ohlcv)?;
    let timestamps = &tail_features.timestamps[..n];

    let mut artifacts = Vec::with_capacity(portfolio.len());
    for gene in portfolio {
        let settings = discovery_backtest_settings(config, gene, tail_ohlcv.close.last().copied());
        let strategy_hash = stable_json_hash(gene)?;
        let evaluation_config_hash = discovery_backtest_policy_hash(config, gene, &settings)?;
        let evaluation_config = config.evaluation_config(tail_ohlcv.close.last().copied());
        let signals = signals_for_gene_full(tail_features, tail_ohlcv, gene, &evaluation_config);
        if signals.len() != n {
            anyhow::bail!(
                "prop-firm signals length {} does not match validation row count {}",
                signals.len(),
                n
            );
        }
        let trades = simulate_trades_core(
            &tail_ohlcv.close[..n],
            &tail_ohlcv.high[..n],
            &tail_ohlcv.low[..n],
            timestamps,
            &signals[..n],
            &settings,
        );
        let summary = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &trades,
            initial_balance: config.initial_balance,
            rules,
        });
        let scope = PropFirmRiskValidationScope::new(
            tail_dataset_hash.clone(),
            evaluation_config_hash,
            strategy_hash,
            &rules,
            &temporal_contract,
        )?;
        artifacts.push(PropFirmRiskValidationArtifactFile::new(scope, summary));
    }
    Ok(artifacts)
}

#[derive(Debug, Serialize)]
struct GeneExport<'a> {
    strategy_id: &'a str,
    indicators: Vec<&'a str>,
    indices: Vec<usize>,
    weights: Vec<f32>,
    long_threshold: f32,
    short_threshold: f32,
    fitness: f64,
    sharpe_ratio: f64,
    win_rate: f64,
    tp_pips: f64,
    sl_pips: f64,
}

pub fn run_discovery_cycle(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<DiscoveryResult> {
    run_discovery_cycle_with_progress(features, ohlcv, config, |_| {})
}

pub fn run_discovery_cycle_with_progress<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    let (mut features, ohlcv, _) = trim_recent_history(features, ohlcv, config)?;

    // Feature Pre-filtering (Idea #3)
    let prefilter_top_k = config.runtime_overrides.prefilter_top_k;
    let prefilter_insample_frac = config.runtime_overrides.resolved_prefilter_insample_frac();

    if prefilter_top_k > 0 && features.names.len() > prefilter_top_k {
        features = prefilter_features(&features, &ohlcv, prefilter_top_k, prefilter_insample_frac);
    }
    // Capture names after prefilter — gene indices refer to this list.
    let effective_feature_names = features.names.clone();

    // Multi-stage Funnel: Stage 1 (Fast Evaluation)
    let stage1_pct = config.runtime_overrides.resolved_funnel_stage1_pct();

    let stage1_len = (ohlcv.close.len() as f64 * stage1_pct) as usize;
    let ohlcv_stage1 = slice_ohlcv(&ohlcv, ohlcv.close.len() - stage1_len, ohlcv.close.len());
    let features_stage1 = FeatureFrame {
        timestamps: features.timestamps[features.timestamps.len() - stage1_len..].to_vec(),
        names: features.names.clone(),
        data: features
            .data
            .slice(ndarray::s![features.data.nrows() - stage1_len.., ..])
            .to_owned(),
    };
    progress_fn(DiscoveryProgress::SearchStarted {
        population: config.population,
        generations: config.generations,
        max_indicators: config.max_indicators,
    });
    let max_runtime = if config.max_hours > 0.0 {
        Some(std::time::Duration::from_secs_f64(
            config.max_hours * 3600.0,
        ))
    } else {
        None
    };
    let search = evolve_search_with_progress_and_limits(
        &features_stage1,
        &ohlcv_stage1,
        config.population,
        config.generations,
        config.max_indicators,
        max_runtime,
        Some(config.evaluation_config(ohlcv_stage1.close.last().copied())),
        |generation, total_generations, best_fitness, stagnant_generations, archived_profitable| {
            progress_fn(DiscoveryProgress::GenerationCompleted {
                generation,
                total_generations,
                best_fitness,
                stagnant_generations,
                archived_profitable,
            });
        },
    )?;

    finalize_candidates_with_progress(
        search.genes,
        &features,
        &ohlcv,
        config,
        effective_feature_names,
        progress_fn,
    )
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..x.len() {
        let a = x[i];
        let b = y[i];
        sum_x += a;
        sum_y += b;
        sum_xy += a * b;
        sum_x2 += a * a;
        sum_y2 += b * b;
    }

    let num = n * sum_xy - sum_x * sum_y;
    let den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    if den == 0.0 || !den.is_finite() {
        0.0
    } else {
        num / den
    }
}

fn prefilter_features(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    top_k: usize,
    insample_frac: f64,
) -> FeatureFrame {
    let n_rows = features.data.nrows();
    let n_cols = features.data.ncols();
    if n_rows < 2 || n_cols <= top_k {
        return features.clone();
    }

    // BUGFIX (data snooping): the prefilter ranks indicators by correlation
    // with 1-bar FORWARD returns. Computing this over the full dataset means
    // the "best" indicators are chosen with full knowledge of the OOS bars
    // they will later be evaluated on, inflating in-sample metrics.
    // Restrict the ranking to an IN-SAMPLE prefix so the final 30% of bars
    // (which the GA/walk-forward later treats as held-out) cannot leak into
    // the feature-selection step. The fraction is supplied by the caller
    // through `DiscoveryRuntimeOverrides::prefilter_insample_frac`.
    let train_end = ((n_rows as f64) * insample_frac).floor() as usize;
    let train_end = train_end.clamp(2, n_rows.saturating_sub(1)).max(2);

    // Calculate 1-bar forward returns ONLY for the in-sample window.
    // Returns past `train_end-1` are ignored (treated as zeros) so the
    // correlation reflects in-sample behaviour only.
    let mut returns = vec![0.0f32; n_rows];
    for (i, ret_slot) in returns
        .iter_mut()
        .enumerate()
        .take(train_end.saturating_sub(1))
    {
        let denom = ohlcv.close[i];
        if denom.abs() > 1e-12 {
            *ret_slot = ((ohlcv.close[i + 1] - denom) / denom) as f32;
        }
    }

    let mut correlations = Vec::with_capacity(n_cols);
    for col_idx in 0..n_cols {
        let name = &features.names[col_idx];
        if name.starts_with("regime_") {
            // Force keep regime columns by giving them infinite correlation
            correlations.push((col_idx, f32::INFINITY));
        } else {
            // Restrict the column slice to the in-sample window so the
            // Pearson correlation only sees in-sample co-movement.
            let col = features.data.column(col_idx);
            let col_full: Vec<f32> = col.iter().copied().collect();
            let col_train = &col_full[..train_end.saturating_sub(1)];
            let ret_train = &returns[..train_end.saturating_sub(1)];
            let corr = pearson_correlation(col_train, ret_train);
            correlations.push((col_idx, corr.abs()));
        }
    }

    correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate how many to actually keep: top_k + any regime columns
    let regime_count = features
        .names
        .iter()
        .filter(|n| n.starts_with("regime_"))
        .count();
    let actual_top_k = (top_k + regime_count).min(n_cols);

    let mut keep_indices: Vec<usize> = correlations
        .iter()
        .take(actual_top_k)
        .map(|(idx, _)| *idx)
        .collect();
    keep_indices.sort(); // Maintain original order

    let mut new_names = Vec::with_capacity(actual_top_k);
    let mut new_data = ndarray::Array2::zeros((n_rows, actual_top_k));

    for (new_col_idx, &orig_col_idx) in keep_indices.iter().enumerate() {
        new_names.push(features.names[orig_col_idx].clone());
        new_data
            .column_mut(new_col_idx)
            .assign(&features.data.column(orig_col_idx));
    }

    FeatureFrame {
        timestamps: features.timestamps.clone(),
        names: new_names,
        data: new_data,
    }
}

fn validate_regime_robustness(
    trades: &[crate::quality::Trade],
    features: &FeatureFrame,
    initial_balance: f64,
    max_regime_loss_pct: f64,
) -> bool {
    let trend_idx = features
        .names
        .iter()
        .position(|n| n == "regime_trend_strength");
    let vol_idx = features.names.iter().position(|n| n == "regime_vol_state");

    if trend_idx.is_none() || vol_idx.is_none() {
        return true;
    }
    let t_idx = trend_idx.unwrap();
    let v_idx = vol_idx.unwrap();

    let mut trend_pnl = 0.0;
    let mut range_pnl = 0.0;
    let mut high_vol_pnl = 0.0;
    let mut low_vol_pnl = 0.0;

    let mut last_idx = 0;
    let t_len = features.timestamps.len();

    for trade in trades {
        let ts = trade.entry_time;
        while last_idx < t_len && features.timestamps[last_idx] < ts {
            last_idx += 1;
        }
        let idx = if last_idx < t_len {
            last_idx
        } else {
            t_len.saturating_sub(1)
        };
        if idx >= features.data.nrows() {
            continue;
        }

        let trend_str = features.data[(idx, t_idx)];
        let vol_state = features.data[(idx, v_idx)];

        if trend_str > 0.25 {
            trend_pnl += trade.pnl;
        } else if trend_str < 0.15 {
            range_pnl += trade.pnl;
        }

        if vol_state > 0.5 {
            high_vol_pnl += trade.pnl;
        } else if vol_state < -0.5 {
            low_vol_pnl += trade.pnl;
        }
    }

    let limit = -(initial_balance * max_regime_loss_pct / 100.0);

    if trend_pnl < limit || range_pnl < limit || high_vol_pnl < limit || low_vol_pnl < limit {
        return false;
    }

    true
}

fn finalize_candidates_with_progress<F>(
    candidates: Vec<Gene>,
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    effective_feature_names: Vec<String>,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    // Sort by an income-focused ranking score to find reliably profitable ones
    let mut ranked_candidates: Vec<(usize, Gene)> = candidates.into_iter().enumerate().collect();

    let calculate_income_score = |gene: &Gene| -> f64 {
        let pf_capped = gene.profit_factor.min(3.0) / 3.0; // Normalized 0-1
        let safety = (1.0 - gene.max_drawdown / 0.07).clamp(0.0, 1.0);
        let consistency_score = gene.consistency; // 0-1
        let win_rate_score = gene.win_rate; // 0-1

        let multiplier =
            (consistency_score * 0.4) + (win_rate_score * 0.3) + (safety * 0.2) + (pf_capped * 0.1);

        // Bonus for high consistency (proxy for 10/12+ positive months)
        let bonus = if consistency_score > 0.8 { 2.0 } else { 1.0 };

        gene.fitness * multiplier * bonus
    };

    ranked_candidates.sort_by(|(idx_a, a), (idx_b, b)| {
        let score_a = calculate_income_score(a);
        let score_b = calculate_income_score(b);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.consistency
                    .partial_cmp(&a.consistency)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.strategy_id.cmp(&b.strategy_id))
            .then_with(|| idx_a.cmp(idx_b))
    });
    let max_candidates =
        candidate_truncation_limit(config.candidate_count, ranked_candidates.len());
    ranked_candidates.truncate(max_candidates);
    let ranked_candidate_genes: Vec<Gene> = ranked_candidates
        .iter()
        .map(|(_, gene)| gene.clone())
        .collect();
    progress_fn(DiscoveryProgress::CandidatesRanked {
        candidate_count: ranked_candidates.len(),
        truncated_to: max_candidates,
    });

    let min_trades = min_trades_required(
        &features.timestamps,
        config.min_trades_per_day,
        features.data.nrows(),
    );
    // Generate signals for all qualifying candidates in parallel — each call
    // accumulates weighted-feature columns over n_samples bars and the work
    // grows with the candidate count, so this scales well across cores.
    let prefiltered: Vec<(usize, Gene)> = ranked_candidates
        .iter()
        .filter(|(_, g)| g.passes_filter(&config.filtering))
        .map(|(idx, g)| (*idx, g.clone()))
        .collect();
    // Item 6: use the SMC-gated signal path so the post-search "min_trades"
    // filter sees the SAME trade count the evaluator scored. The previous
    // `signals_for_gene` ignored gene SMC flags; some candidates passed the
    // search archive (with their SMC-gated trade count) but were then pruned
    // here because the un-gated count was higher than min_trades.
    let eval_config_for_signals = config.evaluation_config(ohlcv.close.last().copied());
    let signals_with_idx: Vec<(usize, Gene, Vec<i8>)> = prefiltered
        .into_par_iter()
        .filter_map(|(candidate_idx, gene)| {
            let sig = signals_for_gene_full(features, ohlcv, &gene, &eval_config_for_signals);
            let trade_count = sig.iter().filter(|v| **v != 0).count() as f64;
            if trade_count >= min_trades as f64 {
                Some((candidate_idx, gene, sig))
            } else {
                None
            }
        })
        .collect();
    let mut filtered: Vec<(usize, Gene)> = Vec::with_capacity(signals_with_idx.len());
    let mut signals_map: Vec<Vec<i8>> = Vec::with_capacity(signals_with_idx.len());
    for (idx, gene, sig) in signals_with_idx {
        filtered.push((idx, gene));
        signals_map.push(sig);
    }
    progress_fn(DiscoveryProgress::CandidatesFiltered {
        passed_filters: filtered.len(),
        evaluated_candidates: ranked_candidates.len(),
        min_trades_required: min_trades,
    });

    let filtered_count = filtered.len();
    let mut quality_metrics = Vec::new();
    let mut logged_trades = Vec::new();
    if Gene::requires_quality_screen(&config.filtering) {
        type QualityCandidate = (usize, Gene, Vec<i8>, StrategyMetrics, bool, Vec<Trade>);
        let analyzer = quality_analyzer_for_config(config);
        let initial_balance = config.initial_balance;

        // Outer-parallel quality screen: each candidate runs simulate_trades +
        // 100 MC perturbations + spread sensitivity independently. Previously
        // the outer loop was serial and only the 100-run MC was parallel,
        // which under-utilised cores when the candidate set was large. Move
        // parallelism to the outer level and keep the MC loop serial — this
        // avoids rayon nested-parallel oversubscription and gives ~Ncores×
        // throughput on the per-candidate work.
        let pairs: Vec<((usize, Gene), Vec<i8>)> = filtered.into_iter().zip(signals_map).collect();
        let screened: Vec<Option<QualityCandidate>> = pairs
            .into_par_iter()
            .map(|((candidate_idx, gene), sig)| {
                let trades = crate::eval::simulate_trades_core(
                    &ohlcv.close,
                    &ohlcv.high,
                    &ohlcv.low,
                    &features.timestamps,
                    &sig,
                    &discovery_backtest_settings(config, &gene, ohlcv.close.last().copied()),
                );
                let metrics =
                    analyzer.analyze_strategy(&gene.strategy_id, &trades, initial_balance);
                let strict_quality = passes_strict_quality(&metrics, &config.filtering);
                let opportunistic_quality =
                    !strict_quality && passes_opportunistic_quality(&metrics, &config.filtering);

                if !(strict_quality || opportunistic_quality) {
                    return None;
                }

                // Regime-Aware Validation (Idea #3.2)
                let regime_robust = validate_regime_robustness(
                    &trades,
                    features,
                    config.initial_balance,
                    config.max_regime_loss_pct,
                );
                if !regime_robust {
                    return None;
                }

                // Monte Carlo Parameter Perturbation Test (100 runs).
                // Serial here because we are already inside a par_iter on
                // candidates — nesting rayon would oversubscribe cores.
                let mc_runs = 100usize;
                let mut profitable_runs = 0usize;
                let mut rng = rand::rng();
                use rand::Rng;
                for _ in 0..mc_runs {
                    let mut perturbed = gene.clone();
                    perturbed.long_threshold *= 1.0 + rng.random_range(-0.15..=0.15);
                    perturbed.short_threshold *= 1.0 + rng.random_range(-0.15..=0.15);
                    for w in &mut perturbed.weights {
                        *w *= 1.0 + rng.random_range(-0.20..=0.20);
                    }
                    if perturbed.sl_pips.is_finite() && perturbed.sl_pips > 0.0 {
                        perturbed.sl_pips *= 1.0 + rng.random_range(-0.25..=0.25);
                    }
                    if perturbed.tp_pips.is_finite() && perturbed.tp_pips > 0.0 {
                        perturbed.tp_pips *= 1.0 + rng.random_range(-0.25..=0.25);
                    }
                    // Item 6: SMC-gated signal so the MC perturbation reward is
                    // measured against the same execution rule the search used.
                    let p_sig = crate::genetic::signals_for_gene_full(
                        features,
                        ohlcv,
                        &perturbed,
                        &eval_config_for_signals,
                    );
                    let p_trades = crate::eval::simulate_trades_core(
                        &ohlcv.close,
                        &ohlcv.high,
                        &ohlcv.low,
                        &features.timestamps,
                        &p_sig,
                        &discovery_backtest_settings(
                            config,
                            &perturbed,
                            ohlcv.close.last().copied(),
                        ),
                    );
                    let pnl: f64 = p_trades.iter().map(|t| t.pnl).sum();
                    if pnl > 0.0 {
                        profitable_runs += 1;
                    }
                }

                if profitable_runs < 70 {
                    return None;
                }

                // Spread/Slippage Sensitivity Test
                let mut sensitive_settings =
                    discovery_backtest_settings(config, &gene, ohlcv.close.last().copied());
                sensitive_settings.spread_pips = 2.0;
                sensitive_settings.commission_per_trade = 7.0;
                let sens_trades = crate::eval::simulate_trades_core(
                    &ohlcv.close,
                    &ohlcv.high,
                    &ohlcv.low,
                    &features.timestamps,
                    &sig,
                    &sensitive_settings,
                );
                let sens_pnl: f64 = sens_trades.iter().map(|t| t.pnl).sum();
                if sens_pnl < 0.0 {
                    return None;
                }

                Some((
                    candidate_idx,
                    gene,
                    sig,
                    metrics,
                    opportunistic_quality,
                    trades,
                ))
            })
            .collect();

        let mut strict_passed: Vec<QualityCandidate> = Vec::new();
        let mut opportunistic_passed = 0usize;
        for entry in screened.into_iter().flatten() {
            if entry.4 {
                opportunistic_passed += 1;
            }
            quality_metrics.push(entry.3.clone());
            strict_passed.push(entry);
        }

        strict_passed.sort_by(|a, b| {
            let lane_a = if a.4 { 0_u8 } else { 1_u8 };
            let lane_b = if b.4 { 0_u8 } else { 1_u8 };
            lane_b
                .cmp(&lane_a)
                .then_with(|| {
                    b.3.quality_score
                        .partial_cmp(&a.3.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    b.1.fitness
                        .partial_cmp(&a.1.fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.1.strategy_id.cmp(&b.1.strategy_id))
                .then_with(|| a.0.cmp(&b.0))
        });

        if config.filtering.log_trades {
            logged_trades = strict_passed
                .iter()
                .filter(|entry| !entry.5.is_empty())
                .take(config.filtering.trade_log_max)
                .map(|entry| LoggedStrategyTrades {
                    strategy_id: entry.1.strategy_id.clone(),
                    opportunistic: entry.4,
                    trades: entry.5.clone(),
                })
                .collect();
        }
        let logged_trade_sets = logged_trades.len();

        progress_fn(DiscoveryProgress::QualityScreened {
            strict_passed: strict_passed.len().saturating_sub(opportunistic_passed),
            opportunistic_passed,
            evaluated_candidates: filtered_count,
            logged_trade_sets,
        });

        let mut screened_genes = Vec::with_capacity(strict_passed.len());
        let mut screened_signals = Vec::with_capacity(strict_passed.len());
        for (candidate_idx, gene, sig, _, _, _) in strict_passed {
            screened_genes.push((candidate_idx, gene));
            screened_signals.push(sig);
        }
        filtered = screened_genes;
        signals_map = screened_signals;
    }

    let mut portfolio = Vec::new();
    let mut portfolio_signals: Vec<Vec<i8>> = Vec::new();
    let mut rejected_by_correlation = 0usize;
    for ((_, gene), sig) in filtered.into_iter().zip(signals_map) {
        if portfolio.len() >= config.portfolio_size {
            break;
        }
        let mut ok = true;
        for existing in &portfolio_signals {
            let pearson = pearson_corr_i8(&sig, existing);
            // DS-2: also check Spearman to catch non-linear dependencies
            let spearman = spearman_corr_i8(&sig, existing);
            // Reject if EITHER correlation exceeds threshold
            if pearson.abs() >= config.corr_threshold || spearman.abs() >= config.corr_threshold {
                ok = false;
                rejected_by_correlation += 1;
                break;
            }
        }
        if ok {
            portfolio_signals.push(sig);
            portfolio.push(gene);
        }
    }
    progress_fn(DiscoveryProgress::PortfolioSelected {
        portfolio_size: portfolio.len(),
        rejected_by_correlation,
        target_portfolio: config.portfolio_size,
    });
    let (validation_gates, canonical_backtest_artifacts, walkforward_validation_artifacts) =
        build_discovery_validation_artifacts(
            &portfolio,
            &portfolio_signals,
            features,
            ohlcv,
            config,
        )?;
    progress_fn(DiscoveryProgress::Completed {
        candidate_count: ranked_candidate_genes.len(),
        filtered_count,
        portfolio_size: portfolio.len(),
    });

    Ok(DiscoveryResult {
        portfolio,
        candidates: ranked_candidate_genes,
        quality_metrics,
        logged_trades,
        effective_feature_names,
        validation_gates,
        canonical_backtest_artifacts,
        walkforward_validation_artifacts,
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
    })
}

fn candidate_truncation_limit(requested: usize, available: usize) -> usize {
    if available == 0 {
        0
    } else if requested == 0 {
        available
    } else {
        requested.min(available)
    }
}

fn min_trades_required(timestamps: &[i64], min_trades_per_day: f64, n_rows: usize) -> usize {
    if timestamps.is_empty() {
        let days = (n_rows as f64 / 1440.0).max(1.0);
        return (days * min_trades_per_day).ceil() as usize;
    }
    let mut days = HashSet::new();
    for ts in timestamps {
        if let Some(dt) = Utc.timestamp_millis_opt(*ts).single()
            && dt.weekday().num_days_from_monday() < 5
        {
            let key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
            days.insert(key);
        }
    }
    let day_count = days.len().max(1) as f64;
    (day_count * min_trades_per_day).ceil() as usize
}

/// DS-2: Spearman rank correlation for i8 signals.
/// For discrete values (-1, 0, 1), ranks ties by mean rank. Detects monotonic (non-linear) dependency.
fn spearman_corr_i8(a: &[i8], b: &[i8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    // For i8 with only 3 distinct values, compute rank as fractional rank
    // mean_rank(v) = (first_idx + last_idx) / 2 over sorted positions
    let rank_of = |vals: &[i8], v: i8| -> f64 {
        let count = vals[..n].iter().filter(|&&x| x == v).count() as f64;
        let before = vals[..n].iter().filter(|&&x| x < v).count() as f64;
        before + (count + 1.0) / 2.0
    };
    let ranks_a: Vec<f64> = a[..n].iter().map(|&v| rank_of(&a[..n], v)).collect();
    let ranks_b: Vec<f64> = b[..n].iter().map(|&v| rank_of(&b[..n], v)).collect();
    let mean_a: f64 = ranks_a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = ranks_b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut denom_a = 0.0_f64;
    let mut denom_b = 0.0_f64;
    for i in 0..n {
        let da = ranks_a[i] - mean_a;
        let db = ranks_b[i] - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if denom_a <= 1e-12 || denom_b <= 1e-12 {
        return 0.0;
    }
    num / (denom_a.sqrt() * denom_b.sqrt())
}

fn pearson_corr_i8(a: &[i8], b: &[i8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..n {
        sum_a += a[i] as f64;
        sum_b += b[i] as f64;
    }
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;
    let mut num = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    for i in 0..n {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if denom_a <= 1e-12 || denom_b <= 1e-12 {
        return 0.0;
    }
    num / (denom_a.sqrt() * denom_b.sqrt())
}

pub fn ensure_portfolio_export_ready(result: &DiscoveryResult) -> Result<()> {
    if result.validation_gates.is_portfolio_export_ready() {
        return Ok(());
    }
    anyhow::bail!(
        "portfolio export requires validation gates: walkforward_passed={} cpcv_passed={}",
        result.validation_gates.walkforward_passed,
        result.validation_gates.cpcv_passed
    );
}

fn build_portfolio_exports<'a>(
    portfolio: &'a [Gene],
    feature_names: &'a [String],
) -> Vec<GeneExport<'a>> {
    let mut exports = Vec::new();
    for gene in portfolio {
        let mut names = Vec::new();
        for idx in &gene.indices {
            if let Some(name) = feature_names.get(*idx) {
                names.push(name.as_str());
            }
        }
        exports.push(GeneExport {
            strategy_id: &gene.strategy_id,
            indicators: names,
            indices: gene.indices.clone(),
            weights: gene.weights.clone(),
            long_threshold: gene.long_threshold,
            short_threshold: gene.short_threshold,
            fitness: gene.fitness,
            sharpe_ratio: gene.sharpe_ratio,
            win_rate: gene.win_rate,
            tp_pips: gene.tp_pips,
            sl_pips: gene.sl_pips,
        });
    }
    exports
}

pub fn save_portfolio_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    ensure_portfolio_export_ready(result)?;
    let exports = build_portfolio_exports(&result.portfolio, &result.effective_feature_names);
    write_json_atomic(path, &exports)
}

pub fn save_quality_report_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    write_json_atomic(path, &result.quality_metrics)
}

pub fn save_trade_log_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    write_json_atomic(path, &result.logged_trades)
}

fn artifact_filename_for_strategy_hash(strategy_hash: &str, fallback_index: usize) -> String {
    let cleaned: String = strategy_hash
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            _ => '_',
        })
        .collect();
    if cleaned.is_empty() {
        format!("strategy_{fallback_index:04}.json")
    } else {
        format!("{cleaned}.json")
    }
}

pub fn save_canonical_backtest_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    if result.canonical_backtest_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create canonical backtest dir {}", dir.display()))?;
    for (idx, artifact) in result.canonical_backtest_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(&artifact.scope.strategy_hash, idx);
        write_canonical_backtest_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.canonical_backtest_artifacts.len())
}

pub fn save_walkforward_validation_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    if result.walkforward_validation_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create walk-forward validation dir {}", dir.display()))?;
    for (idx, artifact) in result.walkforward_validation_artifacts.iter().enumerate() {
        let strategy_hash = artifact
            .scope
            .strategy_hash
            .as_deref()
            .unwrap_or("portfolio");
        let file_name = artifact_filename_for_strategy_hash(strategy_hash, idx);
        write_walkforward_validation_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.walkforward_validation_artifacts.len())
}

pub fn save_forward_test_validation_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    if result.forward_test_validation_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create forward-test validation dir {}", dir.display()))?;
    for (idx, artifact) in result.forward_test_validation_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(&artifact.scope.strategy_hash, idx);
        write_forward_test_validation_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.forward_test_validation_artifacts.len())
}

/// Persist a focused promotion-readiness summary at `path` derived
/// from the discovery result. The summary is the same per-kind
/// evidence + missing-kinds + producer-side-completeness payload that
/// already lives on `DiscoveryRunProfile` (Phase 49), but written to
/// its own file so operators / UI scrapers can poll it without
/// parsing the full profile JSON.
pub fn save_promotion_summary_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    #[derive(Serialize)]
    struct PromotionSummary<'a> {
        validation_evidence_hashes: &'a DiscoveryPerKindEvidenceHashes,
        validation_evidence_complete: bool,
        validation_evidence_missing_kinds: Vec<&'static str>,
        producer_side_complete: bool,
        check_summary: Vec<(&'static str, &'static str)>,
        determinism_policy: forex_core::contracts::DeterminismPolicy,
    }
    let hashes = discovery_per_kind_evidence_hashes(result)?;
    let summary = PromotionSummary {
        producer_side_complete: hashes.all_producer_kinds_present(),
        check_summary: hashes.check_summary(),
        validation_evidence_complete: hashes.all_present(),
        validation_evidence_missing_kinds: hashes.missing_kinds(),
        validation_evidence_hashes: &hashes,
        determinism_policy: crate::genetic::current_determinism_policy(),
    };
    write_json_atomic(path, &summary)
}

pub fn save_prop_firm_validation_artifacts(
    dir: impl AsRef<Path>,
    result: &DiscoveryResult,
) -> Result<usize> {
    let dir = dir.as_ref();
    if result.prop_firm_validation_artifacts.is_empty() {
        return Ok(0);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create prop-firm validation dir {}", dir.display()))?;
    for (idx, artifact) in result.prop_firm_validation_artifacts.iter().enumerate() {
        let file_name = artifact_filename_for_strategy_hash(&artifact.scope.strategy_hash, idx);
        write_prop_firm_risk_validation_artifact_atomic(dir.join(file_name), artifact)?;
    }
    Ok(result.prop_firm_validation_artifacts.len())
}

/// Translate a [`DiscoveryResult`] into a typed
/// [`forex_core::contracts::LiveValidationEvidence`] record so a live
/// bridge can call `LiveExecutionContract::validate_evidence` without
/// re-deriving any pass/fail logic itself. The mapping is:
///
/// - `walkforward_passed` / `cpcv_passed` come straight from
///   `result.validation_gates`.
/// - `forward_test_passed` is `Some(true)` only when the result carries
///   at least one forward-test artifact AND every artifact reports a
///   non-zero trade count with strictly positive net profit.
///   `Some(false)` is returned when artifacts exist but at least one
///   fails the rule, and `None` when no artifact was produced (the live
///   bridge will treat that as missing evidence if it requires the
///   gate).
/// - `prop_firm_passed` aggregates the per-strategy
///   [`PropFirmRiskValidationArtifactFile::summary.all_rules_passed`]
///   flags: `Some(true)` when every persisted prop-firm artifact passes,
///   `Some(false)` when at least one fails, and `None` when no
///   prop-firm artifact was produced (the live bridge will treat that
///   as missing evidence whenever the gate is required).
/// - `live_sim_runtime_model_hash` stays `None` until a live-execution
///   simulator is wired into the discovery pipeline.
pub fn live_validation_evidence_from_discovery(result: &DiscoveryResult) -> LiveValidationEvidence {
    let forward_test_passed = if result.forward_test_validation_artifacts.is_empty() {
        None
    } else {
        let all_pass = result
            .forward_test_validation_artifacts
            .iter()
            .all(|artifact| {
                artifact.summary.metrics.trade_count > 0
                    && artifact.summary.metrics.net_profit > 0.0
            });
        Some(all_pass)
    };
    let prop_firm_passed = if result.prop_firm_validation_artifacts.is_empty() {
        None
    } else {
        let all_pass = result
            .prop_firm_validation_artifacts
            .iter()
            .all(|artifact| artifact.summary.all_rules_passed);
        Some(all_pass)
    };
    LiveValidationEvidence {
        walkforward_passed: result.validation_gates.walkforward_passed,
        cpcv_passed: result.validation_gates.cpcv_passed,
        forward_test_passed,
        prop_firm_passed,
        live_sim_runtime_model_hash: None,
    }
}

/// Build a [`ValidationEvidenceManifest`] from the persisted discovery
/// artifacts. The helper computes one stable hash per artifact kind by
/// hashing the full vector of per-strategy artifacts; an empty vector
/// produces an empty hash, which causes
/// [`ValidationEvidenceManifest::validate`] to surface a typed
/// `MissingValidationEvidence` error naming the missing kind.
///
/// Today this always returns an error for the
/// `live_execution_simulation_hash` kind because `DiscoveryResult` does
/// not yet carry live-sim artifacts (the simulator is still deferred).
/// Callers that want a partial manifest for diagnostic display should
/// use the per-kind helpers below; callers that need a fully-validated
/// manifest must wait until the live-execution simulator lands.
pub fn discovery_validation_evidence_manifest(
    result: &DiscoveryResult,
) -> Result<ValidationEvidenceManifest> {
    let canonical = hash_validation_artifacts(&result.canonical_backtest_artifacts)?;
    let walkforward = hash_validation_artifacts(&result.walkforward_validation_artifacts)?;
    let forward_test = hash_validation_artifacts(&result.forward_test_validation_artifacts)?;
    let prop_firm = hash_validation_artifacts(&result.prop_firm_validation_artifacts)?;
    // Live-execution simulation artifacts are not yet emitted by the
    // discovery pipeline — propagate as the empty string so the
    // manifest's `validate()` rejects with the typed
    // `MissingValidationEvidence("live_execution_simulation_hash")`
    // variant rather than silently filling a placeholder.
    let live_sim = String::new();
    ValidationEvidenceManifest::new(canonical, walkforward, forward_test, live_sim, prop_firm)
        .map_err(|err| anyhow::anyhow!(err.to_string()))
}

/// Build a [`ValidationEvidenceManifest`] without enforcing the
/// always-missing `live_execution_simulation_hash` gate. Producer-side
/// kinds that are missing still return an error — the relaxation only
/// covers the simulator hash that is structurally absent until the
/// simulator lands. Operators / UI layers can use this for diagnostic
/// display ("which producer-side kinds shipped?") without tripping on
/// the structural live-sim absence.
pub fn discovery_validation_evidence_manifest_excluding_live_sim(
    result: &DiscoveryResult,
) -> Result<ValidationEvidenceManifest> {
    let canonical = hash_validation_artifacts(&result.canonical_backtest_artifacts)?;
    let walkforward = hash_validation_artifacts(&result.walkforward_validation_artifacts)?;
    let forward_test = hash_validation_artifacts(&result.forward_test_validation_artifacts)?;
    let prop_firm = hash_validation_artifacts(&result.prop_firm_validation_artifacts)?;
    let live_sim = "deferred:live_execution_simulator_not_wired".to_string();
    ValidationEvidenceManifest::new(canonical, walkforward, forward_test, live_sim, prop_firm)
        .map_err(|err| anyhow::anyhow!(err.to_string()))
}

/// Per-kind helper that returns `Some(hash)` when the artifact vector
/// is non-empty and `None` otherwise. Operator/UI layers can use this
/// to build a diagnostic view ("forward-test artifact present, live-sim
/// missing") without forcing a full manifest validation.
pub fn discovery_per_kind_evidence_hashes(
    result: &DiscoveryResult,
) -> Result<DiscoveryPerKindEvidenceHashes> {
    Ok(DiscoveryPerKindEvidenceHashes {
        canonical_backtest: optional_hash_validation_artifacts(
            &result.canonical_backtest_artifacts,
        )?,
        walkforward: optional_hash_validation_artifacts(&result.walkforward_validation_artifacts)?,
        forward_test: optional_hash_validation_artifacts(
            &result.forward_test_validation_artifacts,
        )?,
        prop_firm: optional_hash_validation_artifacts(&result.prop_firm_validation_artifacts)?,
        live_execution_simulation: None,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveryPerKindEvidenceHashes {
    pub canonical_backtest: Option<String>,
    pub walkforward: Option<String>,
    pub forward_test: Option<String>,
    pub prop_firm: Option<String>,
    pub live_execution_simulation: Option<String>,
}

impl DiscoveryPerKindEvidenceHashes {
    /// Returns `true` only when every kind has a non-empty hash. The
    /// live-execution simulation hash is part of this check, so the
    /// summary will currently always return `false` until a simulator
    /// produces evidence.
    pub fn all_present(&self) -> bool {
        self.canonical_backtest.is_some()
            && self.walkforward.is_some()
            && self.forward_test.is_some()
            && self.prop_firm.is_some()
            && self.live_execution_simulation.is_some()
    }

    /// Returns `true` when every producer-side kind (canonical,
    /// walkforward, forward-test, prop-firm) is present, ignoring the
    /// always-missing `live_execution_simulation` hash. Operators that
    /// want to gauge producer-side completeness without waiting for the
    /// simulator can use this instead of `all_present()`.
    pub fn all_producer_kinds_present(&self) -> bool {
        self.canonical_backtest.is_some()
            && self.walkforward.is_some()
            && self.forward_test.is_some()
            && self.prop_firm.is_some()
    }

    /// Returns one `(kind_name, status)` tuple per validation kind,
    /// where `status` is `"present"` or `"missing"`. Render directly
    /// in operator-facing log lines / UI tables without re-deriving
    /// per-kind logic.
    pub fn check_summary(&self) -> Vec<(&'static str, &'static str)> {
        let label = |opt: &Option<String>| if opt.is_some() { "present" } else { "missing" };
        vec![
            ("canonical_backtest", label(&self.canonical_backtest)),
            ("walkforward", label(&self.walkforward)),
            ("forward_test", label(&self.forward_test)),
            ("prop_firm", label(&self.prop_firm)),
            (
                "live_execution_simulation",
                label(&self.live_execution_simulation),
            ),
        ]
    }

    /// Returns the list of kinds that have no hash on this profile.
    /// Operators / UI layers can render this directly without parsing
    /// `MissingValidationEvidence` strings.
    pub fn missing_kinds(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.canonical_backtest.is_none() {
            missing.push("canonical_backtest");
        }
        if self.walkforward.is_none() {
            missing.push("walkforward");
        }
        if self.forward_test.is_none() {
            missing.push("forward_test");
        }
        if self.prop_firm.is_none() {
            missing.push("prop_firm");
        }
        if self.live_execution_simulation.is_none() {
            missing.push("live_execution_simulation");
        }
        missing
    }
}

fn hash_validation_artifacts<T: Serialize>(artifacts: &[T]) -> Result<String> {
    if artifacts.is_empty() {
        Ok(String::new())
    } else {
        stable_json_hash(artifacts)
    }
}

fn optional_hash_validation_artifacts<T: Serialize>(artifacts: &[T]) -> Result<Option<String>> {
    if artifacts.is_empty() {
        Ok(None)
    } else {
        stable_json_hash(artifacts).map(Some)
    }
}

pub fn build_discovery_profile(
    config: &DiscoveryConfig,
    result: &DiscoveryResult,
) -> DiscoveryRunProfile {
    let validation_evidence_hashes =
        discovery_per_kind_evidence_hashes(result).unwrap_or_else(|_| {
            DiscoveryPerKindEvidenceHashes {
                canonical_backtest: None,
                walkforward: None,
                forward_test: None,
                prop_firm: None,
                live_execution_simulation: None,
            }
        });
    DiscoveryRunProfile {
        timeframe_label: config.timeframe_label.clone(),
        population: config.population,
        generations: config.generations,
        max_indicators: config.max_indicators,
        candidate_count_target: config.candidate_count,
        portfolio_size_target: config.portfolio_size,
        max_rows: row_cap_for_config(config),
        max_runtime_hours: config.max_hours,
        corr_threshold: config.corr_threshold,
        min_trades_per_day: config.min_trades_per_day,
        walkforward_splits: config.walkforward_splits,
        embargo_minutes: config.embargo_minutes,
        enable_cpcv: config.enable_cpcv,
        cpcv_n_splits: config.cpcv_n_splits,
        cpcv_n_test_groups: config.cpcv_n_test_groups,
        cpcv_embargo_pct: config.cpcv_embargo_pct,
        cpcv_purge_pct: config.cpcv_purge_pct,
        cpcv_min_phi: config.cpcv_min_phi,
        filters: DiscoveryFilterProfile {
            max_dd: config.filtering.max_dd,
            min_profit: config.filtering.min_profit,
            min_trades: config.filtering.min_trades,
            min_sharpe: config.filtering.min_sharpe,
            min_win_rate: config.filtering.min_win_rate,
            min_profit_factor: config.filtering.min_profit_factor,
            min_positive_months: config.filtering.min_positive_months,
            min_trades_per_month: config.filtering.min_trades_per_month,
            min_monthly_return_pct: config.filtering.min_monthly_return_pct,
            opportunistic_enabled: config.filtering.use_opportunistic_candidates
                && config.filtering.opportunistic_enabled,
            opportunistic_min_positive_months: config.filtering.opportunistic_min_positive_months,
            opportunistic_min_trades_per_month: config.filtering.opportunistic_min_trades_per_month,
            opportunistic_min_trade_return_pct: config.filtering.opportunistic_min_trade_return_pct,
            opportunistic_max_dd: config.filtering.opportunistic_max_dd,
            log_trades: config.filtering.log_trades,
            trade_log_max: config.filtering.trade_log_max,
        },
        candidates_observed: result.candidates.len(),
        portfolio_observed: result.portfolio.len(),
        quality_metrics_observed: result.quality_metrics.len(),
        logged_trade_sets: result.logged_trades.len(),
        walkforward_passed: result.validation_gates.walkforward_passed,
        cpcv_passed: result.validation_gates.cpcv_passed,
        canonical_backtest_artifacts_observed: result.validation_gates.canonical_backtest_artifacts,
        walkforward_validation_artifacts_observed: result
            .validation_gates
            .walkforward_validation_artifacts,
        forward_test_validation_artifacts_observed: result.forward_test_validation_artifacts.len(),
        prop_firm_validation_artifacts_observed: result.prop_firm_validation_artifacts.len(),
        cpcv_fold_count: result.validation_gates.cpcv_fold_count,
        cpcv_profitable_fold_ratio: result.validation_gates.cpcv_profitable_fold_ratio,
        validation_temporal_contract_hash: result.validation_gates.temporal_contract_hash.clone(),
        prefilter_top_k: config.runtime_overrides.prefilter_top_k,
        prefilter_insample_frac: config.runtime_overrides.resolved_prefilter_insample_frac(),
        funnel_stage1_pct: config.runtime_overrides.resolved_funnel_stage1_pct(),
        validation_evidence_hashes: validation_evidence_hashes.clone(),
        validation_evidence_complete: validation_evidence_hashes.all_present(),
        validation_evidence_missing_kinds: validation_evidence_hashes
            .missing_kinds()
            .into_iter()
            .map(str::to_string)
            .collect(),
        determinism_policy: crate::genetic::current_determinism_policy(),
    }
}

pub fn save_discovery_profile_json(
    path: impl AsRef<Path>,
    config: &DiscoveryConfig,
    result: &DiscoveryResult,
) -> Result<()> {
    write_json_atomic(path, &build_discovery_profile(config, result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FilteringConfig;
    use ndarray::Array2;

    fn sample_feature_frame() -> FeatureFrame {
        let start = 1_704_067_200_000_i64;
        FeatureFrame {
            timestamps: (0..10).map(|idx| start + idx * 60_000).collect(),
            names: vec!["signal".to_string()],
            data: Array2::from_shape_vec(
                (10, 1),
                vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            )
            .expect("feature frame shape should be valid"),
        }
    }

    fn sample_ohlcv() -> Ohlcv {
        let start = 1_704_067_200_000_i64;
        let close: Vec<f64> = vec![
            1.1000, 1.1010, 1.1020, 1.1015, 1.1030, 1.1025, 1.1040, 1.1035, 1.1050, 1.1045,
        ];
        let open: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                if idx == 0 {
                    *value - 0.0005
                } else {
                    close[idx - 1]
                }
            })
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .map(|(open, close)| open.max(*close) + 0.0004)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .map(|(open, close)| open.min(*close) - 0.0004)
            .collect();
        let volume: Vec<f64> = (0..10).map(|idx| 1000.0 + (idx as f64 * 25.0)).collect();

        Ohlcv {
            timestamp: Some((0..10).map(|idx| start + idx * 60_000).collect()),
            open,
            high,
            low,
            close,
            volume: Some(volume),
        }
    }

    fn profitable_gene(strategy_id: &str) -> Gene {
        Gene {
            strategy_id: strategy_id.to_string(),
            indices: vec![0],
            weights: vec![1.0],
            long_threshold: 0.5,
            short_threshold: -0.5,
            fitness: 150.0,
            sharpe_ratio: 1.4,
            win_rate: 0.61,
            max_drawdown: 0.04,
            profit_factor: 1.3,
            trades_count: 10,
            consistency: 0.8,
            ..Gene::default()
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("forex-discovery-{name}-{unique}.json"))
    }

    #[test]
    fn empty_portfolio_is_an_explicit_error() {
        let result = DiscoveryResult {
            portfolio: Vec::new(),
            candidates: vec![Gene::default()],
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: Vec::new(),
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        };

        let err = ensure_non_empty_portfolio(&result, "EURUSD M1")
            .expect_err("expected empty discovery portfolio to fail");
        let msg = err.to_string();
        assert!(msg.contains("empty portfolio"), "unexpected error: {msg}");
        assert!(msg.contains("candidates=1"), "unexpected error: {msg}");
    }

    #[test]
    fn non_empty_portfolio_is_accepted() {
        let result = DiscoveryResult {
            portfolio: vec![Gene::default()],
            candidates: vec![Gene::default()],
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: Vec::new(),
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        };

        ensure_non_empty_portfolio(&result, "EURUSD M1")
            .expect("expected non-empty portfolio to pass");
    }

    #[test]
    fn candidate_truncation_honors_small_explicit_limits() {
        assert_eq!(candidate_truncation_limit(2, 500), 2);
        assert_eq!(candidate_truncation_limit(0, 500), 500);
        assert_eq!(candidate_truncation_limit(500, 2), 2);
        assert_eq!(candidate_truncation_limit(5, 0), 0);
    }

    #[test]
    fn finalize_candidates_with_progress_emits_filter_and_portfolio_milestones() {
        let features = sample_feature_frame();
        let ohlcv = sample_ohlcv();
        let config = DiscoveryConfig {
            candidate_count: 2,
            portfolio_size: 2,
            corr_threshold: 0.9,
            min_trades_per_day: 1.0,
            filtering: FilteringConfig {
                min_profit: 1.0,
                min_trades: 1.0,
                min_sharpe: 0.1,
                min_win_rate: 0.5,
                min_profit_factor: 1.01,
                max_dd: 0.2,
                anomaly_guard: false,
                elite_mode: false,
                ..FilteringConfig::default()
            },
            ..DiscoveryConfig::default()
        };
        let candidates = vec![profitable_gene("alpha-1"), profitable_gene("alpha-2")];
        let mut progress_events = Vec::new();

        let result = finalize_candidates_with_progress(
            candidates,
            &features,
            &ohlcv,
            &config,
            features.names.clone(),
            |event| progress_events.push(event),
        )
        .expect("candidate finalization should succeed");

        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.portfolio.len(), 1);
        assert_eq!(
            result.canonical_backtest_artifacts.len(),
            result.portfolio.len()
        );
        assert_eq!(
            result.walkforward_validation_artifacts.len(),
            result.portfolio.len()
        );
        assert_eq!(
            result.validation_gates.canonical_backtest_artifacts,
            result.portfolio.len()
        );
        assert!(result.validation_gates.temporal_contract_hash.is_some());
        assert!(progress_events.iter().any(|event| matches!(
            event,
            DiscoveryProgress::CandidatesRanked { candidate_count, truncated_to }
                if *candidate_count == 2 && *truncated_to == 2
        )));
        assert!(progress_events.iter().any(|event| matches!(
            event,
            DiscoveryProgress::CandidatesFiltered { passed_filters, evaluated_candidates, min_trades_required }
                if *passed_filters == 2 && *evaluated_candidates == 2 && *min_trades_required == 1
        )));
        assert!(progress_events.iter().any(|event| matches!(
            event,
            DiscoveryProgress::PortfolioSelected { portfolio_size, rejected_by_correlation, target_portfolio }
                if *portfolio_size == 1 && *rejected_by_correlation == 1 && *target_portfolio == 2
        )));
        assert!(progress_events.iter().any(|event| matches!(
            event,
            DiscoveryProgress::Completed { candidate_count, filtered_count, portfolio_size }
                if *candidate_count == 2 && *filtered_count == 2 && *portfolio_size == 1
        )));
    }

    #[test]
    fn portfolio_export_requires_validation_gates() {
        let result = DiscoveryResult {
            portfolio: vec![profitable_gene("alpha-1")],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["signal".to_string()],
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        };
        let path = temp_path("portfolio-gates");

        let err = save_portfolio_json(&path, &result)
            .expect_err("portfolio export must fail before validation gates pass");
        assert!(err.to_string().contains("walkforward_passed"));
        assert!(!path.exists());
    }

    #[test]
    fn portfolio_export_uses_effective_names_after_validation_gates_pass() {
        let mut result = DiscoveryResult {
            portfolio: vec![profitable_gene("alpha-1")],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["filtered_signal".to_string()],
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        };
        result.validation_gates.walkforward_passed = true;
        result.validation_gates.cpcv_passed = true;
        let path = temp_path("portfolio-export");

        save_portfolio_json(&path, &result)
            .expect("portfolio export should pass once validation gates are true");
        let exported = std::fs::read_to_string(&path).expect("portfolio export should exist");
        assert!(exported.contains("filtered_signal"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn discovery_profile_exports_validation_gate_status() {
        let mut result = DiscoveryResult {
            portfolio: vec![profitable_gene("alpha-1")],
            candidates: vec![profitable_gene("alpha-1")],
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["signal".to_string()],
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        };
        result.validation_gates.walkforward_passed = true;
        result.validation_gates.cpcv_passed = true;
        result.validation_gates.canonical_backtest_artifacts = 1;
        result.validation_gates.walkforward_validation_artifacts = 1;
        result.validation_gates.cpcv_fold_count = 3;
        result.validation_gates.cpcv_profitable_fold_ratio = 1.0;

        let profile = build_discovery_profile(&DiscoveryConfig::default(), &result);

        assert!(profile.walkforward_passed);
        assert!(profile.cpcv_passed);
        assert_eq!(profile.canonical_backtest_artifacts_observed, 1);
        assert_eq!(profile.walkforward_validation_artifacts_observed, 1);
        assert_eq!(profile.cpcv_fold_count, 3);
        assert_eq!(profile.cpcv_profitable_fold_ratio, 1.0);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("forex-discovery-{name}-{unique}"))
    }

    fn sample_temporal_contract() -> TemporalFeatureContract {
        discovery_temporal_contract(&DiscoveryConfig::default(), &["signal".to_string()])
            .expect("temporal contract for default discovery config")
    }

    fn sample_canonical_backtest_artifact(strategy_hash: &str) -> CanonicalBacktestArtifactFile {
        let contract = sample_temporal_contract();
        let scope = CanonicalBacktestScope::new("dataset", "evaluation", strategy_hash, &contract);
        CanonicalBacktestArtifactFile::new(scope, BacktestMetrics::from_metric_array([0.0; 11]))
    }

    fn sample_walkforward_summary() -> WalkforwardSummary {
        WalkforwardSummary {
            walk_forward_splits: 1,
            avg_pnl: 1.0,
            avg_win_rate: 0.5,
            avg_max_dd: 0.1,
            avg_max_consec_losses: 0.0,
            avg_daily_min_dd: 0.0,
            avg_max_daily_loss: 0.0,
            any_daily_loss_breach: false,
            any_consistency_violation: false,
            any_trade_limit_violation: false,
            all_min_trading_days_ok: true,
            splits: Vec::new(),
        }
    }

    fn sample_walkforward_validation_artifact(
        strategy_hash: &str,
    ) -> WalkforwardValidationArtifactFile {
        let contract = sample_temporal_contract();
        let scope = WalkforwardValidationScope::for_strategy(
            "dataset",
            "evaluation",
            strategy_hash,
            &contract,
        );
        WalkforwardValidationArtifactFile::new(scope, sample_walkforward_summary())
    }

    #[test]
    fn save_canonical_backtest_artifacts_writes_one_file_per_strategy() {
        let dir = temp_dir("canonical-backtests");
        let result = DiscoveryResult {
            portfolio: vec![profitable_gene("alpha-1"), profitable_gene("alpha-2")],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["signal".to_string()],
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: vec![
                sample_canonical_backtest_artifact("fnv64:0123456789abcdef"),
                sample_canonical_backtest_artifact("fnv64:fedcba9876543210"),
            ],
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        };

        let written = save_canonical_backtest_artifacts(&dir, &result)
            .expect("canonical backtest artifacts should persist");
        assert_eq!(written, 2);

        let entries: Vec<_> = std::fs::read_dir(&dir)
            .expect("backtest dir should exist")
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
            .collect();
        assert_eq!(entries.len(), 2);
        for entry in &entries {
            let payload = std::fs::read_to_string(entry.path()).expect("artifact readable");
            assert!(payload.contains(crate::validation::CANONICAL_BACKTEST_ARTIFACT_KIND));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_walkforward_validation_artifacts_writes_one_file_per_strategy() {
        let dir = temp_dir("walkforward-validations");
        let result = DiscoveryResult {
            portfolio: vec![profitable_gene("alpha-1")],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["signal".to_string()],
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: vec![sample_walkforward_validation_artifact(
                "fnv64:0011223344556677",
            )],
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        };

        let written = save_walkforward_validation_artifacts(&dir, &result)
            .expect("walk-forward validation artifacts should persist");
        assert_eq!(written, 1);

        let entries: Vec<_> = std::fs::read_dir(&dir)
            .expect("walkforward dir should exist")
            .filter_map(|entry| entry.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        let payload = std::fs::read_to_string(entries[0].path()).expect("artifact readable");
        assert!(payload.contains(crate::validation::WALKFORWARD_VALIDATION_ARTIFACT_KIND));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_canonical_backtest_artifacts_skips_when_empty() {
        let dir = temp_dir("canonical-backtests-empty");
        let result = DiscoveryResult {
            portfolio: Vec::new(),
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: Vec::new(),
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        };

        let written = save_canonical_backtest_artifacts(&dir, &result)
            .expect("empty canonical backtest list should be a no-op");
        assert_eq!(written, 0);
        assert!(!dir.exists());
    }

    #[test]
    fn artifact_filename_strips_invalid_characters() {
        let name = artifact_filename_for_strategy_hash("fnv64:abc123", 0);
        assert!(!name.contains(':'));
        assert!(name.ends_with(".json"));
        assert!(name.contains("abc123"));
    }

    #[test]
    fn discovery_runtime_overrides_defaults_match_legacy_env_defaults() {
        let defaults = DiscoveryRuntimeOverrides::default();
        assert_eq!(defaults.prefilter_top_k, 50);
        assert!((defaults.prefilter_insample_frac - 0.70).abs() < 1e-9);
        assert!((defaults.funnel_stage1_pct - 0.25).abs() < 1e-9);
    }

    #[test]
    fn discovery_runtime_overrides_clamp_invalid_values() {
        let overrides = DiscoveryRuntimeOverrides {
            prefilter_top_k: 0,
            prefilter_insample_frac: f64::NAN,
            funnel_stage1_pct: 5.0,
        };
        assert!((overrides.resolved_prefilter_insample_frac() - 0.70).abs() < 1e-9);
        assert!((overrides.resolved_funnel_stage1_pct() - 1.0).abs() < 1e-9);

        let too_small = DiscoveryRuntimeOverrides {
            prefilter_top_k: 0,
            prefilter_insample_frac: 0.0,
            funnel_stage1_pct: 0.0001,
        };
        assert!((too_small.resolved_prefilter_insample_frac() - 0.70).abs() < 1e-9);
        assert!((too_small.resolved_funnel_stage1_pct() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn default_discovery_config_does_not_read_environment() {
        // Sanity guard: the default config should be deterministic regardless
        // of the legacy env vars set by other test runners.
        let cfg = DiscoveryConfig::default();
        assert_eq!(
            cfg.runtime_overrides,
            DiscoveryRuntimeOverrides::default(),
            "default DiscoveryConfig must not pick up legacy env overrides"
        );
    }

    #[test]
    fn discovery_profile_exports_runtime_override_resolution() {
        let mut config = DiscoveryConfig::default();
        config.runtime_overrides = DiscoveryRuntimeOverrides {
            prefilter_top_k: 17,
            prefilter_insample_frac: 0.6,
            funnel_stage1_pct: 0.5,
        };
        let result = DiscoveryResult {
            portfolio: vec![profitable_gene("alpha-1")],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: Vec::new(),
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        };

        let profile = build_discovery_profile(&config, &result);
        assert_eq!(profile.prefilter_top_k, 17);
        assert!((profile.prefilter_insample_frac - 0.6).abs() < 1e-9);
        assert!((profile.funnel_stage1_pct - 0.5).abs() < 1e-9);
    }

    #[test]
    fn compute_discovery_forward_test_artifacts_returns_empty_for_empty_portfolio() {
        let config = DiscoveryConfig::default();
        let features = sample_feature_frame();
        let ohlcv = sample_ohlcv();
        let artifacts = compute_discovery_forward_test_artifacts(
            &[],
            &features.names,
            &features,
            &ohlcv,
            &config,
        )
        .expect("empty portfolio should produce zero artifacts");
        assert!(artifacts.is_empty());
    }

    #[test]
    fn compute_discovery_forward_test_artifacts_rejects_tails_missing_features() {
        let config = DiscoveryConfig::default();
        let portfolio = vec![profitable_gene("alpha-1")];
        let mut tail_features = sample_feature_frame();
        tail_features.names = vec!["unrelated_feature".to_string()];
        let err = compute_discovery_forward_test_artifacts(
            &portfolio,
            &["signal".to_string()],
            &tail_features,
            &sample_ohlcv(),
            &config,
        )
        .expect_err("tail without the effective feature must be rejected");
        assert!(err.to_string().contains("missing feature 'signal'"));
    }

    #[test]
    fn compute_discovery_forward_test_artifacts_produces_one_artifact_per_strategy() {
        let mut config = DiscoveryConfig::default();
        config.runtime_overrides.prefilter_top_k = 0;
        let portfolio = vec![profitable_gene("alpha-1"), profitable_gene("alpha-2")];
        let features = sample_feature_frame();
        let ohlcv = sample_ohlcv();
        let artifacts = compute_discovery_forward_test_artifacts(
            &portfolio,
            &features.names,
            &features,
            &ohlcv,
            &config,
        )
        .expect("forward-test artifacts should build for in-band tail");
        assert_eq!(artifacts.len(), portfolio.len());
        for artifact in &artifacts {
            assert_eq!(
                artifact.artifact_kind,
                crate::validation::FORWARD_TEST_VALIDATION_ARTIFACT_KIND
            );
            assert!(artifact.summary.bars > 0);
            assert!(!artifact.scope.strategy_hash.is_empty());
        }
    }

    #[test]
    fn save_forward_test_validation_artifacts_writes_one_file_per_strategy() {
        let dir = temp_dir("forward-test-validations");
        let config = DiscoveryConfig::default();
        let portfolio = vec![profitable_gene("alpha-1")];
        let features = sample_feature_frame();
        let ohlcv = sample_ohlcv();
        let artifacts = compute_discovery_forward_test_artifacts(
            &portfolio,
            &features.names,
            &features,
            &ohlcv,
            &config,
        )
        .expect("forward-test artifacts should build");

        let result = DiscoveryResult {
            portfolio,
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: features.names.clone(),
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: artifacts,
            prop_firm_validation_artifacts: Vec::new(),
        };

        let written = save_forward_test_validation_artifacts(&dir, &result)
            .expect("forward-test artifacts should persist");
        assert_eq!(written, 1);

        let entries: Vec<_> = std::fs::read_dir(&dir)
            .expect("forward-test dir should exist")
            .filter_map(|entry| entry.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        let payload = std::fs::read_to_string(entries[0].path()).expect("artifact readable");
        assert!(payload.contains(crate::validation::FORWARD_TEST_VALIDATION_ARTIFACT_KIND));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn discovery_profile_exports_forward_test_artifact_count() {
        let config = DiscoveryConfig::default();
        let temporal = discovery_temporal_contract(&config, &["signal".to_string()])
            .expect("temporal contract for default discovery config");
        let scope = ForwardTestValidationScope::new("dataset", "eval", "strategy", &temporal);
        let summary = crate::validation::ForwardTestSummary {
            bars: 5,
            metrics: BacktestMetrics::from_metric_array([0.0; 11]),
            span_days: 0.0,
        };
        let mut result = DiscoveryResult {
            portfolio: vec![profitable_gene("alpha-1")],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["signal".to_string()],
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: vec![ForwardTestValidationArtifactFile::new(
                scope, summary,
            )],
            prop_firm_validation_artifacts: Vec::new(),
        };
        result.validation_gates.walkforward_passed = true;
        result.validation_gates.cpcv_passed = true;

        let profile = build_discovery_profile(&config, &result);
        assert_eq!(profile.forward_test_validation_artifacts_observed, 1);
    }

    fn forward_test_artifact_with_metrics(
        strategy_hash: &str,
        net_profit: f64,
        trade_count: usize,
    ) -> ForwardTestValidationArtifactFile {
        let config = DiscoveryConfig::default();
        let temporal = discovery_temporal_contract(&config, &["signal".to_string()])
            .expect("temporal contract for default discovery config");
        let scope = ForwardTestValidationScope::new("dataset", "eval", strategy_hash, &temporal);
        let mut metrics_array = [0.0_f64; 11];
        metrics_array[0] = net_profit; // net_profit
        metrics_array[8] = trade_count as f64; // trade_count
        let summary = crate::validation::ForwardTestSummary {
            bars: 5,
            metrics: BacktestMetrics::from_metric_array(metrics_array),
            span_days: 0.0,
        };
        ForwardTestValidationArtifactFile::new(scope, summary)
    }

    fn empty_discovery_result_with_gates(
        walkforward_passed: bool,
        cpcv_passed: bool,
    ) -> DiscoveryResult {
        let mut gates = DiscoveryValidationGates::pending();
        gates.walkforward_passed = walkforward_passed;
        gates.cpcv_passed = cpcv_passed;
        DiscoveryResult {
            portfolio: Vec::new(),
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: Vec::new(),
            validation_gates: gates,
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
        }
    }

    #[test]
    fn evidence_bridge_mirrors_discovery_validation_gates_with_no_forward_test_artifacts() {
        let result = empty_discovery_result_with_gates(true, true);
        let evidence = live_validation_evidence_from_discovery(&result);
        assert!(evidence.walkforward_passed);
        assert!(evidence.cpcv_passed);
        assert_eq!(evidence.forward_test_passed, None);
        assert_eq!(evidence.prop_firm_passed, None);
        assert!(evidence.live_sim_runtime_model_hash.is_none());
    }

    #[test]
    fn evidence_bridge_marks_forward_test_passed_when_every_artifact_is_profitable() {
        let mut result = empty_discovery_result_with_gates(true, true);
        result.forward_test_validation_artifacts = vec![
            forward_test_artifact_with_metrics("fnv64:abc", 25.0, 3),
            forward_test_artifact_with_metrics("fnv64:def", 10.0, 1),
        ];
        let evidence = live_validation_evidence_from_discovery(&result);
        assert_eq!(evidence.forward_test_passed, Some(true));
    }

    #[test]
    fn evidence_bridge_marks_forward_test_failed_when_any_artifact_is_unprofitable() {
        let mut result = empty_discovery_result_with_gates(true, true);
        result.forward_test_validation_artifacts = vec![
            forward_test_artifact_with_metrics("fnv64:abc", 25.0, 3),
            forward_test_artifact_with_metrics("fnv64:def", -10.0, 2),
        ];
        let evidence = live_validation_evidence_from_discovery(&result);
        assert_eq!(evidence.forward_test_passed, Some(false));
    }

    #[test]
    fn evidence_bridge_marks_forward_test_failed_when_artifact_has_zero_trades() {
        let mut result = empty_discovery_result_with_gates(true, true);
        result.forward_test_validation_artifacts =
            vec![forward_test_artifact_with_metrics("fnv64:abc", 5.0, 0)];
        let evidence = live_validation_evidence_from_discovery(&result);
        assert_eq!(evidence.forward_test_passed, Some(false));
    }

    #[test]
    fn evidence_bridge_propagates_failed_walkforward_and_cpcv() {
        let result = empty_discovery_result_with_gates(false, false);
        let evidence = live_validation_evidence_from_discovery(&result);
        assert!(!evidence.walkforward_passed);
        assert!(!evidence.cpcv_passed);
    }

    fn prop_firm_artifact_with_pass_flag(
        strategy_hash: &str,
        all_rules_passed: bool,
    ) -> PropFirmRiskValidationArtifactFile {
        let config = DiscoveryConfig::default();
        let temporal = discovery_temporal_contract(&config, &["signal".to_string()])
            .expect("temporal contract for default discovery config");
        let rules = PropFirmRiskRules::default();
        let scope =
            PropFirmRiskValidationScope::new("dataset", "eval", strategy_hash, &rules, &temporal)
                .expect("scope construction should succeed");
        let summary = crate::validation::PropFirmRiskValidationSummary {
            rules,
            trades_observed: 0,
            trading_days_observed: 0,
            max_daily_loss_pct_observed: 0.0,
            max_overall_drawdown_pct_observed: 0.0,
            largest_profit_share_observed: 0.0,
            max_trades_per_day_observed: 0,
            net_return_pct: 0.0,
            daily_loss_breach: false,
            overall_drawdown_breach: false,
            consistency_violation: false,
            trade_limit_violation: false,
            min_trading_days_ok: true,
            profit_target_met: true,
            all_rules_passed,
        };
        PropFirmRiskValidationArtifactFile::new(scope, summary)
    }

    #[test]
    fn evidence_bridge_marks_prop_firm_passed_when_every_artifact_passes() {
        let mut result = empty_discovery_result_with_gates(true, true);
        result.prop_firm_validation_artifacts = vec![
            prop_firm_artifact_with_pass_flag("fnv64:abc", true),
            prop_firm_artifact_with_pass_flag("fnv64:def", true),
        ];
        let evidence = live_validation_evidence_from_discovery(&result);
        assert_eq!(evidence.prop_firm_passed, Some(true));
    }

    #[test]
    fn evidence_bridge_marks_prop_firm_failed_when_any_artifact_fails() {
        let mut result = empty_discovery_result_with_gates(true, true);
        result.prop_firm_validation_artifacts = vec![
            prop_firm_artifact_with_pass_flag("fnv64:abc", true),
            prop_firm_artifact_with_pass_flag("fnv64:def", false),
        ];
        let evidence = live_validation_evidence_from_discovery(&result);
        assert_eq!(evidence.prop_firm_passed, Some(false));
    }

    #[test]
    fn compute_discovery_prop_firm_artifacts_returns_empty_for_empty_portfolio() {
        let config = DiscoveryConfig::default();
        let features = sample_feature_frame();
        let ohlcv = sample_ohlcv();
        let artifacts = compute_discovery_prop_firm_artifacts(
            &[],
            &features.names,
            &features,
            &ohlcv,
            &config,
            PropFirmRiskRules::default(),
        )
        .expect("empty portfolio should produce zero artifacts");
        assert!(artifacts.is_empty());
    }

    #[test]
    fn compute_discovery_prop_firm_artifacts_rejects_tails_missing_features() {
        let config = DiscoveryConfig::default();
        let portfolio = vec![profitable_gene("alpha-1")];
        let mut tail_features = sample_feature_frame();
        tail_features.names = vec!["unrelated_feature".to_string()];
        let err = compute_discovery_prop_firm_artifacts(
            &portfolio,
            &["signal".to_string()],
            &tail_features,
            &sample_ohlcv(),
            &config,
            PropFirmRiskRules::default(),
        )
        .expect_err("tail without the effective feature must be rejected");
        assert!(err.to_string().contains("missing feature 'signal'"));
    }

    #[test]
    fn compute_discovery_prop_firm_artifacts_produces_one_artifact_per_strategy() {
        let mut config = DiscoveryConfig::default();
        config.runtime_overrides.prefilter_top_k = 0;
        let portfolio = vec![profitable_gene("alpha-1"), profitable_gene("alpha-2")];
        let features = sample_feature_frame();
        let ohlcv = sample_ohlcv();
        let artifacts = compute_discovery_prop_firm_artifacts(
            &portfolio,
            &features.names,
            &features,
            &ohlcv,
            &config,
            PropFirmRiskRules::default(),
        )
        .expect("prop-firm artifacts should build");
        assert_eq!(artifacts.len(), portfolio.len());
        for artifact in &artifacts {
            assert_eq!(
                artifact.artifact_kind,
                crate::validation::PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND
            );
            assert!(!artifact.scope.strategy_hash.is_empty());
        }
    }

    #[test]
    fn save_prop_firm_validation_artifacts_writes_one_file_per_strategy() {
        let dir = temp_dir("prop-firm-validations");
        let result = DiscoveryResult {
            portfolio: vec![profitable_gene("alpha-1")],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["signal".to_string()],
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: vec![prop_firm_artifact_with_pass_flag(
                "fnv64:abc",
                true,
            )],
        };

        let written = save_prop_firm_validation_artifacts(&dir, &result)
            .expect("prop-firm artifacts should persist");
        assert_eq!(written, 1);

        let entries: Vec<_> = std::fs::read_dir(&dir)
            .expect("prop-firm dir should exist")
            .filter_map(|entry| entry.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        let payload = std::fs::read_to_string(entries[0].path()).expect("artifact readable");
        assert!(payload.contains(crate::validation::PROP_FIRM_RISK_VALIDATION_ARTIFACT_KIND));

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn populated_discovery_result(
        canonical_count: usize,
        walkforward_count: usize,
        forward_test_count: usize,
        prop_firm_count: usize,
    ) -> DiscoveryResult {
        DiscoveryResult {
            portfolio: vec![profitable_gene("alpha-1")],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec!["signal".to_string()],
            validation_gates: DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: (0..canonical_count)
                .map(|idx| sample_canonical_backtest_artifact(&format!("canonical-{idx}")))
                .collect(),
            walkforward_validation_artifacts: (0..walkforward_count)
                .map(|idx| sample_walkforward_validation_artifact(&format!("walkforward-{idx}")))
                .collect(),
            forward_test_validation_artifacts: (0..forward_test_count)
                .map(|idx| forward_test_artifact_with_metrics(&format!("forward-{idx}"), 1.0, 1))
                .collect(),
            prop_firm_validation_artifacts: (0..prop_firm_count)
                .map(|idx| prop_firm_artifact_with_pass_flag(&format!("prop-{idx}"), true))
                .collect(),
        }
    }

    #[test]
    fn discovery_validation_evidence_manifest_rejects_missing_live_sim_evidence() {
        let result = populated_discovery_result(1, 1, 1, 1);
        let err = discovery_validation_evidence_manifest(&result)
            .expect_err("manifest must surface missing live-sim evidence");
        assert!(err.to_string().contains("live_execution_simulation_hash"));
    }

    #[test]
    fn discovery_validation_evidence_manifest_rejects_missing_walkforward_evidence() {
        let result = populated_discovery_result(1, 0, 1, 1);
        let err = discovery_validation_evidence_manifest(&result)
            .expect_err("manifest must surface missing walkforward evidence");
        assert!(err.to_string().contains("walkforward_validation_hash"));
    }

    #[test]
    fn discovery_per_kind_evidence_hashes_returns_some_only_for_present_kinds() {
        let result = populated_discovery_result(1, 0, 1, 1);
        let hashes = discovery_per_kind_evidence_hashes(&result)
            .expect("per-kind hash extraction should succeed");
        assert!(hashes.canonical_backtest.is_some());
        assert!(hashes.walkforward.is_none());
        assert!(hashes.forward_test.is_some());
        assert!(hashes.prop_firm.is_some());
        assert!(hashes.live_execution_simulation.is_none());
    }

    #[test]
    fn discovery_per_kind_evidence_hashes_returns_none_for_empty_result() {
        let result = populated_discovery_result(0, 0, 0, 0);
        let hashes = discovery_per_kind_evidence_hashes(&result)
            .expect("per-kind hash extraction should succeed");
        assert!(hashes.canonical_backtest.is_none());
        assert!(hashes.walkforward.is_none());
        assert!(hashes.forward_test.is_none());
        assert!(hashes.prop_firm.is_none());
        assert!(hashes.live_execution_simulation.is_none());
    }

    #[test]
    fn lossy_manifest_accepts_complete_producer_side_evidence() {
        let result = populated_discovery_result(1, 1, 1, 1);
        let manifest = discovery_validation_evidence_manifest_excluding_live_sim(&result)
            .expect("lossy manifest should accept complete producer-side evidence");
        assert!(
            manifest
                .live_execution_simulation_hash
                .starts_with("deferred:")
        );
    }

    #[test]
    fn lossy_manifest_still_rejects_missing_producer_side_evidence() {
        let result = populated_discovery_result(1, 0, 1, 1);
        let err = discovery_validation_evidence_manifest_excluding_live_sim(&result)
            .expect_err("lossy manifest must still reject missing walk-forward");
        assert!(err.to_string().contains("walkforward_validation_hash"));
    }

    #[test]
    fn all_producer_kinds_present_ignores_live_sim() {
        let hashes = DiscoveryPerKindEvidenceHashes {
            canonical_backtest: Some("h1".into()),
            walkforward: Some("h2".into()),
            forward_test: Some("h3".into()),
            prop_firm: Some("h4".into()),
            live_execution_simulation: None,
        };
        assert!(hashes.all_producer_kinds_present());
        assert!(!hashes.all_present());
    }

    #[test]
    fn full_validation_chain_with_complete_producer_evidence_passes_lossy_manifest() {
        // Build a result with all four producer-side artifact kinds populated.
        let result = populated_discovery_result(2, 1, 1, 2);

        // 1. Per-kind hashes know which kinds are present.
        let hashes = discovery_per_kind_evidence_hashes(&result)
            .expect("per-kind hash extraction should succeed");
        assert!(hashes.canonical_backtest.is_some());
        assert!(hashes.walkforward.is_some());
        assert!(hashes.forward_test.is_some());
        assert!(hashes.prop_firm.is_some());
        assert!(hashes.live_execution_simulation.is_none());
        assert!(hashes.all_producer_kinds_present());
        assert!(!hashes.all_present()); // live-sim missing keeps full check off

        // 2. Strict manifest rejects on missing live-sim.
        let strict_err = discovery_validation_evidence_manifest(&result)
            .expect_err("strict manifest must reject when live-sim hash is empty");
        assert!(strict_err.to_string().contains("live_execution_simulation"));

        // 3. Lossy manifest accepts the same result.
        let lossy = discovery_validation_evidence_manifest_excluding_live_sim(&result)
            .expect("lossy manifest accepts complete producer-side evidence");
        assert!(
            lossy
                .live_execution_simulation_hash
                .starts_with("deferred:")
        );

        // 4. Evidence bridge surfaces the producer-side outcomes.
        let mut result_for_evidence = result.clone();
        result_for_evidence.validation_gates.walkforward_passed = true;
        result_for_evidence.validation_gates.cpcv_passed = true;
        let evidence = live_validation_evidence_from_discovery(&result_for_evidence);
        assert!(evidence.walkforward_passed);
        assert!(evidence.cpcv_passed);
        assert_eq!(evidence.forward_test_passed, Some(true));
        assert_eq!(evidence.prop_firm_passed, Some(true));
        assert!(evidence.live_sim_runtime_model_hash.is_none());

        // 5. Profile carries the same data without re-deriving anything.
        let profile = build_discovery_profile(&DiscoveryConfig::default(), &result_for_evidence);
        // The Phase 49 prop-firm count IS sourced from the artifact
        // vector directly (not from validation_gates), so it should
        // reflect the constructed fixture.
        assert_eq!(profile.prop_firm_validation_artifacts_observed, 2);
        assert_eq!(profile.forward_test_validation_artifacts_observed, 1);
        assert!(!profile.validation_evidence_complete); // live-sim still missing
        assert!(
            profile
                .validation_evidence_missing_kinds
                .iter()
                .any(|k| k == "live_execution_simulation")
        );
        // Producer-side completeness is true (all four kinds present).
        assert!(
            profile
                .validation_evidence_hashes
                .all_producer_kinds_present()
        );
    }

    #[test]
    fn discovery_run_profile_records_typed_determinism_policy() {
        // The OnceLock-installed determinism policy may carry whatever
        // any earlier test in this process installed, so we assert only
        // that the profile carries one of the three legal variants —
        // every one of which is serializable, which is the property the
        // promotion-readiness runbook documents.
        let config = DiscoveryConfig::default();
        let result = populated_discovery_result(0, 0, 0, 0);
        let profile = build_discovery_profile(&config, &result);
        match profile.determinism_policy {
            DeterminismPolicy::Deterministic { seed: _ }
            | DeterminismPolicy::BestEffort
            | DeterminismPolicy::NonDeterministicAllowed => {}
        }
    }

    #[test]
    fn discovery_run_profile_exposes_validation_evidence_hashes_and_missing_kinds() {
        let config = DiscoveryConfig::default();
        let result = populated_discovery_result(1, 0, 1, 1);
        let profile = build_discovery_profile(&config, &result);
        assert!(
            profile
                .validation_evidence_hashes
                .canonical_backtest
                .is_some()
        );
        assert!(profile.validation_evidence_hashes.walkforward.is_none());
        assert!(profile.validation_evidence_hashes.forward_test.is_some());
        assert!(profile.validation_evidence_hashes.prop_firm.is_some());
        assert!(
            profile
                .validation_evidence_hashes
                .live_execution_simulation
                .is_none()
        );
        assert!(!profile.validation_evidence_complete);
        assert!(
            profile
                .validation_evidence_missing_kinds
                .iter()
                .any(|k| k == "walkforward")
        );
        assert!(
            profile
                .validation_evidence_missing_kinds
                .iter()
                .any(|k| k == "live_execution_simulation")
        );
        assert_eq!(profile.prop_firm_validation_artifacts_observed, 1);
    }
}
