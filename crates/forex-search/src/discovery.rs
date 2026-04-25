use crate::genetic::strategy_gene::EvaluationConfig;
use crate::genetic::{Gene, evolve_search_with_progress_and_limits, signals_for_gene};
use crate::quality::{StrategyMetrics, StrategyQualityAnalyzer, Trade};
use anyhow::Result;
use chrono::{Datelike, TimeZone, Utc};
use forex_data::{FeatureFrame, Ohlcv};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

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
        }
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

fn discovery_backtest_settings(gene: &Gene) -> crate::eval::BacktestSettings {
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
    let prefilter_top_k = std::env::var("FOREX_BOT_PREFILTER_TOP_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50);

    if prefilter_top_k > 0 && features.names.len() > prefilter_top_k {
        features = prefilter_features(&features, &ohlcv, prefilter_top_k);
    }

    // Multi-stage Funnel: Stage 1 (Fast Evaluation)
    let stage1_pct = std::env::var("FOREX_BOT_FUNNEL_STAGE1_PCT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.25)
        .clamp(0.01, 1.0);

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

    finalize_candidates_with_progress(search.genes, &features, &ohlcv, config, progress_fn)
}

fn slice_features(features: &FeatureFrame, keep_ratio: f64) -> FeatureFrame {
    if keep_ratio >= 1.0 || keep_ratio <= 0.0 {
        return features.clone();
    }
    let total = features.data.nrows();
    let start = total.saturating_sub((total as f64 * keep_ratio) as usize);
    if start == 0 {
        return features.clone();
    }
    FeatureFrame {
        timestamps: features.timestamps[start..].to_vec(),
        names: features.names.clone(),
        data: features.data.slice(ndarray::s![start.., ..]).to_owned(),
    }
}

fn slice_ohlcv_by_ratio(ohlcv: &Ohlcv, keep_ratio: f64) -> Ohlcv {
    if keep_ratio >= 1.0 || keep_ratio <= 0.0 {
        return ohlcv.clone();
    }
    let total = ohlcv.close.len();
    let start = total.saturating_sub((total as f64 * keep_ratio) as usize);
    if start == 0 {
        return ohlcv.clone();
    }
    Ohlcv {
        timestamp: ohlcv.timestamp.as_ref().map(|ts| ts[start..].to_vec()),
        open: ohlcv.open[start..].to_vec(),
        high: ohlcv.high[start..].to_vec(),
        low: ohlcv.low[start..].to_vec(),
        close: ohlcv.close[start..].to_vec(),
        volume: ohlcv.volume.as_ref().map(|v| v[start..].to_vec()),
    }
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

fn prefilter_features(features: &FeatureFrame, ohlcv: &Ohlcv, top_k: usize) -> FeatureFrame {
    let n_rows = features.data.nrows();
    let n_cols = features.data.ncols();
    if n_rows < 2 || n_cols <= top_k {
        return features.clone();
    }

    // Calculate 1-bar forward returns
    let mut returns = vec![0.0f32; n_rows];
    for i in 0..(n_rows - 1) {
        let ret = (ohlcv.close[i + 1] - ohlcv.close[i]) / ohlcv.close[i];
        returns[i] = ret as f32;
    }

    let mut correlations = Vec::with_capacity(n_cols);
    for col_idx in 0..n_cols {
        let name = &features.names[col_idx];
        if name.starts_with("regime_") {
            // Force keep regime columns by giving them infinite correlation
            correlations.push((col_idx, f32::INFINITY));
        } else {
            let col = features.data.column(col_idx);
            let corr = pearson_correlation(col.to_slice().unwrap_or(&col.to_vec()), &returns);
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

fn validate_regime_robustness(trades: &[crate::quality::Trade], features: &FeatureFrame) -> bool {
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

    // We reject if the strategy wipes out > 3% of account in ANY specific regime
    // Assuming 100k account balance, 3% is 3000.
    let limit = -3000.0;

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
    let mut filtered: Vec<(usize, Gene)> = Vec::new();
    let mut signals_map = Vec::new();
    for (candidate_idx, gene) in &ranked_candidates {
        if !gene.passes_filter(&config.filtering) {
            continue;
        }
        let sig = signals_for_gene(features, gene);
        let trade_count = sig.iter().filter(|v| **v != 0).count() as f64;
        if trade_count >= min_trades as f64 {
            filtered.push((*candidate_idx, gene.clone()));
            signals_map.push(sig);
        }
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
        let mut strict_passed: Vec<QualityCandidate> = Vec::new();
        let mut opportunistic_passed = 0usize;

        for ((candidate_idx, gene), sig) in filtered.into_iter().zip(signals_map.into_iter()) {
            let trades = crate::eval::simulate_trades_core(
                &ohlcv.close,
                &ohlcv.high,
                &ohlcv.low,
                &features.timestamps,
                &sig,
                &discovery_backtest_settings(&gene),
            );
            let initial_balance = std::env::var("FOREX_BOT_BACKTEST_INITIAL_EQUITY")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(100_000.0);
            let metrics = analyzer.analyze_strategy(&gene.strategy_id, &trades, initial_balance);
            let strict_quality = passes_strict_quality(&metrics, &config.filtering);
            let opportunistic_quality =
                !strict_quality && passes_opportunistic_quality(&metrics, &config.filtering);

            if strict_quality || opportunistic_quality {
                // Regime-Aware Validation (Idea #3.2)
                let regime_robust = validate_regime_robustness(&trades, features);
                if !regime_robust {
                    continue; // Skip strategy, it fails massively in a specific regime!
                }

                // Monte Carlo Parameter Perturbation Test (100 runs)
                let mc_runs = 100;
                let profitable_runs: usize = (0..mc_runs)
                    .into_par_iter()
                    .map(|_| {
                        use rand::Rng;
                        let mut rng = rand::rng();
                        let mut perturbed = gene.clone();

                        // Thresholds ±15%
                        perturbed.long_threshold *= 1.0 + rng.random_range(-0.15..=0.15);
                        perturbed.short_threshold *= 1.0 + rng.random_range(-0.15..=0.15);

                        // Weights ±20%
                        for w in &mut perturbed.weights {
                            *w *= 1.0 + rng.random_range(-0.20..=0.20);
                        }

                        // SL/TP ±25%
                        if perturbed.sl_pips.is_finite() && perturbed.sl_pips > 0.0 {
                            perturbed.sl_pips *= 1.0 + rng.random_range(-0.25..=0.25);
                        }
                        if perturbed.tp_pips.is_finite() && perturbed.tp_pips > 0.0 {
                            perturbed.tp_pips *= 1.0 + rng.random_range(-0.25..=0.25);
                        }

                        let p_sig = crate::genetic::signals_for_gene(features, &perturbed);
                        let p_trades = crate::eval::simulate_trades_core(
                            &ohlcv.close,
                            &ohlcv.high,
                            &ohlcv.low,
                            &features.timestamps,
                            &p_sig,
                            &discovery_backtest_settings(&perturbed),
                        );
                        let pnl: f64 = p_trades.iter().map(|t| t.pnl).sum();
                        if pnl > 0.0 { 1 } else { 0 }
                    })
                    .sum();

                // Require at least 70% of perturbed variations to be profitable
                if profitable_runs < 70 {
                    continue; // Strategy is too fragile
                }

                // Spread/Slippage Sensitivity Test
                let mut sensitive_settings = discovery_backtest_settings(&gene);
                sensitive_settings.spread_pips = 2.0; // Test with 2.0 spread
                sensitive_settings.commission_per_trade = 7.0; // Baseline commission

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
                    continue; // Strategy becomes unprofitable at 2.0 spread
                }

                if opportunistic_quality {
                    opportunistic_passed += 1;
                }
                quality_metrics.push(metrics.clone());
                strict_passed.push((
                    candidate_idx,
                    gene,
                    sig,
                    metrics,
                    opportunistic_quality,
                    trades,
                ));
            }
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
    for ((_, gene), sig) in filtered.into_iter().zip(signals_map.into_iter()) {
        if portfolio.len() >= config.portfolio_size {
            break;
        }
        let mut ok = true;
        for existing in &portfolio_signals {
            let pearson = pearson_corr_i8(&sig, existing);
            // DS-2: also check Spearman to catch non-linear dependencies
            let spearman = spearman_corr_i8(&sig, existing);
            // Reject if EITHER correlation exceeds threshold
            if pearson.abs() >= config.corr_threshold
                || spearman.abs() >= config.corr_threshold
            {
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
        if let Some(dt) = Utc.timestamp_millis_opt(*ts).single() {
            if dt.weekday().num_days_from_monday() < 5 {
                let key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
                days.insert(key);
            }
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

pub fn save_portfolio_json(
    path: impl AsRef<Path>,
    portfolio: &[Gene],
    feature_names: &[String],
) -> Result<()> {
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
    let payload = serde_json::to_string_pretty(&exports)?;
    fs::write(path, payload)?;
    Ok(())
}

pub fn save_quality_report_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let payload = serde_json::to_string_pretty(&result.quality_metrics)?;
    fs::write(path, payload)?;
    Ok(())
}

pub fn save_trade_log_json(path: impl AsRef<Path>, result: &DiscoveryResult) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let payload = serde_json::to_string_pretty(&result.logged_trades)?;
    fs::write(path, payload)?;
    Ok(())
}

pub fn build_discovery_profile(
    config: &DiscoveryConfig,
    result: &DiscoveryResult,
) -> DiscoveryRunProfile {
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
    }
}

pub fn save_discovery_profile_json(
    path: impl AsRef<Path>,
    config: &DiscoveryConfig,
    result: &DiscoveryResult,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let payload = serde_json::to_string_pretty(&build_discovery_profile(config, result))?;
    fs::write(path, payload)?;
    Ok(())
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

    #[test]
    fn empty_portfolio_is_an_explicit_error() {
        let result = DiscoveryResult {
            portfolio: Vec::new(),
            candidates: vec![Gene::default()],
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
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

        let result =
            finalize_candidates_with_progress(candidates, &features, &ohlcv, &config, |event| {
                progress_events.push(event)
            })
            .expect("candidate finalization should succeed");

        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.portfolio.len(), 1);
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
}
