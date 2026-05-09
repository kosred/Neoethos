use crate::artifact_io::write_json_atomic;
use chrono::{Datelike, TimeZone, Utc};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

/// Typed replacement for the legacy `FOREX_BOT_PROP_MIN_TRADES_PER_MONTH`
/// and `FOREX_BOT_TRADING_DAYS_PER_MONTH` env vars. Previously read inline
/// inside monthly metric aggregation, both knobs change canonical strategy
/// quality scoring, so they belong in typed runtime config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityRuntimeOverrides {
    /// Minimum number of trades a calendar month must contain to count
    /// toward `monthly_win_rate` / `avg_return_pct`.
    pub min_trades_per_month: usize,
    /// Number of trading days per month used to convert observed trading
    /// days into a months-traded estimate.
    pub trading_days_per_month: f64,
}

impl Default for QualityRuntimeOverrides {
    fn default() -> Self {
        Self {
            min_trades_per_month: 4,
            trading_days_per_month: 21.0,
        }
    }
}

impl QualityRuntimeOverrides {
    /// One-shot read of the legacy `FOREX_BOT_PROP_*` quality env vars.
    pub fn from_env() -> Self {
        let mut overrides = Self::default();
        if let Some(value) = std::env::var("FOREX_BOT_PROP_MIN_TRADES_PER_MONTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            overrides.min_trades_per_month = value;
        }
        if let Some(value) = std::env::var("FOREX_BOT_TRADING_DAYS_PER_MONTH")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 1.0)
        {
            overrides.trading_days_per_month = value;
        }
        overrides
    }

    fn resolved_trading_days_per_month(&self) -> f64 {
        if self.trading_days_per_month.is_finite() && self.trading_days_per_month >= 1.0 {
            self.trading_days_per_month
        } else {
            21.0
        }
    }
}

static QUALITY_RUNTIME_OVERRIDES: OnceLock<QualityRuntimeOverrides> = OnceLock::new();

/// Install process-wide quality runtime overrides. Returns `Err(existing)`
/// if overrides were already installed earlier (first install wins).
pub fn install_quality_runtime_overrides(
    overrides: QualityRuntimeOverrides,
) -> Result<(), QualityRuntimeOverrides> {
    QUALITY_RUNTIME_OVERRIDES.set(overrides)
}

/// Convenience wrapper that resolves the legacy `FOREX_BOT_PROP_*` quality
/// env vars once and installs them. Idempotent.
pub fn install_quality_runtime_overrides_from_env() {
    let _ = QUALITY_RUNTIME_OVERRIDES.set(QualityRuntimeOverrides::from_env());
}

/// Returns the currently installed quality runtime overrides, or the
/// deterministic defaults when no install has happened.
pub fn current_quality_runtime_overrides() -> QualityRuntimeOverrides {
    QUALITY_RUNTIME_OVERRIDES.get().copied().unwrap_or_default()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub entry_time: i64,
    pub exit_time: Option<i64>,
    pub pnl: f64,
    pub pnl_pct: Option<f64>,
    pub duration_hours: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub strategy_id: String,
    pub total_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub total_return_pct: f64,
    pub avg_win_pct: f64,
    pub avg_loss_pct: f64,
    pub largest_win_pct: f64,
    pub largest_loss_pct: f64,
    pub max_drawdown_pct: f64,
    pub avg_drawdown_pct: f64,
    pub longest_losing_streak: usize,
    pub longest_winning_streak: usize,
    pub expectancy: f64,
    pub kelly_fraction: f64,
    pub statistical_significance: f64,
    pub monthly_win_rate: f64,
    pub positive_months: usize,
    pub negative_months: usize,
    pub avg_monthly_return_pct: f64,
    pub profit_per_trade: f64,
    pub avg_trade_duration_hours: f64,
    pub trades_per_month: f64,
    pub quality_score: f64,
    pub has_edge: bool,
    pub recommendation: String,
    pub mc_worst_drawdown_95_pct: Option<f64>,
    pub mc_risk_of_ruin_pct: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct StrategyQualityAnalyzer {
    pub min_sharpe: f64,
    pub min_sortino: f64,
    pub min_calmar: f64,
    pub min_profit_factor: f64,
    pub min_win_rate: f64,
    pub min_trades: usize,
    pub max_dd_acceptable: f64,
    pub min_monthly_return_pct: f64,
    pub edge_significance_pvalue: f64,
}

impl Default for StrategyQualityAnalyzer {
    fn default() -> Self {
        Self {
            min_sharpe: 1.2,
            min_sortino: 1.2,
            min_calmar: 1.0,
            min_profit_factor: 1.5,
            min_win_rate: 0.50,
            min_trades: 0,
            max_dd_acceptable: 0.15,
            min_monthly_return_pct: 0.04,
            edge_significance_pvalue: 0.01,
        }
    }
}

impl StrategyQualityAnalyzer {
    pub fn analyze_strategy(
        &self,
        strategy_id: &str,
        trades: &[Trade],
        initial_balance: f64,
    ) -> StrategyMetrics {
        if trades.is_empty() {
            return empty_metrics(strategy_id);
        }

        let mut pnls = Vec::with_capacity(trades.len());
        let mut returns = Vec::with_capacity(trades.len());
        let mut durations = Vec::with_capacity(trades.len());

        for trade in trades {
            let pnl_pct = trade.pnl_pct.unwrap_or(trade.pnl / initial_balance);
            pnls.push(trade.pnl);
            returns.push(pnl_pct);
            if let Some(dur) = trade.duration_hours {
                durations.push(dur);
            } else if let Some(exit) = trade.exit_time
                && trade.entry_time > 0
                && exit >= trade.entry_time
            {
                let hours = (exit - trade.entry_time) as f64 / 3_600_000.0;
                durations.push(hours);
            }
        }

        let total_trades = returns.len();
        let wins: Vec<f64> = returns.iter().cloned().filter(|v| *v > 0.0).collect();
        let losses: Vec<f64> = returns.iter().cloned().filter(|v| *v < 0.0).collect();

        let win_rate = if total_trades > 0 {
            wins.len() as f64 / total_trades as f64
        } else {
            0.0
        };

        let avg_win_pct = if !wins.is_empty() { mean(&wins) } else { 0.0 };
        let losses_cleaned: Vec<f64> = if losses.iter().any(|v| *v < 0.0) {
            losses.iter().cloned().filter(|v| *v < 0.0).collect()
        } else {
            returns.iter().map(|v| -v.abs()).collect()
        };
        let avg_loss_pct = if !losses_cleaned.is_empty() {
            mean(&losses_cleaned)
        } else {
            0.0
        };
        let avg_loss_mag = avg_loss_pct.abs();

        let gross_profit: f64 = pnls.iter().cloned().filter(|v| *v > 0.0).sum();
        let gross_loss: f64 = pnls
            .iter()
            .cloned()
            .filter(|v| *v < 0.0)
            .map(|v| v.abs())
            .sum();
        let eps = 1e-7;
        let mut profit_factor = (gross_profit + eps) / (gross_loss + eps);
        if profit_factor > 100.0 {
            profit_factor = 100.0;
        }

        let mut equity = initial_balance;
        let mut peak = initial_balance;
        let mut drawdowns = Vec::with_capacity(total_trades);
        for pnl in &pnls {
            equity += *pnl;
            if equity > peak {
                peak = equity;
            }
            let dd = if peak > 0.0 {
                (peak - equity) / peak
            } else {
                0.0
            };
            drawdowns.push(dd);
        }
        let max_dd = drawdowns.iter().cloned().fold(0.0, f64::max);
        let avg_dd = if !drawdowns.is_empty() {
            mean(&drawdowns)
        } else {
            0.0
        };

        let trades_per_month_raw = calculate_trade_frequency(trades);
        let trades_per_year = (trades_per_month_raw * 12.0).max(1.0);
        let sharpe = calculate_sharpe(&returns, trades_per_year);
        let sortino = calculate_sortino(&returns, trades_per_year);

        let total_return = pnls.iter().sum::<f64>();
        let total_return_pct = total_return / initial_balance;
        // A flawless equity curve (max_dd ≈ 0 with positive return) used to
        // rank Calmar=0 — i.e. worst — flipping the rank intent. Saturate the
        // ratio so a zero-DD profitable strategy ranks at the top of the sort,
        // then clamp so a single outlier can't dominate downstream weighting.
        let calmar = if max_dd > 1e-9 {
            (total_return_pct / max_dd).clamp(-1000.0, 1000.0)
        } else if total_return_pct > 0.0 {
            1000.0
        } else {
            0.0
        };

        let longest_win_streak = longest_streak(&pnls, true);
        let longest_loss_streak = longest_streak(&pnls, false);

        let expectancy = (win_rate * avg_win_pct) - ((1.0 - win_rate) * avg_loss_mag);
        let kelly = calculate_kelly(win_rate, avg_win_pct, avg_loss_mag);
        let p_value = test_statistical_significance(&returns);

        let monthly_metrics = analyze_monthly_consistency(trades, initial_balance);
        let monthly_win_rate = monthly_metrics.monthly_win_rate;
        let avg_monthly_return_pct = monthly_metrics.avg_return_pct;

        let avg_duration = if durations.is_empty() {
            0.0
        } else {
            mean(&durations)
        };
        let trades_per_month = trades_per_month_raw;

        // --- Monte Carlo Simulation (QA-2: block bootstrap on daily PnL) ---
        let mut rng = rand::rng();
        let mc_iterations = 1000;
        let mut worst_dds = Vec::with_capacity(mc_iterations);
        let mut ruined_count = 0;
        let ruin_threshold = initial_balance * 0.50;

        // Group trade PnLs by calendar day for block bootstrap
        let mut daily_pnl_blocks: std::collections::HashMap<i64, Vec<f64>> =
            std::collections::HashMap::new();
        for trade in trades {
            if trade.entry_time > 0 {
                let day_key = trade.entry_time / 86_400_000;
                daily_pnl_blocks.entry(day_key).or_default().push(trade.pnl);
            }
        }
        let mut day_blocks: Vec<Vec<f64>> = daily_pnl_blocks.into_values().collect();
        // Fallback to trade-level if fewer than 5 distinct days
        let use_blocks = day_blocks.len() >= 5;

        for _ in 0..mc_iterations {
            let shuffled_pnls: Vec<f64> = if use_blocks {
                day_blocks.shuffle(&mut rng);
                day_blocks.iter().flatten().copied().collect()
            } else {
                let mut p = pnls.clone();
                p.shuffle(&mut rng);
                p
            };

            let mut eq = initial_balance;
            let mut pk = initial_balance;
            let mut max_mc_dd = 0.0_f64;
            let mut ruined = false;

            for p in shuffled_pnls {
                eq += p;
                if eq < ruin_threshold {
                    ruined = true;
                }
                if eq > pk {
                    pk = eq;
                }
                let dd = if pk > 0.0 { (pk - eq) / pk } else { 0.0 };
                if dd > max_mc_dd {
                    max_mc_dd = dd;
                }
            }
            worst_dds.push(max_mc_dd);
            if ruined {
                ruined_count += 1;
            }
        }
        worst_dds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_idx = ((mc_iterations as f64 * 0.95) as usize).min(mc_iterations - 1);
        let mc_worst_dd_95 = worst_dds.get(p95_idx).cloned().unwrap_or(max_dd);
        let mc_risk_of_ruin = (ruined_count as f64) / (mc_iterations as f64);
        // -------------------------------------------------------------------

        let mut metrics = StrategyMetrics {
            strategy_id: strategy_id.to_string(),
            total_trades,
            win_rate,
            profit_factor,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            calmar_ratio: calmar,
            total_return_pct,
            avg_win_pct,
            avg_loss_pct,
            largest_win_pct: returns.iter().cloned().fold(0.0, f64::max),
            largest_loss_pct: returns.iter().cloned().fold(0.0, f64::min),
            max_drawdown_pct: max_dd,
            avg_drawdown_pct: avg_dd,
            longest_losing_streak: longest_loss_streak,
            longest_winning_streak: longest_win_streak,
            expectancy,
            kelly_fraction: kelly,
            statistical_significance: p_value,
            monthly_win_rate,
            positive_months: monthly_metrics.positive,
            negative_months: monthly_metrics.negative,
            avg_monthly_return_pct,
            profit_per_trade: if !pnls.is_empty() { mean(&pnls) } else { 0.0 },
            avg_trade_duration_hours: avg_duration,
            trades_per_month,
            quality_score: 0.0,
            has_edge: false,
            recommendation: String::new(),
            mc_worst_drawdown_95_pct: Some(mc_worst_dd_95),
            mc_risk_of_ruin_pct: Some(mc_risk_of_ruin),
        };

        score_strategy(self, &mut metrics);
        metrics
    }
}

#[derive(Debug, Clone)]
struct MonthlyMetrics {
    monthly_win_rate: f64,
    positive: usize,
    negative: usize,
    avg_return_pct: f64,
}

fn analyze_monthly_consistency(trades: &[Trade], initial_balance: f64) -> MonthlyMetrics {
    if trades.is_empty() {
        return MonthlyMetrics {
            monthly_win_rate: 0.0,
            positive: 0,
            negative: 0,
            avg_return_pct: 0.0,
        };
    }

    // Bucket per-month PnL AND per-month trade count so we can drop months
    // with too few trades (a month with 1 lucky trade should not get the same
    // weight in monthly_win_rate as a month with 50 trades). Threshold is
    // resolved from `QualityRuntimeOverrides::min_trades_per_month`.
    let min_trades_per_month = current_quality_runtime_overrides().min_trades_per_month;
    let mut monthly: HashMap<i64, (f64, usize)> = HashMap::new();
    for trade in trades {
        if trade.entry_time <= 0 {
            continue;
        }
        if let Some(dt) = Utc.timestamp_millis_opt(trade.entry_time).single() {
            let key = (dt.year() as i64) * 12 + dt.month() as i64;
            let entry = monthly.entry(key).or_insert((0.0, 0));
            entry.0 += trade.pnl;
            entry.1 += 1;
        }
    }

    if monthly.is_empty() {
        return MonthlyMetrics {
            monthly_win_rate: 0.0,
            positive: 0,
            negative: 0,
            avg_return_pct: 0.0,
        };
    }

    let mut positive = 0;
    let mut negative = 0;
    let mut sum = 0.0;
    let mut counted = 0usize;
    for &(pnl, n) in monthly.values() {
        if n < min_trades_per_month {
            continue;
        }
        sum += pnl;
        counted += 1;
        if pnl > 0.0 {
            positive += 1;
        } else {
            negative += 1;
        }
    }
    let total = counted;
    let avg_return_pct = if total > 0 {
        (sum / total as f64) / initial_balance
    } else {
        0.0
    };

    MonthlyMetrics {
        monthly_win_rate: if total > 0 {
            positive as f64 / total as f64
        } else {
            0.0
        },
        positive,
        negative,
        avg_return_pct,
    }
}

fn calculate_trade_frequency(trades: &[Trade]) -> f64 {
    if trades.is_empty() {
        return 0.0;
    }

    let mut days = std::collections::HashSet::new();
    for trade in trades {
        if trade.entry_time <= 0 {
            continue;
        }
        if let Some(dt) = Utc.timestamp_millis_opt(trade.entry_time).single()
            && dt.weekday().num_days_from_monday() < 5
        {
            let day_key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
            days.insert(day_key);
        }
    }

    if days.is_empty() {
        return 0.0;
    }

    let trading_days = days.len() as f64;
    let days_per_month = current_quality_runtime_overrides().resolved_trading_days_per_month();
    let months = (trading_days / days_per_month).max(1e-6);
    trades.len() as f64 / months
}

// QA-1: Annualize using actual trade frequency, not daily assumption.
// √trades_per_year is the correct annualization factor for per-trade returns.
fn calculate_sharpe(returns: &[f64], trades_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean_ret = mean(returns);
    let std_ret = stddev_sample(returns, mean_ret);
    if std_ret < 1e-9 {
        return 0.0;
    }
    let annualization = trades_per_year.max(1.0).sqrt();
    (mean_ret / std_ret) * annualization
}

fn calculate_sortino(returns: &[f64], trades_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean_ret = mean(returns);
    let downside: Vec<f64> = returns.iter().cloned().filter(|v| *v < 0.0).collect();
    if downside.len() < 2 {
        return 0.0;
    }
    let std_down = stddev_sample(&downside, 0.0);
    if std_down < 1e-9 {
        return 0.0;
    }
    let annualization = trades_per_year.max(1.0).sqrt();
    (mean_ret / std_down) * annualization
}

fn longest_streak(pnls: &[f64], win: bool) -> usize {
    let mut max_streak = 0;
    let mut current = 0;
    for pnl in pnls {
        let is_win = *pnl > 0.0;
        if (win && is_win) || (!win && !is_win) {
            current += 1;
            if current > max_streak {
                max_streak = current;
            }
        } else {
            current = 0;
        }
    }
    max_streak
}

fn calculate_kelly(win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
    if avg_loss < 1e-6 || win_rate <= 0.0 || win_rate >= 1.0 {
        return 0.0;
    }
    let b = avg_win / avg_loss;
    let p = win_rate;
    let q = 1.0 - p;
    let mut kelly = (p * b - q) / b;
    kelly = kelly.clamp(0.0, 1.0);
    kelly * 0.25
}

fn test_statistical_significance(returns: &[f64]) -> f64 {
    if returns.len() < 10 {
        return 1.0;
    }
    let mean_ret = mean(returns);
    let std_ret = stddev_sample(returns, mean_ret);
    if std_ret <= 0.0 {
        return 1.0;
    }
    let n = returns.len() as f64;
    let t_stat = mean_ret / (std_ret / n.sqrt());
    if t_stat <= 0.0 {
        return 1.0;
    }
    let df = n - 1.0;
    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    1.0 - dist.cdf(t_stat)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev_sample(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for v in values {
        let d = *v - mean;
        sum += d * d;
    }
    (sum / (values.len() as f64 - 1.0)).sqrt()
}

fn score_strategy(analyzer: &StrategyQualityAnalyzer, metrics: &mut StrategyMetrics) {
    // QA-3: Continuous scoring with diminishing returns — no cliff effects
    // Each component uses 1 - exp(-k * x) shape: smooth, bounded, no hard steps

    // Sortino (0-30 pts): saturates around 3.0
    let sortino_score = 30.0 * (1.0 - (-metrics.sortino_ratio.max(0.0) * 0.6).exp());

    // Profit Factor (0-20 pts): saturates around 2.5
    let pf_score = 20.0 * (1.0 - (-(metrics.profit_factor.max(0.0) - 1.0).max(0.0) * 1.5).exp());

    // Win Rate (0-15 pts): linear between 0.45-0.70
    let wr_score = 15.0 * ((metrics.win_rate - 0.45) / 0.25).clamp(0.0, 1.0);

    // Calmar (0-20 pts): saturates around 2.0
    let calmar_score = 20.0 * (1.0 - (-metrics.calmar_ratio.max(0.0) * 0.8).exp());

    // Drawdown (0-15 pts): penalizes progressively above 8%
    let dd_score = 15.0 * (1.0 - (metrics.max_drawdown_pct / 0.15).clamp(0.0, 1.0)).max(0.0);

    // Statistical significance (0-10 pts): smooth decay as p-value rises
    let pval = metrics.statistical_significance.clamp(0.0, 1.0);
    let pval_score = 10.0 * (1.0 - pval).powi(3);

    // Monthly consistency (0-10 pts)
    let mwr_score = 10.0 * metrics.monthly_win_rate.clamp(0.0, 1.0);

    // Monthly return (0-10 pts): smooth approach to min target
    let mr_score = if metrics.avg_monthly_return_pct >= analyzer.min_monthly_return_pct {
        10.0 * (metrics.avg_monthly_return_pct / analyzer.min_monthly_return_pct.max(1e-9)).min(1.0)
    } else {
        0.0
    };

    let score = sortino_score
        + pf_score
        + wr_score
        + calmar_score
        + dd_score
        + pval_score
        + mwr_score
        + mr_score;
    metrics.quality_score = score.min(100.0);

    // QA-4: Weighted edge score instead of brittle AND gate
    // Each metric is normalized to [0, 1] relative to its threshold
    let s_sortino = (metrics.sortino_ratio / analyzer.min_sortino.max(1e-9)).min(2.0) * 0.20;
    let s_calmar = (metrics.calmar_ratio / analyzer.min_calmar.max(1e-9)).min(2.0) * 0.15;
    let s_pf = (metrics.profit_factor / analyzer.min_profit_factor.max(1e-9)).min(2.0) * 0.20;
    let s_wr = (metrics.win_rate / analyzer.min_win_rate.max(1e-9)).min(2.0) * 0.15;
    let s_dd = ((analyzer.max_dd_acceptable - metrics.max_drawdown_pct)
        / analyzer.max_dd_acceptable.max(1e-9))
    .clamp(0.0, 2.0)
        * 0.15;
    let s_mr = if analyzer.min_monthly_return_pct > 0.0 {
        (metrics.avg_monthly_return_pct / analyzer.min_monthly_return_pct).clamp(0.0, 2.0) * 0.10
    } else {
        0.10
    };
    let s_pval = (1.0
        - metrics.statistical_significance / analyzer.edge_significance_pvalue.max(1e-9))
    .clamp(0.0, 1.0)
        * 0.05;
    let edge_score = s_sortino + s_calmar + s_pf + s_wr + s_dd + s_mr + s_pval;
    let trades_ok = analyzer.min_trades == 0 || metrics.total_trades >= analyzer.min_trades;
    metrics.has_edge = edge_score >= 0.70 && trades_ok;

    metrics.recommendation = if metrics.quality_score >= 80.0 {
        "EXCELLENT"
    } else if metrics.quality_score >= 70.0 {
        "GOOD"
    } else if metrics.quality_score >= 60.0 {
        "ACCEPTABLE"
    } else {
        "POOR"
    }
    .to_string();
}

fn empty_metrics(strategy_id: &str) -> StrategyMetrics {
    StrategyMetrics {
        strategy_id: strategy_id.to_string(),
        total_trades: 0,
        win_rate: 0.0,
        profit_factor: 0.0,
        sharpe_ratio: 0.0,
        sortino_ratio: 0.0,
        calmar_ratio: 0.0,
        total_return_pct: 0.0,
        avg_win_pct: 0.0,
        avg_loss_pct: 0.0,
        largest_win_pct: 0.0,
        largest_loss_pct: 0.0,
        max_drawdown_pct: 0.0,
        avg_drawdown_pct: 0.0,
        longest_losing_streak: 0,
        longest_winning_streak: 0,
        expectancy: 0.0,
        kelly_fraction: 0.0,
        statistical_significance: 1.0,
        monthly_win_rate: 0.0,
        positive_months: 0,
        negative_months: 0,
        avg_monthly_return_pct: 0.0,
        profit_per_trade: 0.0,
        avg_trade_duration_hours: 0.0,
        trades_per_month: 0.0,
        quality_score: 0.0,
        has_edge: false,
        recommendation: String::new(),
        mc_worst_drawdown_95_pct: None,
        mc_risk_of_ruin_pct: None,
    }
}

pub struct StrategyRanker {
    pub analyzer: StrategyQualityAnalyzer,
    pub strategy_metrics: HashMap<String, StrategyMetrics>,
}

impl StrategyRanker {
    pub fn new(analyzer: Option<StrategyQualityAnalyzer>) -> Self {
        Self {
            analyzer: analyzer.unwrap_or_default(),
            strategy_metrics: HashMap::new(),
        }
    }

    pub fn evaluate_strategies(
        &mut self,
        strategies: &HashMap<String, Vec<Trade>>,
        initial_balance: f64,
    ) -> Vec<StrategyMetrics> {
        let mut results = Vec::new();
        for (strategy_id, trades) in strategies {
            let metrics = self
                .analyzer
                .analyze_strategy(strategy_id, trades, initial_balance);
            self.strategy_metrics
                .insert(strategy_id.clone(), metrics.clone());
            results.push(metrics);
        }
        results.sort_by(|a, b| {
            b.quality_score
                .partial_cmp(&a.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    pub fn get_top_strategies(&self, n: usize, min_quality: f64) -> Vec<String> {
        let mut ranked: Vec<_> = self.strategy_metrics.iter().collect();
        ranked.sort_by(|a, b| {
            b.1.quality_score
                .partial_cmp(&a.1.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
            .into_iter()
            .filter(|(_, m)| m.quality_score >= min_quality && m.has_edge)
            .take(n)
            .map(|(sid, _)| sid.clone())
            .collect()
    }

    pub fn save_rankings(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut rankings = Vec::new();
        for m in self.strategy_metrics.values() {
            rankings.push(serde_json::json!({
                "strategy_id": m.strategy_id,
                "quality_score": m.quality_score,
                "has_edge": m.has_edge,
                "recommendation": m.recommendation,
            }));
        }
        write_json_atomic(path, &rankings).map_err(|err| std::io::Error::other(err.to_string()))
    }
}

#[cfg(test)]
mod overrides_tests {
    use super::*;

    #[test]
    fn quality_runtime_overrides_defaults_match_legacy_env_defaults() {
        let defaults = QualityRuntimeOverrides::default();
        assert_eq!(defaults.min_trades_per_month, 4);
        assert!((defaults.trading_days_per_month - 21.0).abs() < 1e-9);
    }

    #[test]
    fn quality_runtime_overrides_clamp_invalid_trading_days() {
        let bad = QualityRuntimeOverrides {
            min_trades_per_month: 0,
            trading_days_per_month: 0.0,
        };
        assert!((bad.resolved_trading_days_per_month() - 21.0).abs() < 1e-9);

        let nan = QualityRuntimeOverrides {
            min_trades_per_month: 0,
            trading_days_per_month: f64::NAN,
        };
        assert!((nan.resolved_trading_days_per_month() - 21.0).abs() < 1e-9);

        let valid = QualityRuntimeOverrides {
            min_trades_per_month: 8,
            trading_days_per_month: 23.0,
        };
        assert!((valid.resolved_trading_days_per_month() - 23.0).abs() < 1e-9);
    }

    #[test]
    fn current_quality_runtime_overrides_returns_legal_values() {
        let observed = current_quality_runtime_overrides();
        assert!(observed.min_trades_per_month >= 1);
        assert!(observed.trading_days_per_month.is_finite());
    }
}
