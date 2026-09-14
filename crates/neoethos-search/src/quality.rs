use crate::artifact_io::write_json_atomic;
use chrono::{Datelike, TimeZone, Utc};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

const TARGET_DOWNSIDE_SORTINO_SEMANTICS_V1: &str = "neoethos.target-downside-sortino.v1";
const TARGET_DOWNSIDE_SORTINO_PRIMARY_SOURCE_V1: &str =
    "https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf";
const SORTINO_TARGET_RETURN_V1: f64 = 0.0;
const INVALID_TARGET_DOWNSIDE_SORTINO_V1: f64 = f64::NEG_INFINITY;

/// Typed replacement for the legacy `NEOETHOS_BOT_PROP_MIN_TRADES_PER_MONTH`
/// and `NEOETHOS_BOT_TRADING_DAYS_PER_MONTH` env vars. Previously read inline
/// inside monthly metric aggregation, both knobs change canonical strategy
/// quality scoring, so they belong in typed runtime config.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
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
    // `from_env()` DELETED 2026-08-10 with
    // `NEOETHOS_BOT_PROP_MIN_TRADES_PER_MONTH` and
    // `NEOETHOS_BOT_TRADING_DAYS_PER_MONTH`. Both change CANONICAL strategy
    // quality scoring — which months count toward `monthly_win_rate` and how
    // observed trading days become a months-traded estimate — and both are
    // typed on `models.quality_runtime`.

    /// Config-driven constructor (was the `NEOETHOS_BOT_PROP_*` quality
    /// env vars). `trading_days_per_month` is validated finite ≥ 1.0 like
    /// the env reader. A `quality_from_settings_default_matches_env_default`
    /// test guarantees a fresh `Settings` reproduces [`Self::default`].
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        let c = &s.models.quality_runtime;
        let trading_days =
            if c.trading_days_per_month.is_finite() && c.trading_days_per_month >= 1.0 {
                c.trading_days_per_month
            } else {
                Self::default().trading_days_per_month
            };
        Self {
            min_trades_per_month: c.min_trades_per_month,
            trading_days_per_month: trading_days,
        }
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

/// Config-driven install — reads the quality knobs from the single
/// `Settings` instead of the environment. Idempotent.
pub fn install_quality_runtime_overrides_from_settings(s: &neoethos_core::Settings) {
    let _ = QUALITY_RUNTIME_OVERRIDES.set(QualityRuntimeOverrides::from_settings(s));
}

/// Returns the currently installed quality runtime overrides, or the
/// deterministic defaults when no install has happened.
pub fn current_quality_runtime_overrides() -> QualityRuntimeOverrides {
    QUALITY_RUNTIME_OVERRIDES.get().copied().unwrap_or_default()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Trade {
    pub entry_time: i64,
    pub exit_time: Option<i64>,
    pub pnl: f64,
    pub pnl_pct: Option<f64>,
    pub duration_hours: Option<f64>,
    /// Max Favorable Excursion — best unrealized profit reached during the trade ($).
    #[serde(default)]
    pub mfe: f64,
    /// Max Adverse Excursion — worst unrealized loss reached during the trade ($, positive).
    #[serde(default)]
    pub mae: f64,
    /// R-multiple — net P&L / initial $ risk (sl_pips × pip_value_per_lot).
    #[serde(default)]
    pub r_multiple: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub strategy_id: String,
    pub total_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    /// Canonical completed-month return Sharpe when evaluation metrics are supplied;
    /// a trade-return diagnostic otherwise.
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub total_return_pct: f64,
    pub avg_win_pct: f64,
    pub avg_loss_pct: f64,
    /// Average win over average loss — the "2.2 to 1" half of a target profile,
    /// which `profit_factor` cannot express because it folds win rate and payoff
    /// into one number: 30 % of trades at 5:1 and 70 % at 0.6:1 both give 2.1.
    /// Separating them is what makes "57-65 % at 2.2:1" a thing the search can
    /// be pointed at. `0.0` when nothing has lost yet — undefined, not infinite.
    pub payoff_ratio: f64,
    /// Share of the evaluated span spent holding a position.
    ///
    /// A strategy in the market almost always is not selecting entries, it is
    /// participating, and its win rate converges on the base rate of the market
    /// no matter what the entry rule says. Measured because a GPU run showed
    /// candidates emitting a position event on 78 % of all bars — the same
    /// number, read as a strategy property rather than a memory problem.
    pub in_market_pct: f64,
    pub largest_win_pct: f64,
    pub largest_loss_pct: f64,
    /// Canonical intrabar mark-to-market drawdown when evaluation metrics are
    /// supplied; closed-trade balance drawdown for trade-only diagnostics.
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
    /// THE COST-CHARGED NET EXPECTANCY PER TRADE, in account currency.
    ///
    /// The plain mean of the per-trade net P&L the backtest booked, so spread,
    /// commission, swap and the conversion fee are already subtracted. It answers
    /// the only question that decides whether an account grows: after paying the
    /// broker, does the average trade make money?
    ///
    /// This is the PRIMARY survival criterion as of 2026-08-09
    /// (`TargetProfile::accepts`). It replaced a payoff-ratio floor that could
    /// not carry the job: measured, a candidate at payoff 2.53 had an expectancy
    /// of -4.18 pips per trade. Payoff describes the SHAPE of the win/loss split;
    /// only expectancy describes the direction of the money.
    ///
    /// Nothing was renamed to say so — this field already held the number, it was
    /// simply never gated on. Its standard error is `net_expectancy_stderr`.
    pub profit_per_trade: f64,
    /// Standard error of [`Self::profit_per_trade`]: `sd(pnl) / sqrt(n)`.
    ///
    /// An expectancy without one is a point estimate presented as a fact. With
    /// 30 trades and a per-trade sd of 200 currency units, an expectancy of +20
    /// has a standard error of 36 — indistinguishable from zero, and the search
    /// would have ranked it above a +5 with 5 000 trades.
    ///
    /// Sample sd, `n - 1` denominator, `0.0` when fewer than two trades. It
    /// bounds SAMPLING noise only. It says nothing about selection bias across
    /// the thousands of candidates the GA tried — that needs DSR/PBO over the
    /// per-trial return series, which this project does not yet persist and
    /// therefore cannot compute.
    #[serde(default)]
    pub net_expectancy_stderr: f64,
    /// `profit_per_trade / net_expectancy_stderr` — how many standard errors
    /// above zero the expectancy sits. `0.0` when the standard error is zero or
    /// undefined (fewer than two trades, or every trade identical).
    #[serde(default)]
    pub net_expectancy_t_stat: f64,
    pub avg_trade_duration_hours: f64,
    pub trades_per_month: f64,
    pub quality_score: f64,
    pub has_edge: bool,
    pub recommendation: String,
    pub mc_worst_drawdown_95_pct: Option<f64>,
    pub mc_risk_of_ruin_pct: Option<f64>,
    // Pro money-view + equity curve (2026-06-06): the "how much € in how long, with
    // what path" that ratios alone (Sharpe 7) hide. All #[serde(default)] for
    // backward-compat with old <stem>.quality.json artifacts.
    #[serde(default)]
    pub initial_capital: f64,
    #[serde(default)]
    pub net_profit: f64,
    #[serde(default)]
    pub final_balance: f64,
    /// Closed-trade balance drawdown in account currency. This is not the
    /// canonical intrabar money drawdown (which the evaluator does not return).
    #[serde(default)]
    pub max_drawdown_money: f64,
    #[serde(default)]
    pub recovery_factor: f64,
    #[serde(default)]
    pub period_start_ms: i64,
    #[serde(default)]
    pub period_end_ms: i64,
    #[serde(default)]
    pub period_days: f64,
    /// Equity after each closed trade (index 0 = initial_capital) — the
    /// start → trough → end curve the operator wants to graph.
    #[serde(default)]
    pub equity_curve: Vec<f64>,
    // Per-trade excursion aggregates (operator 2026-06-06): "ανά συναλλαγή" pro stats.
    #[serde(default)]
    pub avg_mfe: f64,
    #[serde(default)]
    pub avg_mae: f64,
    #[serde(default)]
    pub avg_r_multiple: f64,
    #[serde(default)]
    pub mfe_capture_ratio: f64,
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
    /// 2026-05-26 operator directive (dual-mode product): canonical threshold
    /// for the "month has enough trades to count toward monthly stats" gate.
    /// `Some(n)` overrides the env-driven `QualityRuntimeOverrides`. The
    /// FilteringConfig path in `quality_analyzer_for_config` (discovery.rs)
    /// sets this so the Settings-driven value wins over the env default.
    /// `None` preserves the old env-driven behaviour for legacy callers.
    pub min_trades_per_month: Option<usize>,
}

impl Default for StrategyQualityAnalyzer {
    fn default() -> Self {
        Self {
            min_sharpe: 1.2,
            min_sortino: 1.2,
            min_calmar: 1.0,
            // 2026-05-26 operator directive (dual-mode product): canonical
            // value across the workspace. Previously 1.5 here, 1.05 in
            // strategy_gene.rs, 1.2 in gauntlet.rs (now deleted). 1.2 matches
            // the FTMO industry baseline and avoids a divergent default per
            // code path. If you change this, also change the matching default
            // in `genetic::strategy_gene::FilteringConfig`.
            min_profit_factor: 1.2,
            min_win_rate: 0.50,
            min_trades: 0,
            max_dd_acceptable: 0.15,
            min_monthly_return_pct: 0.04,
            edge_significance_pvalue: 0.01,
            // None = fall back to env-driven QualityRuntimeOverrides for
            // legacy callers; discovery's `quality_analyzer_for_config`
            // overrides this with the FilteringConfig value.
            min_trades_per_month: None,
        }
    }
}

impl StrategyQualityAnalyzer {
    /// Trade-only diagnostics for callers without the evaluated bar interval.
    /// Frequency, period and exposure describe the observed trades only; these
    /// are not full-span account metrics. Discovery must use
    /// [`Self::analyze_strategy_with_evaluation`] instead.
    pub fn analyze_strategy(
        &self,
        strategy_id: &str,
        trades: &[Trade],
        initial_balance: f64,
    ) -> StrategyMetrics {
        self.analyze_strategy_inner(strategy_id, trades, initial_balance, None)
    }

    /// Analyze the ledger and canonical metrics returned by the SAME account
    /// replay. The interval is the first through last evaluated bar timestamp,
    /// not the first entry through last exit. Frequency counts every weekday in
    /// those inclusive UTC dates, including dates with no trades, using the
    /// configured trading-days-per-month convention. Calendar exposure uses the
    /// elapsed milliseconds between the supplied endpoints.
    #[allow(clippy::too_many_arguments)]
    pub fn analyze_strategy_with_evaluation(
        &self,
        strategy_id: &str,
        trades: &[Trade],
        initial_balance: f64,
        evaluation_start_ms: i64,
        evaluation_end_ms: i64,
        canonical_metrics: &[f64; 11],
    ) -> anyhow::Result<StrategyMetrics> {
        anyhow::ensure!(
            initial_balance.is_finite() && initial_balance > 0.0,
            "quality evaluation initial balance must be finite and positive"
        );
        let weekdays = evaluation_weekdays(evaluation_start_ms, evaluation_end_ms)?;
        anyhow::ensure!(
            trades.is_empty() || weekdays > 0,
            "quality evaluation with trades requires a weekday in its supplied interval"
        );
        anyhow::ensure!(
            canonical_metrics[0].is_finite()
                && (canonical_metrics[1].is_finite()
                    || canonical_metrics[1] == crate::eval::INVALID_MONTHLY_RETURN_SHARPE_V1)
                && canonical_metrics[3].is_finite()
                && canonical_metrics[3] >= 0.0
                && canonical_metrics[8] == trades.len() as f64,
            "quality evaluation requires canonical PnL, Sharpe, nonnegative drawdown and matching trade count"
        );
        let mut net = 0.0;
        let mut absolute_pnl = 0.0;
        for trade in trades {
            anyhow::ensure!(
                trade.pnl.is_finite()
                    && trade.entry_time >= evaluation_start_ms
                    && trade.exit_time.is_some_and(|exit| {
                        exit >= trade.entry_time && exit <= evaluation_end_ms
                    }),
                "quality evaluation ledger contains non-finite PnL or a trade outside its supplied interval"
            );
            net += trade.pnl;
            absolute_pnl += trade.pnl.abs();
        }
        // The ledger sums PnL from zero; the core updates initial balance and
        // subtracts it at the end. Permit their floating-point summation error,
        // not a financial tolerance or a different ledger.
        let rounding_bound =
            f64::EPSILON * (trades.len() as f64 + 2.0) * (initial_balance + absolute_pnl) * 4.0;
        anyhow::ensure!(
            net.is_finite()
                && rounding_bound.is_finite()
                && (net - canonical_metrics[0]).abs() <= rounding_bound,
            "quality evaluation canonical PnL differs from its supplied ledger"
        );
        Ok(self.analyze_strategy_inner(
            strategy_id,
            trades,
            initial_balance,
            Some((
                evaluation_start_ms,
                evaluation_end_ms,
                weekdays,
                canonical_metrics,
            )),
        ))
    }

    fn analyze_strategy_inner(
        &self,
        strategy_id: &str,
        trades: &[Trade],
        initial_balance: f64,
        evaluation: Option<(i64, i64, u64, &[f64; 11])>,
    ) -> StrategyMetrics {
        if trades.is_empty() {
            let mut metrics = empty_metrics(strategy_id);
            if let Some((start, end, _, canonical)) = evaluation {
                metrics.initial_capital = initial_balance;
                metrics.final_balance = initial_balance;
                metrics.period_start_ms = start;
                metrics.period_end_ms = end;
                metrics.period_days = (end - start) as f64 / 86_400_000.0;
                metrics.sharpe_ratio = canonical[1];
                metrics.max_drawdown_pct = canonical[3];
                metrics.equity_curve.push(initial_balance);
            }
            return metrics;
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
        // Pro equity curve + money drawdown (2026-06-06): collect the equity path
        // (index 0 = start) and the worst peak-to-trough in account currency.
        let mut equity_curve = Vec::with_capacity(total_trades + 1);
        equity_curve.push(initial_balance);
        let mut max_dd_money = 0.0_f64;
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
            let dd_money = peak - equity;
            if dd_money > max_dd_money {
                max_dd_money = dd_money;
            }
            equity_curve.push(equity);
        }
        let max_dd = evaluation.map_or_else(
            || drawdowns.iter().cloned().fold(0.0, f64::max),
            |(_, _, _, canonical)| canonical[3],
        );
        let avg_dd = if !drawdowns.is_empty() {
            mean(&drawdowns)
        } else {
            0.0
        };

        let trades_per_month_raw = evaluation.map_or_else(
            || calculate_trade_frequency(trades),
            |(_, _, weekdays, _)| {
                trades.len() as f64
                    * current_quality_runtime_overrides().resolved_trading_days_per_month()
                    / weekdays as f64
            },
        );
        let trades_per_year = (trades_per_month_raw * 12.0).max(1.0);
        let sharpe = evaluation.map_or_else(
            || calculate_sharpe(&returns, trades_per_year),
            |(_, _, _, canonical)| canonical[1],
        );
        let sortino = calculate_sortino(&returns, trades_per_year);
        if !sortino.is_finite() {
            let mut invalid = empty_metrics(strategy_id);
            invalid.sortino_ratio = INVALID_TARGET_DOWNSIDE_SORTINO_V1;
            invalid.quality_score = f64::NEG_INFINITY;
            invalid.recommendation = "INVALID_SORTINO_INPUT".to_string();
            return invalid;
        }

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

        // 2026-05-26 operator directive (dual-mode product): pass the
        // analyzer's `min_trades_per_month` override so the FilteringConfig
        // value wins over the env-driven default. None preserves legacy
        // env-driven behaviour for direct callers using
        // `StrategyQualityAnalyzer::default()`.
        let monthly_metrics =
            analyze_monthly_consistency(trades, initial_balance, self.min_trades_per_month);
        let monthly_win_rate = monthly_metrics.monthly_win_rate;
        let avg_monthly_return_pct = monthly_metrics.avg_return_pct;

        let avg_duration = if durations.is_empty() {
            0.0
        } else {
            mean(&durations)
        };
        let trades_per_month = trades_per_month_raw;

        // --- Monte Carlo Simulation (QA-2: block bootstrap on daily PnL) ---
        //
        // Audit D07 (2026-07-13): this was a permutation, not a bootstrap.
        //   1. `shuffle` REORDERS the existing blocks (sampling WITHOUT
        //      replacement), so every iteration had the identical set of
        //      daily PnLs — the final equity was the SAME every time (the
        //      sum is order-independent) and only the drawdown PATH varied.
        //      That measures ordering risk, not resampling uncertainty, and
        //      systematically UNDERSTATES tail risk: a real bad-luck run
        //      (more losing days than actually occurred) was never sampled.
        //      A proper block bootstrap draws blocks WITH replacement, so
        //      losing days can repeat and the equity outcome genuinely varies.
        //   2. `rand::rng()` is unseeded → the p95 drawdown and
        //      risk-of-ruin (inputs to a promotion-affecting gate) changed
        //      run to run. Seed deterministically from the trade PnLs so the
        //      same strategy always yields the same risk numbers.
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        for t in trades {
            seed ^= (t.pnl.to_bits())
                .wrapping_mul(0x1000_0000_1B3)
                .rotate_left(13);
            seed = seed.wrapping_mul(0x0100_0000_01B3);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mc_iterations = 1000;
        let mut worst_dds = Vec::with_capacity(mc_iterations);
        let mut ruined_count = 0;
        let ruin_threshold = initial_balance * 0.50;

        // Group trade PnLs by calendar day for block bootstrap. Use a
        // BTreeMap so the block order is deterministic (chronological) — a
        // HashMap's `into_values()` order is randomized per process, which
        // (even with a seeded RNG) would make the with-replacement draw pick
        // different blocks each run and defeat D07's reproducibility goal.
        let mut daily_pnl_blocks: std::collections::BTreeMap<i64, Vec<f64>> =
            std::collections::BTreeMap::new();
        for trade in trades {
            if trade.entry_time > 0 {
                let day_key = trade.entry_time / 86_400_000;
                daily_pnl_blocks.entry(day_key).or_default().push(trade.pnl);
            }
        }
        let day_blocks: Vec<Vec<f64>> = daily_pnl_blocks.into_values().collect();
        for _ in 0..mc_iterations {
            let (max_mc_dd, ruined) = bootstrap_draw_risk(
                &day_blocks,
                &pnls,
                initial_balance,
                ruin_threshold,
                &mut rng,
            );
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

        // Pro money-view + recovery factor + period (2026-06-06).
        let net_profit = total_return;
        let final_balance = initial_balance + total_return;
        let recovery_factor = if max_dd_money > 1e-6 {
            (net_profit / max_dd_money).clamp(-1000.0, 1000.0)
        } else if net_profit > 0.0 {
            1000.0
        } else {
            0.0
        };
        let (period_start_ms, period_end_ms) = evaluation.map_or_else(
            || {
                let start = trades
                    .iter()
                    .map(|t| t.entry_time)
                    .filter(|&t| t > 0)
                    .min()
                    .unwrap_or(0);
                let end = trades
                    .iter()
                    .filter_map(|t| t.exit_time)
                    .max()
                    .unwrap_or(start);
                (start, end)
            },
            |(start, end, _, _)| (start, end),
        );
        let period_days = if period_end_ms > period_start_ms {
            (period_end_ms - period_start_ms) as f64 / 86_400_000.0
        } else {
            0.0
        };

        // Per-trade excursion aggregates (operator 2026-06-06): MFE/MAE/R-multiple
        // averaged + MFE-capture-ratio (realized / potential; <40% = noise-driven exits).
        let avg_mfe = if !trades.is_empty() {
            trades.iter().map(|t| t.mfe).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };
        let avg_mae = if !trades.is_empty() {
            trades.iter().map(|t| t.mae).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };
        let avg_r_multiple = if !trades.is_empty() {
            trades.iter().map(|t| t.r_multiple).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };
        let sum_mfe: f64 = trades.iter().map(|t| t.mfe).sum();
        let mfe_capture_ratio = if sum_mfe > 1e-9 {
            total_return / sum_mfe
        } else {
            0.0
        };

        // Cost-charged net expectancy per trade + its standard error. `pnls` are
        // the realised per-trade net P&L, so every cost the engine charges is
        // already in them.
        //
        // Sample sd (n - 1). With one trade there is no dispersion to estimate,
        // so the error is 0.0 and the t-stat is 0.0 — deliberately NOT infinity:
        // a single lucky trade must not read as infinitely significant.
        let net_expectancy_per_trade = if pnls.is_empty() { 0.0 } else { mean(&pnls) };
        let net_expectancy_stderr = if pnls.len() >= 2 {
            let sd = stddev_sample(&pnls, net_expectancy_per_trade);
            let se = sd / (pnls.len() as f64).sqrt();
            if se.is_finite() && se > 0.0 { se } else { 0.0 }
        } else {
            0.0
        };
        let net_expectancy_t_stat = if net_expectancy_stderr > 0.0 {
            net_expectancy_per_trade / net_expectancy_stderr
        } else {
            0.0
        };

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
            payoff_ratio: if avg_loss_mag > 1e-12 {
                avg_win_pct / avg_loss_mag
            } else {
                0.0
            },
            in_market_pct: evaluation.map_or_else(
                || time_in_market(trades),
                |(start, end, _, _)| {
                    trades
                        .iter()
                        .filter_map(|trade| {
                            trade.exit_time.map(|exit| (exit - trade.entry_time) as f64)
                        })
                        .sum::<f64>()
                        / (end - start) as f64
                },
            ),
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
            profit_per_trade: net_expectancy_per_trade,
            net_expectancy_stderr,
            net_expectancy_t_stat,
            avg_trade_duration_hours: avg_duration,
            trades_per_month,
            quality_score: 0.0,
            has_edge: false,
            recommendation: String::new(),
            mc_worst_drawdown_95_pct: Some(mc_worst_dd_95),
            mc_risk_of_ruin_pct: Some(mc_risk_of_ruin),
            initial_capital: initial_balance,
            net_profit,
            final_balance,
            max_drawdown_money: max_dd_money,
            recovery_factor,
            period_start_ms,
            period_end_ms,
            period_days,
            equity_curve,
            avg_mfe,
            avg_mae,
            avg_r_multiple,
            mfe_capture_ratio,
        };

        score_strategy(self, &mut metrics);
        if sharpe == crate::eval::INVALID_MONTHLY_RETURN_SHARPE_V1 {
            // Preserve the core's candidate rejection, not a run-level error or
            // an apparently acceptable score based on other finite metrics.
            metrics.quality_score = f64::NEG_INFINITY;
            metrics.has_edge = false;
            metrics.recommendation = "INVALID_COMPLETED_MONTH_RETURN".to_string();
        }
        metrics
    }
}

/// Inclusive UTC calendar dates; O(1) weeks plus at most six remainder days.
/// Endpoint validation also bounds every later elapsed-time subtraction.
fn evaluation_weekdays(start_ms: i64, end_ms: i64) -> anyhow::Result<u64> {
    anyhow::ensure!(
        start_ms >= 0 && end_ms > start_ms,
        "quality evaluation interval must have ordered nonnegative endpoints"
    );
    let start = Utc
        .timestamp_millis_opt(start_ms)
        .single()
        .ok_or_else(|| anyhow::anyhow!("quality evaluation start is outside the UTC calendar"))?;
    let end = Utc
        .timestamp_millis_opt(end_ms)
        .single()
        .ok_or_else(|| anyhow::anyhow!("quality evaluation end is outside the UTC calendar"))?;
    let days = (end.date_naive() - start.date_naive()).num_days() as u64 + 1;
    let first_weekday = u64::from(start.weekday().num_days_from_monday());
    Ok((days / 7) * 5
        + (0..days % 7)
            .filter(|offset| (first_weekday + offset) % 7 < 5)
            .count() as u64)
}

/// Visit the same with-replacement draws in the same P&L order without storing
/// the resampled tape. Uneven day blocks can otherwise repeat into a temporary
/// vector much larger than the original trade history. Interleaving the equity
/// updates with sampling does not consume additional RNG draws or regroup sums.
fn bootstrap_draw_risk(
    day_blocks: &[Vec<f64>],
    pnls: &[f64],
    initial_balance: f64,
    ruin_threshold: f64,
    rng: &mut StdRng,
) -> (f64, bool) {
    let mut eq = initial_balance;
    let mut pk = initial_balance;
    let mut max_mc_dd = 0.0_f64;
    let mut ruined = false;
    let mut apply_pnl = |p: f64| {
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
    };

    // Fallback to trade-level if fewer than 5 distinct days. The number and
    // order of uniform index draws match the former materialized implementation.
    if day_blocks.len() >= 5 {
        let n = day_blocks.len();
        for _ in 0..n {
            let block = &day_blocks[rng.random_range(0..n)];
            for &p in block {
                apply_pnl(p);
            }
        }
    } else if !pnls.is_empty() {
        let n = pnls.len();
        for _ in 0..n {
            apply_pnl(pnls[rng.random_range(0..n)]);
        }
    }
    (max_mc_dd, ruined)
}

#[derive(Debug, Clone)]
struct MonthlyMetrics {
    monthly_win_rate: f64,
    positive: usize,
    negative: usize,
    avg_return_pct: f64,
}

fn analyze_monthly_consistency(
    trades: &[Trade],
    initial_balance: f64,
    min_trades_per_month_override: Option<usize>,
) -> MonthlyMetrics {
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
    // weight in monthly_win_rate as a month with 50 trades).
    //
    // 2026-05-26 operator directive (dual-mode product): the FilteringConfig
    // value wins via `min_trades_per_month_override`; legacy direct callers
    // (None) still get the env-driven `QualityRuntimeOverrides` default.
    let min_trades_per_month = min_trades_per_month_override
        .unwrap_or_else(|| current_quality_runtime_overrides().min_trades_per_month);
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

// Trade-return diagnostic scaling. It is not the canonical fixed-calendar
// return Sharpe, and square-root scaling alone does not prove independence.
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
    let _authority = (
        TARGET_DOWNSIDE_SORTINO_SEMANTICS_V1,
        TARGET_DOWNSIDE_SORTINO_PRIMARY_SOURCE_V1,
    );
    if !trades_per_year.is_finite() || returns.iter().any(|value| !value.is_finite()) {
        return INVALID_TARGET_DOWNSIDE_SORTINO_V1;
    }
    if returns.len() < 2 {
        return 0.0;
    }

    let mean_return = mean(returns);
    if !mean_return.is_finite() {
        return INVALID_TARGET_DOWNSIDE_SORTINO_V1;
    }
    let mut downside_sum_squares = 0.0;
    for &period_return in returns {
        let shortfall = (period_return - SORTINO_TARGET_RETURN_V1).min(0.0);
        downside_sum_squares += shortfall * shortfall;
        if !downside_sum_squares.is_finite() {
            return INVALID_TARGET_DOWNSIDE_SORTINO_V1;
        }
    }
    let target_downside_deviation = (downside_sum_squares / returns.len() as f64).sqrt();
    if !target_downside_deviation.is_finite() {
        return INVALID_TARGET_DOWNSIDE_SORTINO_V1;
    }
    if target_downside_deviation < 1e-9 {
        return 0.0;
    }
    let annualization = trades_per_year.max(1.0).sqrt();
    let sortino =
        ((mean_return - SORTINO_TARGET_RETURN_V1) / target_downside_deviation) * annualization;
    if sortino.is_finite() {
        sortino
    } else {
        INVALID_TARGET_DOWNSIDE_SORTINO_V1
    }
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

/// Share of the evaluated span spent holding a position.
///
/// Summed holding time over the span from the first entry to the last exit.
/// Overlapping positions are counted once each, so a strategy running several
/// at a time can exceed 1.0 — that is information, not an error: it is more
/// exposed than the timeline, and clamping would hide exactly the case worth
/// seeing.
///
/// `0.0` when the trades carry no usable times, so a missing timestamp reads as
/// "unknown" rather than "never in the market".
fn time_in_market(trades: &[Trade]) -> f64 {
    let spans: Vec<(i64, i64)> = trades
        .iter()
        .filter_map(|t| t.exit_time.map(|exit| (t.entry_time, exit)))
        .filter(|(entry, exit)| exit > entry)
        .collect();
    if spans.is_empty() {
        return 0.0;
    }
    let first = spans.iter().map(|(entry, _)| *entry).min().unwrap_or(0);
    let last = spans.iter().map(|(_, exit)| *exit).max().unwrap_or(0);
    let total = (last - first) as f64;
    if total <= 0.0 {
        return 0.0;
    }
    let held: f64 = spans
        .iter()
        .map(|(entry, exit)| (exit - entry) as f64)
        .sum();
    held / total
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
    // **2026-05-25 unwrap audit**: `StudentsT::new(loc, scale, freedom)`
    // returns `Err` only for non-positive `scale` or `freedom`. Here
    // `scale = 1.0` (literal) and `freedom = n - 1.0 >= 9.0` (because
    // the guard at the top of this function returns early when
    // `returns.len() < 10`). So this is logically infallible — but per
    // the no-panic doctrine we still pattern-match instead of
    // `.unwrap()`. A future regression in the guard above would now
    // return p=1.0 (treat as not-significant) instead of panicking the
    // entire validation pipeline.
    let Ok(dist) = StudentsT::new(0.0, 1.0, df) else {
        return 1.0;
    };
    1.0 - dist.cdf(t_stat)
}

use neoethos_core::utils::{mean, stddev_sample};

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

pub(crate) fn empty_metrics(strategy_id: &str) -> StrategyMetrics {
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
        net_expectancy_stderr: 0.0,
        net_expectancy_t_stat: 0.0,
        avg_trade_duration_hours: 0.0,
        trades_per_month: 0.0,
        quality_score: 0.0,
        has_edge: false,
        recommendation: String::new(),
        mc_worst_drawdown_95_pct: None,
        mc_risk_of_ruin_pct: None,
        initial_capital: 0.0,
        net_profit: 0.0,
        final_balance: 0.0,
        max_drawdown_money: 0.0,
        recovery_factor: 0.0,
        period_start_ms: 0,
        period_end_ms: 0,
        period_days: 0.0,
        equity_curve: Vec::new(),
        avg_mfe: 0.0,
        avg_mae: 0.0,
        avg_r_multiple: 0.0,
        payoff_ratio: 0.0,
        in_market_pct: 0.0,
        mfe_capture_ratio: 0.0,
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

    fn quality_timestamp(year: i32, month: u32, day: u32) -> i64 {
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0)
            .single()
            .unwrap()
            .timestamp_millis()
    }

    #[test]
    fn evaluation_quality_frequency_includes_inactive_weekdays_and_leap_day() {
        assert_eq!(
            evaluation_weekdays(
                quality_timestamp(2024, 2, 28),
                quality_timestamp(2024, 3, 4)
            )
            .unwrap(),
            4, // Wednesday, leap-day Thursday, Friday and Monday.
        );
        assert_eq!(
            evaluation_weekdays(quality_timestamp(2024, 3, 2), quality_timestamp(2024, 3, 3))
                .unwrap(),
            0,
        );
        let start = quality_timestamp(2024, 1, 1);
        let end = quality_timestamp(2024, 12, 31);
        assert_eq!(evaluation_weekdays(start, end).unwrap(), 262);
        let trades = vec![
            Trade {
                entry_time: start,
                exit_time: Some(start + 3_600_000),
                pnl: 10.0,
                ..Trade::default()
            },
            Trade {
                entry_time: start + 86_400_000,
                exit_time: Some(start + 90_000_000),
                pnl: -2.0,
                ..Trade::default()
            },
        ];
        let canonical = [8.0, 0.5, 1010.0, 0.02, 0.5, 5.0, 4.0, 0.0, 2.0, 0.0, 0.02];
        let analyzer = StrategyQualityAnalyzer::default();
        let full = analyzer
            .analyze_strategy_with_evaluation("idle", &trades, 1000.0, start, end, &canonical)
            .unwrap();
        let short_end = quality_timestamp(2024, 1, 2) + 3_600_000;
        let short = analyzer
            .analyze_strategy_with_evaluation("idle", &trades, 1000.0, start, short_end, &canonical)
            .unwrap();
        let expected =
            2.0 * current_quality_runtime_overrides().resolved_trading_days_per_month() / 262.0;
        assert_eq!(full.trades_per_month.to_bits(), expected.to_bits());
        assert!(full.trades_per_month < short.trades_per_month);
        assert!(full.sortino_ratio < short.sortino_ratio);
        assert!(full.in_market_pct < short.in_market_pct);
        assert_eq!(full.period_start_ms, start);
        assert_eq!(full.period_end_ms, end);
        assert_eq!(full.period_days, 365.0);
        assert_eq!(full.sharpe_ratio.to_bits(), canonical[1].to_bits());
        assert_eq!(full.equity_curve, short.equity_curve);
    }

    #[test]
    fn evaluation_quality_uses_same_replay_canonical_sharpe_and_intrabar_drawdown() {
        // Synthetic account replay, not profitability evidence. One trade has
        // open-position excursions before its target; another reaches its stop.
        let start = quality_timestamp(2024, 1, 1);
        let timestamps: Vec<_> = (0..400).map(|day| start + day * 86_400_000).collect();
        let close = vec![100.0; timestamps.len()];
        let mut high = vec![100.5; timestamps.len()];
        let mut low = vec![99.5; timestamps.len()];
        high[2] = 105.0;
        low[2] = 99.0;
        high[3] = 111.0;
        low[102] = 97.0;
        let mut signals = vec![0; timestamps.len()];
        signals[0] = 1;
        signals[100] = 1;
        let settings = crate::eval::BacktestSettings {
            initial_equity_override: Some(10_000.0),
            pip_value: 1.0,
            pip_value_per_lot: 1.0,
            spread_pips: 0.0,
            commission_per_trade: 0.0,
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
            sl_pips: 2.0,
            tp_pips: 10.0,
            risk_based_sizing: false,
            ..crate::eval::BacktestSettings::default()
        };
        let (months, days) = crate::genetic::search_engine::month_day_indices(&timestamps);
        let (canonical, trades) = crate::eval::evaluate_strategy_with_confidence_and_ledger_core(
            &close,
            &high,
            &low,
            &signals,
            &[],
            &months,
            &days,
            &timestamps,
            &settings,
        )
        .unwrap();
        assert_eq!(
            trades.iter().map(|trade| trade.pnl).collect::<Vec<_>>(),
            vec![10.0, -2.0]
        );
        let analyzer = StrategyQualityAnalyzer::default();
        let diagnostic = analyzer.analyze_strategy("canonical", &trades, 10_000.0);
        let actual = analyzer
            .analyze_strategy_with_evaluation(
                "canonical",
                &trades,
                10_000.0,
                start,
                *timestamps.last().unwrap(),
                &canonical,
            )
            .unwrap();
        assert_eq!(actual.sharpe_ratio.to_bits(), canonical[1].to_bits());
        assert_eq!(actual.max_drawdown_pct.to_bits(), canonical[3].to_bits());
        assert!(actual.max_drawdown_pct > diagnostic.max_drawdown_pct);
        assert_ne!(
            actual.sharpe_ratio.to_bits(),
            diagnostic.sharpe_ratio.to_bits()
        );
        assert_eq!(
            actual.calmar_ratio,
            (actual.total_return_pct / canonical[3]).clamp(-1000.0, 1000.0)
        );
        assert_eq!(actual.max_drawdown_money, diagnostic.max_drawdown_money);
        assert_eq!(actual.equity_curve, diagnostic.equity_curve);
        assert_eq!(actual.net_profit, canonical[0]);
        let mut rescored = actual.clone();
        score_strategy(&analyzer, &mut rescored);
        assert_eq!(
            actual.quality_score.to_bits(),
            rescored.quality_score.to_bits()
        );
        assert_eq!(actual.has_edge, rescored.has_edge);
    }

    #[test]
    fn evaluation_quality_rejects_mismatched_ledger_and_preserves_invalid_sharpe() {
        let start = quality_timestamp(2024, 1, 1);
        let end = quality_timestamp(2024, 2, 1);
        let trades = vec![Trade {
            entry_time: start,
            exit_time: Some(start + 3_600_000),
            pnl: 1.0,
            ..Trade::default()
        }];
        let canonical = [1.0, 0.0, 1001.0, 0.01, 1.0, 10.0, 1.0, 0.0, 1.0, 0.0, 0.01];
        let analyzer = StrategyQualityAnalyzer::default();
        for (bad_start, bad_end) in [(start, start), (end, start), (-1, end), (start, i64::MAX)] {
            assert!(
                analyzer
                    .analyze_strategy_with_evaluation(
                        "bad", &trades, 1000.0, bad_start, bad_end, &canonical
                    )
                    .is_err()
            );
        }
        for (slot, value) in [
            (0, 2.0),
            (8, 2.0),
            (1, f64::NAN),
            (3, -0.1),
            (3, f64::INFINITY),
        ] {
            let mut bad = canonical;
            bad[slot] = value;
            assert!(
                analyzer
                    .analyze_strategy_with_evaluation("bad", &trades, 1000.0, start, end, &bad)
                    .is_err()
            );
        }
        let mut invalid = canonical;
        invalid[1] = crate::eval::INVALID_MONTHLY_RETURN_SHARPE_V1;
        let rejected = analyzer
            .analyze_strategy_with_evaluation("invalid", &trades, 1000.0, start, end, &invalid)
            .unwrap();
        assert_eq!(rejected.sharpe_ratio, f64::NEG_INFINITY);
        assert_eq!(rejected.quality_score, f64::NEG_INFINITY);
        assert!(!rejected.has_edge);
        assert_eq!(rejected.recommendation, "INVALID_COMPLETED_MONTH_RETURN");
        let mut outside = trades;
        outside[0].exit_time = Some(end + 1);
        assert!(
            analyzer
                .analyze_strategy_with_evaluation("bad", &outside, 1000.0, start, end, &canonical)
                .is_err()
        );
    }

    #[test]
    fn evaluation_quality_empty_ledger_retains_explicit_span_without_edge_claim() {
        let start = quality_timestamp(2024, 3, 2);
        let end = quality_timestamp(2024, 3, 3);
        let actual = StrategyQualityAnalyzer::default()
            .analyze_strategy_with_evaluation("empty", &[], 1000.0, start, end, &[0.0; 11])
            .unwrap();
        assert_eq!(actual.period_start_ms, start);
        assert_eq!(actual.period_end_ms, end);
        assert_eq!(actual.period_days, 1.0);
        assert_eq!(actual.equity_curve, vec![1000.0]);
        assert_eq!(actual.trades_per_month, 0.0);
        assert_eq!(actual.quality_score, 0.0);
        assert!(!actual.has_edge);
    }

    // Independent reference: preserve the pre-streaming allocation and update
    // order so a changed draw count, block order or floating-point sum is caught.
    fn materialized_bootstrap_draw_risk(
        day_blocks: &[Vec<f64>],
        pnls: &[f64],
        initial_balance: f64,
        ruin_threshold: f64,
        rng: &mut StdRng,
    ) -> (f64, bool, usize) {
        let shuffled_pnls: Vec<f64> = if day_blocks.len() >= 5 {
            let n = day_blocks.len();
            let mut out = Vec::with_capacity(pnls.len());
            for _ in 0..n {
                let b = &day_blocks[rng.random_range(0..n)];
                out.extend_from_slice(b);
            }
            out
        } else if pnls.is_empty() {
            Vec::new()
        } else {
            let n = pnls.len();
            (0..n).map(|_| pnls[rng.random_range(0..n)]).collect()
        };
        let sampled_len = shuffled_pnls.len();
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
        (max_mc_dd, ruined, sampled_len)
    }

    fn assert_streaming_bootstrap_matches_materialized(
        day_blocks: &[Vec<f64>],
        pnls: &[f64],
        initial_balance: f64,
    ) -> usize {
        let mut longest_tape = 0;
        for seed in [0, 1, 7, 0x9E37_79B9_7F4A_7C15] {
            let mut old_rng = StdRng::seed_from_u64(seed);
            let mut new_rng = StdRng::seed_from_u64(seed);
            for draw in 0..1000 {
                let expected = materialized_bootstrap_draw_risk(
                    day_blocks,
                    pnls,
                    initial_balance,
                    initial_balance * 0.50,
                    &mut old_rng,
                );
                let actual = bootstrap_draw_risk(
                    day_blocks,
                    pnls,
                    initial_balance,
                    initial_balance * 0.50,
                    &mut new_rng,
                );
                assert_eq!(
                    actual.0.to_bits(),
                    expected.0.to_bits(),
                    "seed={seed} draw={draw}"
                );
                assert_eq!(actual.1, expected.1, "seed={seed} draw={draw}");
                longest_tape = longest_tape.max(expected.2);
            }
            assert_eq!(
                old_rng.random::<u64>(),
                new_rng.random::<u64>(),
                "RNG state drift for seed {seed}"
            );
        }
        longest_tape
    }

    #[test]
    fn streaming_bootstrap_matches_uneven_repeated_day_blocks_bit_for_bit() {
        let day_blocks = vec![
            vec![1.0e8, 0.1, -1.0e8, -4.5],
            vec![-70.0],
            vec![0.125; 97],
            vec![25.0, -0.25],
            vec![-1.5; 11],
        ];
        let pnls: Vec<f64> = day_blocks.iter().flatten().copied().collect();
        let longest_tape =
            assert_streaming_bootstrap_matches_materialized(&day_blocks, &pnls, 100.0);
        assert!(
            longest_tape > pnls.len(),
            "fixture must exercise expansion beyond input trades"
        );
    }

    #[test]
    fn streaming_bootstrap_matches_negative_pnl_and_ruin_bit_for_bit() {
        let day_blocks = vec![
            vec![-60.0],
            vec![-15.0, -30.0],
            vec![-30.0],
            vec![-40.0],
            vec![-20.0],
        ];
        let pnls: Vec<f64> = day_blocks.iter().flatten().copied().collect();
        assert_streaming_bootstrap_matches_materialized(&day_blocks, &pnls, 100.0);
        let mut rng = StdRng::seed_from_u64(1);
        let (_, ruined) = bootstrap_draw_risk(&day_blocks, &pnls, 100.0, 50.0, &mut rng);
        assert!(ruined);
    }

    #[test]
    fn streaming_bootstrap_matches_trade_level_fallback_and_empty_input() {
        let pnls = [100.0, -300.0, 0.1, 60.0, -0.125, 1.0e-8];
        // Fewer than five day blocks must draw from the complete P&L history,
        // including values without a usable day, not from the partial blocks.
        let partial_day_blocks = vec![vec![999.0]; 4];
        assert_streaming_bootstrap_matches_materialized(&partial_day_blocks, &pnls, 100.0);
        assert_eq!(
            assert_streaming_bootstrap_matches_materialized(&[], &[], 100.0),
            0
        );
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(
            bootstrap_draw_risk(&[], &[], 100.0, 50.0, &mut rng),
            (0.0, false)
        );
    }

    #[test]
    fn money_view_and_equity_curve_are_correct() {
        // Deterministic check of the pro money-view (2026-06-06): a Sharpe number
        // alone hides "how much EUR in how long, with what curve" — verify those.
        let analyzer = StrategyQualityAnalyzer::default();
        let day = 86_400_000_i64;
        // Real timestamps are never 0 (analyze_strategy filters entry_time>0 to
        // ignore unset times), so start at day 1.
        let mk = |i: i64, pnl: f64| Trade {
            entry_time: (i + 1) * day,
            exit_time: Some((i + 2) * day),
            pnl,
            pnl_pct: Some(pnl / 100_000.0),
            duration_hours: Some(24.0),
            ..Default::default()
        };
        // +600, -350, +600, -350 over 4 days on EUR 100,000.
        let trades = vec![mk(0, 600.0), mk(1, -350.0), mk(2, 600.0), mk(3, -350.0)];
        let m = analyzer.analyze_strategy("demo", &trades, 100_000.0);

        assert_eq!(m.initial_capital, 100_000.0);
        assert!((m.net_profit - 500.0).abs() < 1e-6, "net {}", m.net_profit);
        assert!((m.final_balance - 100_500.0).abs() < 1e-6);
        // equity path: 100000 -> 100600 -> 100250 -> 100850 -> 100500
        assert_eq!(m.equity_curve.len(), 5);
        assert_eq!(m.equity_curve[0], 100_000.0);
        assert!((m.equity_curve[4] - 100_500.0).abs() < 1e-6);
        // worst peak-to-trough in EUR = 350
        assert!(
            (m.max_drawdown_money - 350.0).abs() < 1e-6,
            "ddmoney {}",
            m.max_drawdown_money
        );
        // recovery = net / maxDD = 500 / 350
        assert!((m.recovery_factor - (500.0 / 350.0)).abs() < 0.01);
        // period spans 4 days
        assert!((m.period_days - 4.0).abs() < 1e-6, "days {}", m.period_days);
    }

    #[test]
    fn mc_bootstrap_is_deterministic_and_resamples_with_replacement() {
        // Audit D07: the Monte-Carlo tail-risk block bootstrap must be
        // (1) reproducible run-to-run and (2) an actual bootstrap (sampling
        // WITH replacement so the resampled equity varies) rather than a
        // permutation that always reproduces the same total.
        let analyzer = StrategyQualityAnalyzer::default();
        let day = 86_400_000_i64;
        // >=5 distinct days so the day-block path (not the trade-level
        // fallback) runs, with one large loss day that a with-replacement
        // draw can repeat into a ruinous run.
        let mk = |d: i64, pnl: f64| Trade {
            entry_time: (d + 1) * day,
            exit_time: Some((d + 1) * day + 3_600_000),
            pnl,
            pnl_pct: Some(pnl / 100_000.0),
            duration_hours: Some(1.0),
            ..Default::default()
        };
        let trades = vec![
            mk(0, 400.0),
            mk(1, -900.0),
            mk(2, 350.0),
            mk(3, -900.0),
            mk(4, 500.0),
            mk(5, -900.0),
            mk(6, 300.0),
        ];

        // (1) Reproducibility: two analyses of the same trades agree exactly.
        let a = analyzer.analyze_strategy("demo", &trades, 100_000.0);
        let b = analyzer.analyze_strategy("demo", &trades, 100_000.0);
        assert_eq!(
            a.mc_risk_of_ruin_pct, b.mc_risk_of_ruin_pct,
            "seeded bootstrap must be reproducible"
        );
        assert_eq!(a.mc_worst_drawdown_95_pct, b.mc_worst_drawdown_95_pct);

        // (2) With-replacement actually explores worse-than-observed runs:
        // the observed sequence never ruins (net is positive overall), but a
        // bootstrap that can repeat the -900 days must find a strictly
        // deeper p95 drawdown than the observed max drawdown. A permutation
        // (old code) keeps the same day set, so its p95 DD is bounded by the
        // single realized ordering's tail — the with-replacement draw beats it.
        let observed_max_dd = a.max_drawdown_pct; // fraction, same unit as mc dd
        let p95 = a.mc_worst_drawdown_95_pct.expect("mc dd present");
        assert!(
            p95 >= observed_max_dd,
            "with-replacement p95 DD ({p95}) should be >= observed max DD ({observed_max_dd})"
        );
        // And risk-of-ruin must be a real (finite) probability in [0,1].
        let ror = a.mc_risk_of_ruin_pct.expect("ror present");
        assert!((0.0..=1.0).contains(&ror), "ror out of range: {ror}");
    }

    #[test]
    fn quality_runtime_overrides_defaults_match_legacy_env_defaults() {
        let defaults = QualityRuntimeOverrides::default();
        assert_eq!(defaults.min_trades_per_month, 4);
        assert!((defaults.trading_days_per_month - 21.0).abs() < 1e-9);
    }

    #[test]
    fn quality_from_settings_default_matches_env_default() {
        // Behavior-preservation gate: a fresh `Settings` reproduces the
        // engine quality defaults exactly (config-consolidation S2c).
        let s = neoethos_core::Settings::default();
        assert_eq!(
            QualityRuntimeOverrides::from_settings(&s),
            QualityRuntimeOverrides::default()
        );
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
