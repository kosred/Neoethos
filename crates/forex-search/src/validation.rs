use crate::eval::{BacktestSettings, fast_evaluate_strategy_core};
use anyhow::{Result, bail};
use itertools::Itertools;

#[derive(Debug, Clone, serde::Serialize)]
pub struct WalkforwardSplitResult {
    pub split: usize,
    pub trades: usize,
    pub pnl: f64,
    pub win_rate: f64,
    pub max_dd: f64,
    pub max_consec_losses: usize,
    pub daily_min_dd: f64,
    pub max_daily_loss: f64,
    pub daily_loss_breach: bool,
    pub consistency_violation: bool,
    pub trade_limit_violation: bool,
    pub min_trading_days_ok: bool,
    pub daily_returns: Vec<f64>,
    pub max_daily_dd_pct: f64,
    pub prop_compliant: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WalkforwardSummary {
    pub walk_forward_splits: usize,
    pub avg_pnl: f64,
    pub avg_win_rate: f64,
    pub avg_max_dd: f64,
    pub avg_max_consec_losses: f64,
    pub avg_daily_min_dd: f64,
    pub avg_max_daily_loss: f64,
    pub any_daily_loss_breach: bool,
    pub any_consistency_violation: bool,
    pub any_trade_limit_violation: bool,
    pub all_min_trading_days_ok: bool,
    pub splits: Vec<WalkforwardSplitResult>,
}

pub struct WalkforwardBacktestInput<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub signals: &'a [i8],
    pub months: &'a [i64],
    pub days: &'a [i64],
    pub train_ratio: f64,
    pub n_splits: usize,
    pub embargo_bars: usize,
    pub settings: &'a BacktestSettings,
    pub max_daily_loss_pct: f64,
    pub max_daily_profit_pct: f64,
    pub min_trading_days: usize,
    pub max_trades_per_day: usize,
}

pub fn embargoed_walkforward_backtest(
    input: WalkforwardBacktestInput<'_>,
) -> Result<WalkforwardSummary> {
    let WalkforwardBacktestInput {
        close,
        high,
        low,
        signals,
        months,
        days,
        train_ratio,
        n_splits,
        embargo_bars,
        settings,
        max_daily_loss_pct,
        max_daily_profit_pct: _max_daily_profit_pct,
        min_trading_days: _min_trading_days,
        max_trades_per_day: _max_trades_per_day,
    } = input;
    let n = close.len();
    if n == 0 || signals.len() != n {
        bail!("empty data or length mismatch");
    }

    let window = (n / n_splits).max(1);
    let mut split_results = Vec::new();

    for i in 0..n_splits {
        let start = i * window;
        let end = ((i + 1) * window).min(n);
        if end - start < 80 {
            break;
        }

        let train_end = start + ((window as f64) * train_ratio) as usize;
        let test_start = train_end + embargo_bars;

        if test_start >= end || (train_end - start) < 40 || (end - test_start) < 40 {
            continue;
        }

        let slice_close = &close[test_start..end];
        let slice_high = &high[test_start..end];
        let slice_low = &low[test_start..end];
        let slice_sig = &signals[test_start..end];
        let slice_months = &months[test_start..end];
        let slice_days = &days[test_start..end];

        let metrics = fast_evaluate_strategy_core(
            slice_close,
            slice_high,
            slice_low,
            slice_sig,
            slice_months,
            slice_days,
            settings,
        );

        // Map metrics [net_profit, 0.0, peak_equity, max_dd, win_rate, pf, expectancy, 0.0, trade_count, consistency, max_daily_dd]
        let net_profit = metrics[0];
        let max_dd = metrics[3];
        let win_rate = metrics[4];
        let trade_count = metrics[8] as usize;
        let max_daily_dd = metrics[10];

        let res = WalkforwardSplitResult {
            split: i + 1,
            trades: trade_count,
            pnl: net_profit,
            win_rate,
            max_dd,
            max_consec_losses: 0, // Simplified for now
            daily_min_dd: -max_daily_dd,
            max_daily_loss: -max_daily_dd,
            daily_loss_breach: max_daily_dd >= max_daily_loss_pct,
            consistency_violation: false, // Simplified
            trade_limit_violation: false, // Simplified
            min_trading_days_ok: true,    // Simplified
            daily_returns: Vec::new(),
            max_daily_dd_pct: max_daily_dd,
            prop_compliant: max_daily_dd < 0.05,
        };
        split_results.push(res);
    }

    if split_results.is_empty() {
        return Ok(WalkforwardSummary {
            walk_forward_splits: 0,
            avg_pnl: 0.0,
            avg_win_rate: 0.0,
            avg_max_dd: 0.0,
            avg_max_consec_losses: 0.0,
            avg_daily_min_dd: 0.0,
            avg_max_daily_loss: 0.0,
            any_daily_loss_breach: false,
            any_consistency_violation: false,
            any_trade_limit_violation: false,
            all_min_trading_days_ok: false,
            splits: Vec::new(),
        });
    }

    let n_res = split_results.len() as f64;
    let avg_pnl = split_results.iter().map(|r| r.pnl).sum::<f64>() / n_res;
    let avg_win = split_results.iter().map(|r| r.win_rate).sum::<f64>() / n_res;
    let avg_dd = split_results.iter().map(|r| r.max_dd).sum::<f64>() / n_res;

    Ok(WalkforwardSummary {
        walk_forward_splits: split_results.len(),
        avg_pnl,
        avg_win_rate: avg_win,
        avg_max_dd: avg_dd,
        avg_max_consec_losses: 0.0,
        avg_daily_min_dd: 0.0,
        avg_max_daily_loss: 0.0,
        any_daily_loss_breach: split_results.iter().any(|r| r.daily_loss_breach),
        any_consistency_violation: false,
        any_trade_limit_violation: false,
        all_min_trading_days_ok: true,
        splits: split_results,
    })
}

pub struct CombinatorialPurgedCV {
    pub n_splits: usize,
    pub n_test_groups: usize,
    pub embargo_pct: f64,
    pub purge_pct: f64,
}

impl CombinatorialPurgedCV {
    pub fn new(n_splits: usize, n_test_groups: usize, embargo_pct: f64, purge_pct: f64) -> Self {
        Self {
            n_splits,
            n_test_groups,
            embargo_pct,
            purge_pct,
        }
    }

    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        if n_samples == 0 || self.n_splits < 2 {
            return Vec::new();
        }

        // Divide n_samples into S groups
        let group_size = n_samples / self.n_splits;
        if group_size == 0 {
            return Vec::new();
        }

        let mut groups = Vec::with_capacity(self.n_splits);
        for i in 0..self.n_splits {
            let start = i * group_size;
            let end = if i == self.n_splits - 1 {
                n_samples
            } else {
                (i + 1) * group_size
            };
            groups.push(start..end);
        }

        let purge_size = (n_samples as f64 * self.purge_pct).ceil() as usize;
        let embargo_size = (n_samples as f64 * self.embargo_pct).ceil() as usize;

        let mut results = Vec::new();

        // Form all combinations of k test groups
        for combination in (0..self.n_splits).combinations(self.n_test_groups) {
            let mut test_idx = Vec::new();
            let mut candidate_train_groups = Vec::new();

            for (i, group) in groups.iter().enumerate().take(self.n_splits) {
                if combination.contains(&i) {
                    test_idx.extend(group.clone());
                } else {
                    candidate_train_groups.push(i);
                }
            }

            let mut train_idx = Vec::new();

            // For each training group, apply purging and embargoing relative to ALL test groups
            for &g_idx in &candidate_train_groups {
                let group_range = groups[g_idx].clone();
                let group_start = group_range.start;
                let group_end = group_range.end;

                let mut group_valid_start = group_start;
                let mut group_valid_end = group_end;

                for &t_idx in &combination {
                    let test_range = groups[t_idx].clone();

                    // 1. Purge: if training group is BEFORE a test group,
                    // remove samples at the end of training group that look into the test group.
                    if group_end <= test_range.start {
                        let potential_end = test_range.start.saturating_sub(purge_size);
                        if potential_end < group_valid_end && potential_end >= group_start {
                            group_valid_end = potential_end;
                        }
                    }

                    // 2. Embargo: if training group is AFTER a test group,
                    // remove samples at the beginning of training group that are serially correlated.
                    if group_start >= test_range.end {
                        let potential_start = test_range.end + embargo_size;
                        if potential_start > group_valid_start && potential_start <= group_end {
                            group_valid_start = potential_start;
                        }
                    }
                }

                if group_valid_start < group_valid_end {
                    train_idx.extend(group_valid_start..group_valid_end);
                }
            }

            if !test_idx.is_empty() && !train_idx.is_empty() {
                results.push((train_idx, test_idx));
            }
        }

        results
    }
}
