use anyhow::{Result, bail};
use crate::eval::{BacktestSettings, fast_evaluate_strategy_core};
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

pub fn embargoed_walkforward_backtest(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    signals: &[i8],
    months: &[i64],
    days: &[i64],
    train_ratio: f64,
    n_splits: usize,
    embargo_bars: usize,
    settings: &BacktestSettings,
    max_daily_loss_pct: f64,
    _max_daily_profit_pct: f64,
    _min_trading_days: usize,
    _max_trades_per_day: usize,
) -> Result<WalkforwardSummary> {
    let n = close.len();
    if n == 0 || signals.len() != n {
        bail!("empty data or length mismatch");
    }

    let window = (n / n_splits).max(1);
    let mut split_results = Vec::new();

    for i in 0..n_splits {
        let start = i * window;
        let end = ((i + 1) * window).min(n);
        if end - start < 80 { break; }

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
            slice_close, slice_high, slice_low, slice_sig, slice_months, slice_days, settings
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
            min_trading_days_ok: true, // Simplified
            daily_returns: Vec::new(),
            max_daily_dd_pct: max_daily_dd,
            prop_compliant: max_daily_dd < 0.05,
        };
        split_results.push(res);
    }

    if split_results.is_empty() {
        return Ok(WalkforwardSummary {
            walk_forward_splits: 0,
            avg_pnl: 0.0, avg_win_rate: 0.0, avg_max_dd: 0.0, avg_max_consec_losses: 0.0,
            avg_daily_min_dd: 0.0, avg_max_daily_loss: 0.0, any_daily_loss_breach: false,
            any_consistency_violation: false, any_trade_limit_violation: false,
            all_min_trading_days_ok: false, splits: Vec::new(),
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
        Self { n_splits, n_test_groups, embargo_pct, purge_pct }
    }

    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        if n_samples == 0 { return Vec::new(); }

        let purge_size = (n_samples as f64 * self.purge_pct).ceil() as usize;
        let embargo_size = (n_samples as f64 * self.embargo_pct).ceil() as usize;

        let warmup_size = (n_samples / self.n_splits).max(purge_size + embargo_size).max(self.n_splits);
        let warmup_size = if warmup_size + self.n_splits >= n_samples {
            (n_samples / (self.n_splits + 1)).max(1)
        } else { warmup_size };

        let cv_start = (warmup_size + embargo_size).min(n_samples);
        let cv_len = n_samples - cv_start;
        if cv_len < self.n_splits { return Vec::new(); }

        let group_size = cv_len / self.n_splits;
        let mut groups = Vec::new();
        for i in 0..self.n_splits {
            let start = cv_start + i * group_size;
            let end = if i == self.n_splits - 1 { n_samples } else { cv_start + (i + 1) * group_size };
            groups.push((start..end).collect::<Vec<usize>>());
        }

        let mut splits = Vec::new();
        for combination in (0..self.n_splits).combinations(self.n_test_groups) {
            let mut test_idx = Vec::new();
            for &i in &combination {
                test_idx.extend(&groups[i]);
            }
            test_idx.sort_unstable();

            let earliest_group: usize = *combination.iter().min().unwrap();
            let mut train_idx: Vec<usize> = (0..warmup_size).collect();
            if earliest_group > 0 {
                for i in 0..earliest_group {
                    if !combination.contains(&i) {
                        train_idx.extend(&groups[i]);
                    }
                }
            }
            train_idx.sort_unstable();

            // Purge and Embargo
            if !test_idx.is_empty() && !train_idx.is_empty() {
                let test_start: usize = test_idx[0];
                let purge_threshold = test_start.saturating_sub(purge_size);
                train_idx.retain(|&i| i < purge_threshold);

                if !train_idx.is_empty() {
                    let train_end = *train_idx.last().unwrap();
                    let embargo_threshold = (train_end + embargo_size).min(n_samples);
                    test_idx.retain(|&i| i >= embargo_threshold);
                }
            }

            splits.push((train_idx, test_idx));
        }

        splits
    }
}
