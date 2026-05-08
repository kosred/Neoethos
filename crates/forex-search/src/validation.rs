use crate::artifact_io::{read_json, stable_json_hash, write_json_atomic};
use crate::eval::{
    BacktestMetrics, BacktestSettings, fast_evaluate_strategy_core, simulate_trades_core,
};
use anyhow::{Result, bail};
use forex_core::contracts::{TemporalFeatureContract, TemporalScopeHashes};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

pub const WALKFORWARD_VALIDATION_ARTIFACT_KIND: &str = "walkforward_validation_artifact";
pub const WALKFORWARD_VALIDATION_SCHEMA_VERSION: u32 = 1;
pub const CANONICAL_BACKTEST_ARTIFACT_KIND: &str = "canonical_strategy_backtest_artifact";
pub const CANONICAL_BACKTEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalBacktestScope {
    pub dataset_hash: String,
    pub evaluation_config_hash: String,
    pub strategy_hash: String,
    pub temporal_scope: TemporalScopeHashes,
}

impl CanonicalBacktestScope {
    pub fn new(
        dataset_hash: impl Into<String>,
        evaluation_config_hash: impl Into<String>,
        strategy_hash: impl Into<String>,
        temporal_contract: &TemporalFeatureContract,
    ) -> Self {
        Self {
            dataset_hash: dataset_hash.into(),
            evaluation_config_hash: evaluation_config_hash.into(),
            strategy_hash: strategy_hash.into(),
            temporal_scope: TemporalScopeHashes::from_contract(temporal_contract),
        }
    }

    pub fn from_parts<T: Serialize, U: Serialize, V: Serialize>(
        dataset: &T,
        evaluation_config: &U,
        strategy: &V,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<Self> {
        Ok(Self::new(
            stable_json_hash(dataset)?,
            stable_json_hash(evaluation_config)?,
            stable_json_hash(strategy)?,
            temporal_contract,
        ))
    }

    pub fn validate_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        self.temporal_scope
            .validate_contract(temporal_contract)
            .map_err(|err| anyhow::anyhow!("canonical backtest {err}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalBacktestArtifactFile {
    pub artifact_kind: String,
    pub artifact_schema_version: u32,
    pub scope: CanonicalBacktestScope,
    pub metrics: BacktestMetrics,
}

impl CanonicalBacktestArtifactFile {
    pub fn new(scope: CanonicalBacktestScope, metrics: BacktestMetrics) -> Self {
        Self {
            artifact_kind: CANONICAL_BACKTEST_ARTIFACT_KIND.to_string(),
            artifact_schema_version: CANONICAL_BACKTEST_SCHEMA_VERSION,
            scope,
            metrics,
        }
    }

    pub fn validate_for_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        if self.artifact_kind != CANONICAL_BACKTEST_ARTIFACT_KIND {
            bail!(
                "artifact kind {} cannot be used as a canonical backtest artifact",
                self.artifact_kind
            );
        }
        if self.artifact_schema_version != CANONICAL_BACKTEST_SCHEMA_VERSION {
            bail!(
                "unsupported canonical backtest schema version {}",
                self.artifact_schema_version
            );
        }
        self.scope.validate_temporal_contract(temporal_contract)
    }
}

pub fn write_canonical_backtest_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &CanonicalBacktestArtifactFile,
) -> Result<()> {
    write_json_atomic(path, artifact)
}

pub fn read_canonical_backtest_artifact(
    path: impl AsRef<Path>,
    temporal_contract: &TemporalFeatureContract,
) -> Result<CanonicalBacktestArtifactFile> {
    let artifact: CanonicalBacktestArtifactFile = read_json(path, "canonical backtest")?;
    artifact.validate_for_temporal_contract(temporal_contract)?;
    Ok(artifact)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WalkforwardValidationScope {
    pub dataset_hash: String,
    pub evaluation_config_hash: String,
    pub temporal_scope: TemporalScopeHashes,
}

impl WalkforwardValidationScope {
    pub fn new(
        dataset_hash: impl Into<String>,
        evaluation_config_hash: impl Into<String>,
        temporal_contract: &TemporalFeatureContract,
    ) -> Self {
        Self {
            dataset_hash: dataset_hash.into(),
            evaluation_config_hash: evaluation_config_hash.into(),
            temporal_scope: TemporalScopeHashes::from_contract(temporal_contract),
        }
    }

    pub fn from_parts<T: Serialize, U: Serialize>(
        dataset: &T,
        evaluation_config: &U,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<Self> {
        Ok(Self::new(
            stable_json_hash(dataset)?,
            stable_json_hash(evaluation_config)?,
            temporal_contract,
        ))
    }

    pub fn validate_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        self.temporal_scope
            .validate_contract(temporal_contract)
            .map_err(|err| anyhow::anyhow!("walk-forward validation {err}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkforwardValidationArtifactFile {
    pub artifact_kind: String,
    pub artifact_schema_version: u32,
    pub scope: WalkforwardValidationScope,
    pub summary: WalkforwardSummary,
}

impl WalkforwardValidationArtifactFile {
    pub fn new(scope: WalkforwardValidationScope, summary: WalkforwardSummary) -> Self {
        Self {
            artifact_kind: WALKFORWARD_VALIDATION_ARTIFACT_KIND.to_string(),
            artifact_schema_version: WALKFORWARD_VALIDATION_SCHEMA_VERSION,
            scope,
            summary,
        }
    }

    pub fn validate_for_temporal_contract(
        &self,
        temporal_contract: &TemporalFeatureContract,
    ) -> Result<()> {
        if self.artifact_kind != WALKFORWARD_VALIDATION_ARTIFACT_KIND {
            bail!(
                "artifact kind {} cannot be used as a walk-forward validation artifact",
                self.artifact_kind
            );
        }
        if self.artifact_schema_version != WALKFORWARD_VALIDATION_SCHEMA_VERSION {
            bail!(
                "unsupported walk-forward validation schema version {}",
                self.artifact_schema_version
            );
        }
        self.scope.validate_temporal_contract(temporal_contract)
    }
}

pub fn write_walkforward_validation_artifact_atomic(
    path: impl AsRef<Path>,
    artifact: &WalkforwardValidationArtifactFile,
) -> Result<()> {
    write_json_atomic(path, artifact)
}

pub fn read_walkforward_validation_artifact(
    path: impl AsRef<Path>,
    temporal_contract: &TemporalFeatureContract,
) -> Result<WalkforwardValidationArtifactFile> {
    let artifact: WalkforwardValidationArtifactFile = read_json(path, "walk-forward validation")?;
    artifact.validate_for_temporal_contract(temporal_contract)?;
    Ok(artifact)
}

pub struct WalkforwardBacktestInput<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub signals: &'a [i8],
    pub months: &'a [i64],
    pub days: &'a [i64],
    /// Real bar timestamps (ms or ns, same unit as `simulate_trades_core` expects).
    /// Used for gap detection, kill-zone rules, and day/week/month boundaries.
    pub timestamps: &'a [i64],
    pub train_ratio: f64,
    pub n_splits: usize,
    pub embargo_bars: usize,
    pub settings: &'a BacktestSettings,
    pub max_daily_loss_pct: f64,
    pub max_daily_profit_pct: f64,
    pub min_trading_days: usize,
    pub max_trades_per_day: usize,
    /// Starting account balance used to convert absolute PnL into daily return %.
    pub initial_balance: f64,
}

#[derive(Debug, Clone, Default)]
struct WalkforwardRiskDiagnostics {
    max_consec_losses: usize,
    daily_min_dd: f64,
    max_daily_loss: f64,
    daily_loss_breach: bool,
    consistency_violation: bool,
    trade_limit_violation: bool,
    min_trading_days_ok: bool,
    daily_returns: Vec<f64>,
    max_daily_dd_pct: f64,
    prop_compliant: bool,
}

fn normalized_pct_threshold(value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        0.0
    } else if value > 1.0 {
        value / 100.0
    } else {
        value
    }
}

#[allow(clippy::too_many_arguments)]
fn walkforward_risk_diagnostics(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    signals: &[i8],
    days: &[i64],
    timestamps: &[i64],
    settings: &BacktestSettings,
    evaluator_max_daily_dd: f64,
    max_daily_loss_pct: f64,
    max_daily_profit_pct: f64,
    min_trading_days: usize,
    max_trades_per_day: usize,
    initial_balance: f64,
) -> WalkforwardRiskDiagnostics {
    if close.is_empty() || days.is_empty() {
        return WalkforwardRiskDiagnostics::default();
    }
    let initial_balance = if initial_balance.is_finite() && initial_balance > 0.0 {
        initial_balance
    } else {
        100_000.0
    };

    let mut day_offsets = BTreeMap::<i64, usize>::new();
    let mut daily_pnl = Vec::<f64>::new();
    let mut daily_trade_counts = Vec::<usize>::new();
    for &day in days {
        day_offsets.entry(day).or_insert_with(|| {
            let offset = daily_pnl.len();
            daily_pnl.push(0.0);
            daily_trade_counts.push(0);
            offset
        });
    }

    // Use real timestamps so simulate_trades_core applies correct gap/session/kill-zone logic.
    let ts = if timestamps.len() == close.len() {
        timestamps
    } else {
        days
    };
    let trades = simulate_trades_core(close, high, low, ts, signals, settings);
    let mut max_consec_losses = 0usize;
    let mut current_consec_losses = 0usize;

    for trade in &trades {
        if trade.pnl < 0.0 {
            current_consec_losses += 1;
            max_consec_losses = max_consec_losses.max(current_consec_losses);
        } else if trade.pnl > 0.0 {
            current_consec_losses = 0;
        }

        let exit_day = trade.exit_time.unwrap_or(trade.entry_time);
        let offset = if let Some(&offset) = day_offsets.get(&exit_day) {
            offset
        } else {
            let offset = daily_pnl.len();
            day_offsets.insert(exit_day, offset);
            daily_pnl.push(0.0);
            daily_trade_counts.push(0);
            offset
        };
        daily_pnl[offset] += trade.pnl;
        daily_trade_counts[offset] += 1;
    }

    let daily_returns: Vec<f64> = daily_pnl.iter().map(|pnl| pnl / initial_balance).collect();
    let daily_min_return = daily_returns.iter().copied().fold(0.0, f64::min);
    let closed_trade_daily_loss = daily_returns
        .iter()
        .filter(|ret| **ret < 0.0)
        .map(|ret| ret.abs())
        .fold(0.0, f64::max);
    let evaluator_max_daily_dd = if evaluator_max_daily_dd.is_finite() {
        evaluator_max_daily_dd.max(0.0)
    } else {
        0.0
    };
    let max_daily_loss = closed_trade_daily_loss.max(evaluator_max_daily_dd);
    let daily_min_dd = daily_min_return.min(-evaluator_max_daily_dd);

    let max_daily_loss_limit = normalized_pct_threshold(max_daily_loss_pct);
    let daily_loss_breach = max_daily_loss_limit > 0.0 && max_daily_loss >= max_daily_loss_limit;

    let profit_consistency_limit = normalized_pct_threshold(max_daily_profit_pct);
    let total_positive_daily_pnl: f64 = daily_pnl.iter().filter(|pnl| **pnl > 0.0).sum();
    let largest_positive_daily_pnl = daily_pnl.iter().copied().fold(0.0, f64::max);
    let largest_profit_share = if total_positive_daily_pnl > f64::EPSILON {
        largest_positive_daily_pnl / total_positive_daily_pnl
    } else {
        0.0
    };
    let consistency_violation =
        profit_consistency_limit > 0.0 && largest_profit_share > profit_consistency_limit;

    let trade_limit_violation = max_trades_per_day > 0
        && daily_trade_counts
            .iter()
            .any(|&count| count > max_trades_per_day);
    let trading_days = daily_trade_counts
        .iter()
        .filter(|&&count| count > 0)
        .count();
    let min_trading_days_ok = min_trading_days == 0 || trading_days >= min_trading_days;
    let prop_compliant = !daily_loss_breach
        && !consistency_violation
        && !trade_limit_violation
        && min_trading_days_ok;

    WalkforwardRiskDiagnostics {
        max_consec_losses,
        daily_min_dd,
        max_daily_loss,
        daily_loss_breach,
        consistency_violation,
        trade_limit_violation,
        min_trading_days_ok,
        daily_returns,
        max_daily_dd_pct: max_daily_loss,
        prop_compliant,
    }
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
        timestamps,
        train_ratio,
        n_splits,
        embargo_bars,
        settings,
        max_daily_loss_pct,
        max_daily_profit_pct,
        min_trading_days,
        max_trades_per_day,
        initial_balance,
    } = input;
    let n = close.len();
    if n == 0
        || high.len() != n
        || low.len() != n
        || signals.len() != n
        || months.len() != n
        || days.len() != n
    {
        bail!("empty data or length mismatch");
    }
    if n_splits == 0 {
        bail!("n_splits must be greater than zero");
    }
    if !train_ratio.is_finite() || !(0.0..1.0).contains(&train_ratio) {
        bail!("train_ratio must be finite and in the open interval (0, 1)");
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
        let slice_ts = if timestamps.len() == n {
            &timestamps[test_start..end]
        } else {
            slice_days
        };

        let metrics = fast_evaluate_strategy_core(
            slice_close,
            slice_high,
            slice_low,
            slice_sig,
            slice_months,
            slice_days,
            &[],
            settings,
        );

        // Map metrics [net_profit, 0.0, peak_equity, max_dd, win_rate, pf, expectancy, 0.0, trade_count, consistency, max_daily_dd]
        let net_profit = metrics[0];
        let max_dd = metrics[3];
        let win_rate = metrics[4];
        let trade_count = metrics[8] as usize;
        let max_daily_dd = metrics[10];
        let risk = walkforward_risk_diagnostics(
            slice_close,
            slice_high,
            slice_low,
            slice_sig,
            slice_days,
            slice_ts,
            settings,
            max_daily_dd,
            max_daily_loss_pct,
            max_daily_profit_pct,
            min_trading_days,
            max_trades_per_day,
            initial_balance,
        );

        let res = WalkforwardSplitResult {
            split: i + 1,
            trades: trade_count,
            pnl: net_profit,
            win_rate,
            max_dd,
            max_consec_losses: risk.max_consec_losses,
            daily_min_dd: risk.daily_min_dd,
            max_daily_loss: risk.max_daily_loss,
            daily_loss_breach: risk.daily_loss_breach,
            consistency_violation: risk.consistency_violation,
            trade_limit_violation: risk.trade_limit_violation,
            min_trading_days_ok: risk.min_trading_days_ok,
            daily_returns: risk.daily_returns,
            max_daily_dd_pct: risk.max_daily_dd_pct,
            prop_compliant: risk.prop_compliant,
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
    let avg_max_consec_losses = split_results
        .iter()
        .map(|r| r.max_consec_losses as f64)
        .sum::<f64>()
        / n_res;
    let avg_daily_min_dd = split_results.iter().map(|r| r.daily_min_dd).sum::<f64>() / n_res;
    let avg_max_daily_loss = split_results.iter().map(|r| r.max_daily_loss).sum::<f64>() / n_res;

    Ok(WalkforwardSummary {
        walk_forward_splits: split_results.len(),
        avg_pnl,
        avg_win_rate: avg_win,
        avg_max_dd: avg_dd,
        avg_max_consec_losses,
        avg_daily_min_dd,
        avg_max_daily_loss,
        any_daily_loss_breach: split_results.iter().any(|r| r.daily_loss_breach),
        any_consistency_violation: split_results.iter().any(|r| r.consistency_violation),
        any_trade_limit_violation: split_results.iter().any(|r| r.trade_limit_violation),
        all_min_trading_days_ok: split_results.iter().all(|r| r.min_trading_days_ok),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temporal_contract(label_policy_hash: &str) -> TemporalFeatureContract {
        TemporalFeatureContract::strict_live(
            "UTC",
            "alignment-policy-v1",
            label_policy_hash,
            "walk-forward-policy-v1",
            "live-readiness-policy-v1",
        )
        .expect("strict temporal contract should be valid")
    }

    fn sample_summary() -> WalkforwardSummary {
        WalkforwardSummary {
            walk_forward_splits: 1,
            avg_pnl: 12.0,
            avg_win_rate: 0.5,
            avg_max_dd: 0.1,
            avg_max_consec_losses: 1.0,
            avg_daily_min_dd: -0.01,
            avg_max_daily_loss: 0.01,
            any_daily_loss_breach: false,
            any_consistency_violation: false,
            any_trade_limit_violation: false,
            all_min_trading_days_ok: true,
            splits: Vec::new(),
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("forex-validation-{name}-{unique}.json"))
    }

    fn flat_settings() -> BacktestSettings {
        BacktestSettings {
            sl_pips: 1_000_000.0,
            tp_pips: 1_000_000.0,
            max_hold_bars: 1,
            min_hold_bars: 1,
            max_trades_per_day: 0,
            gap_threshold_ms: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: 1.0,
            spread_pips: 0.0,
            commission_per_trade: 0.0,
            pip_value_per_lot: 10_000.0,
            kill_zones_enabled: false,
        }
    }

    #[test]
    fn risk_diagnostics_enforce_prop_constraints_from_simulated_trades() {
        let close = [100.0, 101.0, 103.0, 102.0, 100.0, 99.0, 98.0];
        let high = close;
        let low = close;
        let signals = [1, 0, 1, 0, 1, 0, 0];
        let days = [1, 1, 1, 2, 2, 2, 2];

        let risk = walkforward_risk_diagnostics(
            &close,
            &high,
            &low,
            &signals,
            &days,
            &days,
            &flat_settings(),
            0.0,
            0.01,
            0.50,
            3,
            1,
            100_000.0,
        );

        assert_eq!(risk.max_consec_losses, 2);
        assert!(risk.daily_loss_breach);
        assert!(risk.consistency_violation);
        assert!(risk.trade_limit_violation);
        assert!(!risk.min_trading_days_ok);
        assert!(!risk.prop_compliant);
        assert_eq!(risk.daily_returns.len(), 2);
    }

    #[test]
    fn walkforward_validation_artifact_binds_temporal_scope() {
        let contract = temporal_contract("label-policy-v1");
        let scope = WalkforwardValidationScope::new("dataset-a", "eval-a", &contract);
        let artifact = WalkforwardValidationArtifactFile::new(scope.clone(), sample_summary());

        assert_eq!(artifact.artifact_kind, WALKFORWARD_VALIDATION_ARTIFACT_KIND);
        assert_eq!(artifact.scope, scope);
        artifact
            .validate_for_temporal_contract(&contract)
            .expect("matching temporal contract should validate");
    }

    #[test]
    fn walkforward_validation_artifact_rejects_temporal_drift_and_wrong_kind() {
        let contract = temporal_contract("label-policy-v1");
        let changed_contract = temporal_contract("label-policy-v2");
        let scope = WalkforwardValidationScope::new("dataset-a", "eval-a", &contract);
        let mut artifact = WalkforwardValidationArtifactFile::new(scope, sample_summary());

        let err = artifact
            .validate_for_temporal_contract(&changed_contract)
            .expect_err("changed temporal contract must not validate");
        assert!(err.to_string().contains("temporal_contract_hash"));

        artifact.artifact_kind = "search_checkpoint_artifact".to_string();
        let err = artifact
            .validate_for_temporal_contract(&contract)
            .expect_err("wrong artifact kind must not validate");
        assert!(err.to_string().contains("cannot be used as a walk-forward"));
    }

    #[test]
    fn walkforward_validation_artifact_uses_shared_atomic_io() {
        let contract = temporal_contract("label-policy-v1");
        let scope = WalkforwardValidationScope::new("dataset-a", "eval-a", &contract);
        let artifact = WalkforwardValidationArtifactFile::new(scope, sample_summary());
        let path = temp_path("artifact");

        write_walkforward_validation_artifact_atomic(&path, &artifact)
            .expect("atomic validation artifact write should succeed");
        let loaded = read_walkforward_validation_artifact(&path, &contract)
            .expect("matching validation artifact should load");
        assert_eq!(loaded.artifact_kind, WALKFORWARD_VALIDATION_ARTIFACT_KIND);
        assert_eq!(loaded.summary.walk_forward_splits, 1);

        let changed_contract = temporal_contract("label-policy-v2");
        let err = read_walkforward_validation_artifact(&path, &changed_contract)
            .expect_err("temporal drift must reject persisted validation artifact");
        assert!(err.to_string().contains("temporal_contract_hash"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn backtest_metrics_preserve_canonical_metric_layout() {
        let raw = [
            12.0, 1.5, 100_012.0, 0.02, 0.60, 1.8, 4.0, 0.0, 7.0, 0.9, 0.01,
        ];
        let metrics = BacktestMetrics::from_metric_array(raw);

        assert_eq!(metrics.net_profit, 12.0);
        assert_eq!(metrics.sharpe, 1.5);
        assert_eq!(metrics.trade_count, 7);
        assert_eq!(metrics.to_metric_array(), raw);
    }

    #[test]
    fn canonical_backtest_artifact_uses_shared_atomic_io_and_temporal_scope() {
        let contract = temporal_contract("label-policy-v1");
        let scope = CanonicalBacktestScope::new("dataset-a", "eval-a", "strategy-a", &contract);
        let artifact = CanonicalBacktestArtifactFile::new(
            scope,
            BacktestMetrics::from_metric_array([
                12.0, 1.5, 100_012.0, 0.02, 0.60, 1.8, 4.0, 0.0, 7.0, 0.9, 0.01,
            ]),
        );
        let path = temp_path("canonical-backtest");

        write_canonical_backtest_artifact_atomic(&path, &artifact)
            .expect("atomic canonical backtest artifact write should succeed");
        let loaded = read_canonical_backtest_artifact(&path, &contract)
            .expect("matching canonical backtest artifact should load");
        assert_eq!(loaded.artifact_kind, CANONICAL_BACKTEST_ARTIFACT_KIND);
        assert_eq!(loaded.metrics.trade_count, 7);

        let changed_contract = temporal_contract("label-policy-v2");
        let err = read_canonical_backtest_artifact(&path, &changed_contract)
            .expect_err("temporal drift must reject persisted backtest artifact");
        assert!(err.to_string().contains("temporal_contract_hash"));

        let _ = std::fs::remove_file(path);
    }
}
