//! Detached, bounded combined-strategy/model screening on the canonical CPU
//! lifecycle. These are OHLC/assumption results, never quote fills or promotion.

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use super::{BacktestMetrics, BacktestSettings};
use crate::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchRunInputV2, CanonicalSearchWindowRoleV1,
    CanonicalTrendbarResearchExecutionContractV3, EvaluationConfig,
};

/// All decision fields belong to the same just-closed signal row. The next
/// bar's close is the existing CPU screening fill convention, not a live quote.
#[derive(Clone, Copy, Debug)]
pub struct NettedBarDecisionTapeV1<'a> {
    pub signals: &'a [i8],
    pub confidences: &'a [f64],
    pub sl_pips: &'a [f64],
    pub tp_pips: &'a [f64],
    pub ml_multipliers: &'a [f64],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BarReplayLotGridV1 {
    min_lots: f64,
    max_lots: f64,
    lot_step: f64,
}

impl BarReplayLotGridV1 {
    pub fn new(min_lots: f64, max_lots: f64, lot_step: f64) -> Result<Self> {
        ensure!(
            [min_lots, max_lots, lot_step]
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
                && min_lots <= max_lots
                && lot_step <= max_lots,
            "invalid explicit broker lot grid"
        );
        Ok(Self {
            min_lots,
            max_lots,
            lot_step,
        })
    }

    pub(super) fn normalize_down(self, requested: f64) -> Option<f64> {
        let capped = requested.min(self.max_lots);
        let value = (capped / self.lot_step).floor() * self.lot_step;
        (requested.is_finite()
            && requested > 0.0
            && value.is_finite()
            && value >= self.min_lots
            && value <= capped)
            .then_some(value)
    }
}

/// An open modeled position is NOT a completed trade. Gross mark and pending
/// commission remain separate; no exit quote, swap settlement or liquidation is
/// invented. `marked_equity_before_pending_costs` is not broker account equity.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TerminalBarPositionV1 {
    pub direction: i8,
    pub entry_bar_index: usize,
    pub entry_timestamp_ms: i64,
    pub modeled_entry_price: f64,
    pub lots: f64,
    /// Initial stop distance captured at entry; an active trail may tighten it.
    pub stop_pips: f64,
    pub target_pips: f64,
    pub active_trailing_stop_price: Option<f64>,
    pub mark_timestamp_ms: i64,
    pub mark_close_price: f64,
    pub gross_unrealized_account: f64,
    pub pending_round_trip_commission_account: f64,
    pub marked_equity_before_pending_costs: f64,
}

#[derive(Debug, Serialize)]
pub struct NettedBarResearchResultV1 {
    pub execution_basis: &'static str,
    pub promotion_eligible: bool,
    pub scope_identity_sha256: String,
    pub cost_contract_identity_sha256: String,
    pub account_currency: String,
    pub metrics: BacktestMetrics,
    pub closed_trades: Vec<crate::quality::Trade>,
    pub ending_realized_balance: f64,
    pub terminal_open: Option<TerminalBarPositionV1>,
    pub below_min_entries: usize,
    /// None explicitly means unconstrained fractional research lots, not an
    /// assertion that the broker accepts every modeled quantity.
    pub lot_grid: Option<BarReplayLotGridV1>,
}

pub(super) struct NettedBarCoreState<'a> {
    pub tape: NettedBarDecisionTapeV1<'a>,
    pub lot_grid: Option<BarReplayLotGridV1>,
    pub month_capacity: usize,
    pub below_min_entries: usize,
    pub ending_realized_balance: f64,
    pub terminal_open: Option<TerminalBarPositionV1>,
    pub nonfinite_metrics: bool,
}

/// Process-local detached preparation. Private fields and no deserializer/raw
/// public constructor: preparation proves the existing canonical input first,
/// then retains only the held-out OHLC/timestamps and compact policy metadata.
/// The multi-gigabyte Search cube can be released before model preparation.
pub struct PreparedNettedCanonicalBarResearchV1 {
    close: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    timestamps: Vec<i64>,
    months: Vec<i64>,
    days: Vec<i64>,
    settings: BacktestSettings,
    account_currency: String,
    scope_identity_sha256: String,
    cost_contract_identity_sha256: String,
}

impl PreparedNettedCanonicalBarResearchV1 {
    pub fn from_input(
        input: &CanonicalSearchRunInputV2<'_>,
        scope: &CanonicalSearchArtifactScopeV2,
        contract: &CanonicalTrendbarResearchExecutionContractV3,
        evaluation: &EvaluationConfig,
    ) -> Result<Self> {
        scope.validate()?;
        contract.validate_against_input(input)?;
        ensure!(
            scope.receipt() == input.receipt()
                && scope.evaluated_window().role() == CanonicalSearchWindowRoleV1::Holdout,
            "netted bar research requires the exact input's locked holdout scope"
        );
        let timestamps = input
            .ohlcv()
            .timestamp
            .as_deref()
            .context("canonical base has no timestamps")?;
        let start = timestamps
            .binary_search(&scope.evaluated_window().timestamp_start_ms())
            .map_err(|_| anyhow::anyhow!("holdout start is absent from the canonical input"))?;
        let end = timestamps
            .binary_search(&scope.evaluated_window().timestamp_end_ms())
            .map_err(|_| anyhow::anyhow!("holdout end is absent from the canonical input"))?
            + 1;
        let actual = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::Holdout,
            input,
            start..end,
        )?;
        ensure!(
            &actual == scope,
            "holdout row range differs from its exact canonical source"
        );
        contract.validate_evaluation_costs(evaluation)?;
        let settings = crate::genetic::search_engine::evaluation_backtest_settings(evaluation)?;
        Self::from_checked_parts(
            input.ohlcv().close[start..end].to_vec(),
            input.ohlcv().high[start..end].to_vec(),
            input.ohlcv().low[start..end].to_vec(),
            timestamps[start..end].to_vec(),
            settings,
            evaluation.account_currency.clone(),
            scope.identity_sha256()?,
            contract.identity_sha256()?,
        )
    }

    fn from_checked_parts(
        close: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        timestamps: Vec<i64>,
        settings: BacktestSettings,
        account_currency: String,
        scope_identity_sha256: String,
        cost_contract_identity_sha256: String,
    ) -> Result<Self> {
        let n = close.len();
        ensure!(
            n >= 2 && high.len() == n && low.len() == n && timestamps.len() == n,
            "netted bar preparation requires complete aligned OHLC with at least two rows"
        );
        for row in 0..n {
            ensure!(
                timestamps[row] >= 0
                    && (row == 0 || timestamps[row] > timestamps[row - 1])
                    && [close[row], high[row], low[row]]
                        .iter()
                        .all(|v| v.is_finite() && *v > 0.0)
                    && low[row] <= close[row]
                    && high[row] >= close[row],
                "invalid canonical netted bar at row {row}"
            );
        }
        let (months, days) = crate::genetic::search_engine::month_day_indices(&timestamps);
        Ok(Self {
            close,
            high,
            low,
            timestamps,
            months,
            days,
            settings,
            account_currency,
            scope_identity_sha256,
            cost_contract_identity_sha256,
        })
    }

    pub fn timestamps(&self) -> &[i64] {
        &self.timestamps
    }

    pub fn evaluate(
        &self,
        tape: NettedBarDecisionTapeV1<'_>,
        lot_grid: Option<BarReplayLotGridV1>,
    ) -> Result<NettedBarResearchResultV1> {
        let n = self.close.len();
        ensure!(
            [
                tape.signals.len(),
                tape.confidences.len(),
                tape.sl_pips.len(),
                tape.tp_pips.len(),
                tape.ml_multipliers.len()
            ]
            .iter()
            .all(|length| *length == n),
            "netted decision tape must exactly match held-out rows"
        );
        super::validate_sizing_confidences(n, tape.confidences, &self.settings)?;
        for row in 0..n {
            ensure!(
                (-1..=1).contains(&tape.signals[row])
                    && tape.ml_multipliers[row].is_finite()
                    && (0.0..=1.0).contains(&tape.ml_multipliers[row])
                    && [tape.sl_pips[row], tape.tp_pips[row]]
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.0)
                    && (tape.signals[row] == 0
                        || (tape.sl_pips[row] > 0.0 && tape.tp_pips[row] > 0.0)),
                "invalid netted direction/brackets/ML multiplier at row {row}"
            );
            ensure!(
                [
                    tape.sl_pips[row] * self.settings.pip_value_per_lot,
                    tape.tp_pips[row] * self.settings.pip_value_per_lot,
                    tape.sl_pips[row] * self.settings.pip_value,
                    tape.tp_pips[row] * self.settings.pip_value,
                ]
                .iter()
                .all(|value| value.is_finite()),
                "netted bracket/account risk arithmetic overflow at row {row}"
            );
        }
        let mut state = NettedBarCoreState {
            tape,
            lot_grid,
            month_capacity: 1 + self
                .months
                .windows(2)
                .filter(|pair| pair[0] != pair[1])
                .count(),
            below_min_entries: 0,
            ending_realized_balance: self.settings.initial_equity(),
            terminal_open: None,
            nonfinite_metrics: false,
        };
        let mut trades = Vec::new();
        let metrics = super::evaluate_strategy_with_ledger_core::<true>(
            &self.close,
            &self.high,
            &self.low,
            tape.signals,
            tape.confidences,
            &self.months,
            &self.days,
            &self.timestamps,
            &self.settings,
            &mut trades,
            Some(&mut state),
        );
        ensure!(
            !state.nonfinite_metrics
                && state.ending_realized_balance.is_finite()
                && trades
                    .iter()
                    .all(|trade| [trade.pnl, trade.mfe, trade.mae, trade.r_multiple]
                        .iter()
                        .all(|v| v.is_finite())),
            "netted bar account arithmetic overflow; no sanitized statistics accepted"
        );
        if let Some(open) = &state.terminal_open {
            ensure!(
                [
                    open.modeled_entry_price,
                    open.lots,
                    open.stop_pips,
                    open.target_pips,
                    open.mark_close_price,
                    open.gross_unrealized_account,
                    open.pending_round_trip_commission_account,
                    open.marked_equity_before_pending_costs
                ]
                .iter()
                .all(|v| v.is_finite())
                    && open.active_trailing_stop_price.is_none_or(f64::is_finite),
                "terminal open-position mark overflow"
            );
        }
        Ok(NettedBarResearchResultV1 {
            execution_basis: "canonical_cpu_ohlc_screening_prior_bar_signal_next_close_fill; scalar_account_pip_and_cost_assumptions; closed_trade_realized_pnl; open_gross_mark_before_pending_costs",
            promotion_eligible: false,
            scope_identity_sha256: self.scope_identity_sha256.clone(),
            cost_contract_identity_sha256: self.cost_contract_identity_sha256.clone(),
            account_currency: self.account_currency.clone(),
            metrics: metrics.into(),
            closed_trades: trades,
            ending_realized_balance: state.ending_realized_balance,
            terminal_open: state.terminal_open,
            below_min_entries: state.below_min_entries,
            lot_grid,
        })
    }
}

#[cfg(test)]
#[path = "netted_bar_research_tests.rs"]
mod tests;
