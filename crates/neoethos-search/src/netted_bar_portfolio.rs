//! Simultaneous expert sleeves on the existing canonical OHLC research owner.
//! A shared ledger is deliberate; this does not attest broker hedging, margin,
//! quotes, challenge rules, or promotion. V1 single-position receipts stay V1.
use super::{
    BarReplayLotGridV1, NettedBarDecisionTapeV1, PreparedNettedCanonicalBarResearchV1,
    remaining_fixed_margin_lots,
};
use crate::quality::{Trade, TradeExitReason};
use crate::scoring::RiskyGrowthGoal;
use anyhow::{Context, Result, ensure};
pub use neoethos_core::config::RiskyFixedNotionalLeverageV1 as JointBarFixedNotionalLeverageV1;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointBarAccountPolicyV1 {
    pub max_open_positions: usize,
    /// Aggregate INITIAL stop-price risk at admission, excluding costs/gaps.
    pub max_entry_stop_risk_fraction: f64,
    pub max_gross_notional_multiple: f64,
    /// The ordinary intraday policy cannot silently enable exceptional days.
    pub max_target_pips: f64,
}

impl JointBarAccountPolicyV1 {
    pub fn validate(&self) -> Result<()> {
        self.validate_fields()?;
        ensure!(
            self.max_target_pips <= 20.0,
            "invalid shared-account admission policy (normal target <=20 pips)"
        );
        Ok(())
    }

    fn validate_fields(&self) -> Result<()> {
        ensure!(
            (1..=16).contains(&self.max_open_positions)
                && self.max_entry_stop_risk_fraction.is_finite()
                && (0.0..=1.0).contains(&self.max_entry_stop_risk_fraction)
                && self.max_entry_stop_risk_fraction > 0.0
                && self.max_gross_notional_multiple.is_finite()
                && self.max_gross_notional_multiple > 0.0
                && self.max_target_pips.is_finite()
                && self.max_target_pips > 0.0,
            "invalid shared-account admission policy"
        );
        Ok(())
    }
}

/// The published policy keeps the same capped fields for ordinary replays.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct JointBarAppliedPolicyV1 {
    pub max_open_positions: usize,
    pub max_entry_stop_risk_fraction: f64,
    pub max_gross_notional_multiple: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_account_notional_per_lot: Option<f64>,
    pub max_target_pips: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_stop_risk_per_entry: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_stop_risk_band: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rounded_order_stop_risk_ceiling: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_to_buying_power: Option<bool>,
}

impl From<JointBarAccountPolicyV1> for JointBarAppliedPolicyV1 {
    fn from(policy: JointBarAccountPolicyV1) -> Self {
        Self {
            max_open_positions: policy.max_open_positions,
            max_entry_stop_risk_fraction: policy.max_entry_stop_risk_fraction,
            max_gross_notional_multiple: Some(policy.max_gross_notional_multiple),
            fixed_account_notional_per_lot: None,
            max_target_pips: policy.max_target_pips,
            exact_stop_risk_per_entry: None,
            confidence_stop_risk_band: None,
            rounded_order_stop_risk_ceiling: None,
            fit_to_buying_power: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct JointBarRiskyOrderSizingV1<'a> {
    pub leverage: Option<JointBarFixedNotionalLeverageV1>,
    pub confidence_risk_floor: Option<f64>,
    pub lot_grid: Option<BarReplayLotGridV1>,
    pub fit_to_buying_power: bool,
    pub max_open_positions: usize,
    pub price_stops: Option<JointBarPriceStopLevelsV1<'a>>,
    pub price_action_risk_ceiling: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct JointBarPriceStopLevelsV1<'a> {
    pub signal_row_levels: &'a [f64],
    pub distance_bounds_pips: [f64; 2],
}

pub struct JointBarLaneV1<'a> {
    pub expert_id: &'a str,
    /// ml_multipliers are the expert's allocated fraction, applied once.
    pub tape: NettedBarDecisionTapeV1<'a>,
}

#[derive(Debug, Serialize)]
pub struct JointBarPositionTermsV1 {
    pub direction: i8,
    pub lots: f64,
    pub stop_pips: f64,
    pub target_pips: f64,
}

#[derive(Debug, Serialize)]
pub struct JointBarClosedTradeV1 {
    pub expert_id: String,
    pub trade: Trade,
    #[serde(flatten)]
    pub terms: JointBarPositionTermsV1,
    pub account_equity_at_entry: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modeled_execution: Option<JointBarModeledExecutionV1>,
}

#[derive(Debug, Serialize)]
pub struct JointBarModeledExecutionV1 {
    pub entry_price: f64,
    pub exit_reference_before_half_spread: f64,
    pub entry_signal_confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_lots: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct JointBarMarkV1 {
    pub expert_id: String,
    #[serde(flatten)]
    pub terms: JointBarPositionTermsV1,
    pub entry_timestamp_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_equity_at_entry: Option<f64>,
    pub modeled_entry_price: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_lots: Option<f64>,
    pub gross_unrealized_account: f64,
    /// Hypothetical close mark including pending modeled fees and carry;
    /// this is not a settlement or a credit to realized balance.
    pub modeled_net_liquidation_pnl: f64,
}

#[derive(Debug, Serialize)]
pub struct JointBarEquityPointV1 {
    pub timestamp_ms: i64,
    pub realized_balance: f64,
    pub modeled_net_equity: f64,
    pub open_positions: usize,
}

#[derive(Debug, Serialize)]
pub struct JointBarAccountResultV1 {
    pub execution_basis: &'static str,
    pub promotion_eligible: bool,
    pub scope_identity_sha256: String,
    pub cost_contract_identity_sha256: String,
    pub account_currency: String,
    pub policy: JointBarAppliedPolicyV1,
    #[serde(flatten)]
    pub valuation: JointBarValuationV1,
    pub admission: JointBarAdmissionStatsV1,
    pub closed_trades: Vec<JointBarClosedTradeV1>,
    pub terminal_open: Vec<JointBarMarkV1>,
    pub equity_trace: Vec<JointBarEquityPointV1>,
    pub lot_grid: Option<BarReplayLotGridV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risky_goal: Option<JointBarRiskyGoalOutcomeV1>,
}

/// First-passage evidence for this account only. A cash crossing while other
/// sleeves remain open does not bank the target. No forced/free liquidation.
#[derive(Debug, Serialize)]
pub struct JointBarRiskyGoalOutcomeV1 {
    pub goal: RiskyGrowthGoal,
    pub deadline_timestamp_ms: i64,
    pub full_horizon_available: bool,
    pub first_realized_crossing_timestamp_ms: Option<i64>,
    pub banked_target_timestamp_ms: Option<i64>,
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct JointBarValuationV1 {
    pub initial_balance: f64,
    pub ending_realized_balance: f64,
    pub ending_modeled_net_equity: f64,
    #[serde(flatten)]
    pub drawdown: JointBarDrawdownV1,
    pub last_processed_timestamp_ms: i64,
}

#[derive(Debug, Default, Serialize)]
pub struct JointBarDrawdownV1 {
    pub max_close_equity_drawdown: f64,
    /// Sum of independent non-positive per-sleeve adverse OHLC marks relative
    /// to observed CLOSE peak. Conservative stress bound, not an intrabar path.
    pub max_adverse_bar_drawdown_bound_from_close_peak: f64,
    pub capital_bound_breached: bool,
}

#[derive(Debug, Default, Serialize)]
pub struct JointBarAdmissionStatsV1 {
    pub max_simultaneous_positions: usize,
    pub max_admitted_stop_risk_fraction: f64,
    pub max_admitted_gross_notional_multiple: f64,
    /// Accepted or refused entry requests, grouped by the actual cause.
    pub entry_outcomes: std::collections::BTreeMap<&'static str, usize>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub accepted_size_adjustments: std::collections::BTreeMap<&'static str, usize>,
}

impl JointBarAdmissionStatsV1 {
    pub fn entries(&self, outcome: &str) -> usize {
        self.entry_outcomes.get(outcome).copied().unwrap_or(0)
    }
    fn record(&mut self, outcome: &'static str) {
        *self.entry_outcomes.entry(outcome).or_default() += 1;
    }
}

struct Position {
    direction: i8,
    entry: usize,
    price: f64,
    lots: f64,
    stop: f64,
    target: f64,
    trail: f64,
    mfe: f64,
    mae: f64,
    entry_equity: f64,
    signal_confidence: f64,
    requested_lots: Option<f64>,
}

struct EntryRequest {
    lane: usize,
    requested_lots: Option<f64>,
    lots: f64,
    stop: f64,
    price: f64,
    fit_adjusted: bool,
}

impl Position {
    fn report_terms(&self) -> JointBarPositionTermsV1 {
        JointBarPositionTermsV1 {
            direction: self.direction,
            lots: self.lots,
            stop_pips: self.stop,
            target_pips: self.target,
        }
    }
}

impl PreparedNettedCanonicalBarResearchV1 {
    /// Explicit shared-account version: each expert may own one position;
    /// opposite sleeves remain separate modeled positions, without netting.
    /// All closes settle before a simultaneous, proportionally scaled batch
    /// of new entries. Sorted expert IDs make capacity ties deterministic.
    pub fn evaluate_joint(
        &self,
        lanes: &[JointBarLaneV1<'_>],
        policy: JointBarAccountPolicyV1,
        lot_grid: Option<BarReplayLotGridV1>,
        record_trace: bool,
    ) -> Result<JointBarAccountResultV1> {
        policy.validate()?;
        self.evaluate_joint_inner(lanes, policy, lot_grid, record_trace, None, None)
    }

    /// Risky variant of the SAME shared ledger: stop at the configured calendar
    /// deadline, bank only a flat realized target, and retain unclosed marks.
    pub fn evaluate_joint_risky(
        &self,
        lanes: &[JointBarLaneV1<'_>],
        policy: JointBarAccountPolicyV1,
        lot_grid: Option<BarReplayLotGridV1>,
        record_trace: bool,
        goal: RiskyGrowthGoal,
    ) -> Result<JointBarAccountResultV1> {
        policy.validate()?;
        self.evaluate_joint_inner(lanes, policy, lot_grid, record_trace, Some(goal), None)
    }

    /// Replay a frozen research cohort under its already recorded fixed target
    /// ceiling. The caller validates each gene and derives the ceiling from
    /// that cohort. The ordinary specialist policy retains its 20-pip ceiling.
    pub fn evaluate_joint_risky_archived_brackets(
        &self,
        lanes: &[JointBarLaneV1<'_>],
        policy: JointBarAccountPolicyV1,
        goal: RiskyGrowthGoal,
    ) -> Result<JointBarAccountResultV1> {
        policy.validate_fields()?;
        self.evaluate_joint_inner(lanes, policy, None, true, Some(goal), None)
    }

    /// Admit full-sized Risky orders against one shared 90% stop-risk budget.
    /// The default requests 30% per entry. The optional confidence band uses
    /// that same upper bound.
    /// Supplies whole modeled orders unless the research mode binds a broker
    /// quantity grid and explicitly allows rounding or downsizing.
    /// A missing goal keeps the same account through the full supplied period.
    pub fn evaluate_joint_risky_full_size(
        &self,
        lanes: &[JointBarLaneV1<'_>],
        max_target_pips: f64,
        record_trace: bool,
        goal: Option<RiskyGrowthGoal>,
        order_sizing: JointBarRiskyOrderSizingV1<'_>,
    ) -> Result<JointBarAccountResultV1> {
        const EXACT_STOP_RISK: f64 = 0.30;
        ensure!(
            self.settings.risk_per_trade_min == EXACT_STOP_RISK
                && self.settings.risk_per_trade_max == EXACT_STOP_RISK,
            "exact Risky replay requires the sealed fixed 30% risk band"
        );
        ensure!(
            max_target_pips.is_finite() && max_target_pips > 0.0,
            "exact Risky replay requires a finite positive target ceiling"
        );
        let leverage = order_sizing.leverage;
        let lot_grid = order_sizing.lot_grid;
        ensure!(
            (1..=3).contains(&order_sizing.max_open_positions),
            "Risky replay requires one to three position slots"
        );
        ensure!(
            lot_grid.is_none()
                || (leverage.is_some() && order_sizing.confidence_risk_floor.is_none()),
            "rounded Risky replay requires fixed leverage and a fixed stop-risk ceiling"
        );
        ensure!(
            !order_sizing.fit_to_buying_power || lot_grid.is_some(),
            "order resizing requires an explicit lot grid"
        );
        if let Some(price_stops) = order_sizing.price_stops {
            let [min, max] = price_stops.distance_bounds_pips;
            ensure!(
                lot_grid.is_some()
                    && lanes.len() == 1
                    && price_stops.signal_row_levels.len() == self.close.len()
                    && min.is_finite()
                    && max.is_finite()
                    && min > 0.0
                    && max >= min,
                "structural stops require a single aligned lane and valid pip bounds"
            );
        }
        let rounded_risk_ceiling = order_sizing
            .price_action_risk_ceiling
            .unwrap_or(EXACT_STOP_RISK);
        ensure!(
            order_sizing.price_action_risk_ceiling.is_none()
                || (order_sizing.price_stops.is_some()
                    && rounded_risk_ceiling > 0.0
                    && rounded_risk_ceiling <= EXACT_STOP_RISK),
            "a lower risk ceiling requires an explicit price-action replay"
        );
        let fixed_account_notional_per_lot = leverage.map(|limit| limit.account_notional_per_lot);
        ensure!(
            fixed_account_notional_per_lot.is_none_or(|value| value.is_finite() && value > 0.0),
            "fixed leverage requires finite positive account notional per lot"
        );
        let confidence_stop_risk_band = order_sizing
            .confidence_risk_floor
            .map(|floor| [floor, EXACT_STOP_RISK]);
        if let Some([minimum, maximum]) = confidence_stop_risk_band {
            ensure!(
                leverage.is_some(),
                "adaptive Risky replay requires fixed leverage"
            );
            neoethos_core::domain::risky_mode::RiskyLimitsPolicy::PerTradeOnlyV1.validate_band(
                minimum,
                maximum,
                self.settings.high_quality_confidence,
            )?;
        }
        let policy = JointBarAppliedPolicyV1 {
            max_open_positions: order_sizing.max_open_positions,
            max_entry_stop_risk_fraction: 0.90,
            max_gross_notional_multiple: leverage.map(|limit| f64::from(limit.maximum.get())),
            fixed_account_notional_per_lot,
            max_target_pips,
            exact_stop_risk_per_entry: (confidence_stop_risk_band.is_none() && lot_grid.is_none())
                .then_some(EXACT_STOP_RISK),
            confidence_stop_risk_band,
            rounded_order_stop_risk_ceiling: lot_grid.map(|_| rounded_risk_ceiling),
            fit_to_buying_power: lot_grid.map(|_| order_sizing.fit_to_buying_power),
        };
        self.evaluate_joint_inner(
            lanes,
            policy,
            lot_grid,
            record_trace,
            goal,
            order_sizing.price_stops,
        )
    }

    fn evaluate_joint_inner(
        &self,
        lanes: &[JointBarLaneV1<'_>],
        policy: impl Into<JointBarAppliedPolicyV1>,
        lot_grid: Option<BarReplayLotGridV1>,
        record_trace: bool,
        goal: Option<RiskyGrowthGoal>,
        price_stops: Option<JointBarPriceStopLevelsV1<'_>>,
    ) -> Result<JointBarAccountResultV1> {
        let policy = policy.into();
        let fixed_leverage = policy
            .fixed_account_notional_per_lot
            .and(policy.max_gross_notional_multiple);
        let rounded_risky = policy.rounded_order_stop_risk_ceiling.is_some();
        let full_size_risky = policy.exact_stop_risk_per_entry.is_some()
            || policy.confidence_stop_risk_band.is_some()
            || rounded_risky;
        ensure!(
            full_size_risky || lanes.len() <= 256,
            "bounded research supports at most 256 archived expert lanes"
        );
        let settings = &self.settings;
        if let Some(bound) = settings
            .risky_account_experiment
            .as_ref()
            .and_then(|account| account.fixed_leverage)
        {
            ensure!(
                rounded_risky
                    && policy.fit_to_buying_power == Some(true)
                    && fixed_leverage == Some(f64::from(bound.maximum.get()))
                    && policy.fixed_account_notional_per_lot
                        == Some(bound.account_notional_per_lot),
                "shared replay must use the fixed margin and order sizing sealed during search"
            );
        }
        ensure!(
            settings.risk_based_sizing
                && settings.pip_value.is_finite()
                && settings.pip_value > 0.0
                && settings.pip_value_per_lot.is_finite()
                && settings.pip_value_per_lot > 0.0,
            "shared account requires checked risk sizing and account pip economics"
        );
        let grid = settings.resolved_lot_grid(lot_grid, rounded_risky)?;
        ensure!(
            !full_size_risky || grid.is_some() == rounded_risky,
            "Risky lot quantization must match the selected order policy"
        );
        let mut order: Vec<_> = (0..lanes.len()).collect();
        order.sort_by_key(|&i| lanes[i].expert_id);
        for (rank, &i) in order.iter().enumerate() {
            ensure!(
                !lanes[i].expert_id.is_empty()
                    && (rank == 0 || lanes[order[rank - 1]].expert_id != lanes[i].expert_id),
                "expert identities must be nonempty and unique"
            );
            self.validate_tape(lanes[i].tape)?;
            ensure!(
                !full_size_risky
                    || lanes[i]
                        .tape
                        .ml_multipliers
                        .iter()
                        .all(|&multiplier| multiplier == 0.0 || multiplier == 1.0),
                "full-sized Risky lanes require binary entry multipliers"
            );
        }
        // Validate economics even for a cash portfolio with no lanes.
        super::super::validate_sizing_confidences(
            self.close.len(),
            &vec![0.; self.close.len()],
            settings,
        )?;
        ensure!(
            [settings.spread_pips, settings.commission_per_trade]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
                && [
                    settings.swap_long_pips_per_day,
                    settings.swap_short_pips_per_day
                ]
                .iter()
                .all(|v| v.is_finite()),
            "invalid shared-account modeled costs"
        );
        let initial = settings.initial_equity();
        let risky_goal = goal
            .map(|goal| -> Result<JointBarRiskyGoalOutcomeV1> {
                goal.validate().map_err(anyhow::Error::msg)?;
                ensure!(
                    goal.start_balance == initial,
                    "Risky shared account must replay at its configured starting capital"
                );
                if let Some(experiment) = &settings.risky_account_experiment {
                    ensure!(
                        experiment.target_balance == goal.target_balance,
                        "Risky shared account target disagrees with sealed accounting policy"
                    );
                }
                let duration = goal.horizon_days * 86_400_000.0;
                ensure!(
                    duration.is_finite() && duration >= 1.0 && duration < i64::MAX as f64,
                    "Risky calendar horizon overflows milliseconds"
                );
                let deadline = self.timestamps[0]
                    .checked_add(duration as i64)
                    .context("Risky deadline overflow")?;
                Ok(JointBarRiskyGoalOutcomeV1 {
                    goal,
                    deadline_timestamp_ms: deadline,
                    full_horizon_available: *self.timestamps.last().unwrap() >= deadline,
                    first_realized_crossing_timestamp_ms: None,
                    banked_target_timestamp_ms: None,
                    status: "incomplete_history",
                })
            })
            .transpose()?;
        let end_row = risky_goal.as_ref().map_or(self.close.len(), |g| {
            self.timestamps
                .partition_point(|&t| t <= g.deadline_timestamp_ms)
        });
        let mut out = JointBarAccountResultV1 {
            execution_basis: if rounded_risky {
                "canonical_cpu_joint_ohlc_risky_volume_v1; prior_row_signal_next_close_fill; broker_symbol_lot_grid; scalar_costs; fixed_gross_notional_admission; research_only"
            } else if fixed_leverage.is_some() {
                "canonical_cpu_joint_ohlc_risky_leverage_v1; prior_row_signal_next_close_fill; fractional_lots; scalar_costs; shared_realized_cash; pending_cost_net_marks; fixed_gross_notional_admission; research_only"
            } else if policy.exact_stop_risk_per_entry.is_some() {
                "canonical_cpu_joint_ohlc_exact_risky_v1; prior_row_signal_next_close_fill; fractional_lots; scalar_costs; shared_realized_cash; pending_cost_net_marks; not_broker_margin_or_challenge_simulation"
            } else {
                "canonical_cpu_joint_ohlc_v1; prior_row_signal_next_close_fill; independent_hedged_sleeves; scalar_costs; shared_realized_cash; pending_cost_net_marks; not_broker_margin_or_challenge_simulation"
            },
            promotion_eligible: false,
            scope_identity_sha256: self.scope_identity_sha256.clone(),
            cost_contract_identity_sha256: self.cost_contract_identity_sha256.clone(),
            account_currency: self.account_currency.clone(),
            policy,
            valuation: JointBarValuationV1 {
                initial_balance: initial,
                ending_realized_balance: initial,
                ending_modeled_net_equity: initial,
                drawdown: JointBarDrawdownV1::default(),
                last_processed_timestamp_ms: self.timestamps[0],
            },
            admission: JointBarAdmissionStatsV1::default(),
            closed_trades: Vec::new(),
            terminal_open: Vec::new(),
            equity_trace: Vec::new(),
            lot_grid: grid,
            risky_goal,
        };
        let mut positions: Vec<Option<Position>> = (0..lanes.len()).map(|_| None).collect();
        let mut balance = initial;
        let mut peak = initial;
        let mut day = self.days[0];
        let mut day_entries = 0usize;
        let pip = settings.pip_value;
        let ppv = settings.pip_value_per_lot;
        for row in 1..end_row {
            let timestamp = self.timestamps[row];
            if self.days[row] != day {
                day = self.days[row];
                day_entries = 0;
            }
            let spread = settings
                .session_spread_profile
                .map_or(settings.spread_pips, |p| p.spread_pips_at(timestamp));
            ensure!(
                spread.is_finite() && spread >= 0.0,
                "invalid session spread"
            );
            let exit_cost = spread * 0.5 * ppv;
            let net = |p: &Position, price| {
                super::super::closed_position_net_pnl(
                    p.direction,
                    p.price,
                    price,
                    p.lots,
                    exit_cost,
                    self.timestamps[p.entry],
                    timestamp,
                    settings,
                )
            };
            let (session_close, session_block) =
                super::super::weekend_session_policy(timestamp, settings.kill_zones_enabled);
            let gap = settings.gap_threshold_ms > 0
                && timestamp - self.timestamps[row - 1] >= settings.gap_threshold_ms;
            let mut just_closed = vec![false; lanes.len()];
            let mut adverse_bound = balance;
            for &i in &order {
                let Some(p) = positions[i].as_mut() else {
                    continue;
                };
                let exit = if gap {
                    Some((self.close[row], TradeExitReason::DataGap))
                } else {
                    super::super::protective_exit_price(
                        p.direction,
                        p.price,
                        p.stop,
                        p.target,
                        &mut p.trail,
                        self.high[row],
                        self.low[row],
                        self.close[row],
                        row - p.entry,
                        pip,
                        settings,
                        true,
                    )
                    .or_else(|| {
                        session_close.then_some((self.close[row], TradeExitReason::SessionClose))
                    })
                };
                let (lo, hi, _) = super::super::observed_position_bar(
                    p.direction,
                    exit,
                    self.low[row],
                    self.high[row],
                );
                let (favorable, adverse, worst) = if p.direction == 1 {
                    (hi - p.price, p.price - lo, lo)
                } else {
                    (p.price - lo, hi - p.price, hi)
                };
                p.mfe = p.mfe.max(favorable / pip * ppv * p.lots);
                p.mae = p.mae.max(adverse / pip * ppv * p.lots);
                adverse_bound += net(p, worst).min(0.0);
                if let Some((price, reason)) = exit {
                    let pnl = net(p, price);
                    ensure!(pnl.is_finite(), "shared settlement overflow");
                    let mut trade = Vec::with_capacity(1);
                    super::super::record_closed_cpu_trade(
                        &mut trade,
                        self.timestamps[p.entry],
                        timestamp,
                        pnl,
                        initial,
                        p.stop * ppv * p.lots,
                        p.mfe,
                        p.mae,
                        reason,
                    );
                    out.closed_trades.push(JointBarClosedTradeV1 {
                        expert_id: lanes[i].expert_id.to_owned(),
                        terms: p.report_terms(),
                        account_equity_at_entry: p.entry_equity,
                        modeled_execution: fixed_leverage.map(|_| JointBarModeledExecutionV1 {
                            entry_price: p.price,
                            exit_reference_before_half_spread: price,
                            entry_signal_confidence: p.signal_confidence,
                            requested_lots: p.requested_lots,
                        }),
                        trade: trade.pop().expect("recorded closed trade"),
                    });
                    balance += pnl;
                    positions[i] = None;
                    just_closed[i] = true;
                }
            }
            ensure!(
                adverse_bound.is_finite() && balance.is_finite(),
                "shared account overflow"
            );
            out.valuation
                .drawdown
                .max_adverse_bar_drawdown_bound_from_close_peak = out
                .valuation
                .drawdown
                .max_adverse_bar_drawdown_bound_from_close_peak
                .max(((peak - adverse_bound) / peak).max(0.0));
            if adverse_bound <= 0.0 || balance <= 0.0 {
                out.valuation.drawdown.capital_bound_breached = true;
            }
            let mut marked = balance;
            let mut reserved_risk = 0.0;
            let mut reserved_lots = 0.0;
            for p in positions.iter().flatten() {
                marked += net(p, self.close[row]);
                reserved_risk += p.stop * ppv * p.lots;
                reserved_lots += p.lots;
            }
            ensure!(marked.is_finite(), "shared equity mark overflow");
            if marked <= 0.0 {
                out.valuation.drawdown.capital_bound_breached = true;
            }
            let mut target_banked = false;
            let mut entry_before_deadline = true;
            if let Some(goal) = &mut out.risky_goal {
                if balance >= goal.goal.target_balance {
                    goal.first_realized_crossing_timestamp_ms
                        .get_or_insert(timestamp);
                    if positions.iter().all(Option::is_none)
                        && !out.valuation.drawdown.capital_bound_breached
                    {
                        goal.banked_target_timestamp_ms = Some(timestamp);
                        target_banked = true;
                    }
                }
                // Pause additional exposure while a realized crossing waits for
                // existing sleeves to settle. Those sleeves keep their exits.
                entry_before_deadline =
                    timestamp < goal.deadline_timestamp_ms && balance < goal.goal.target_balance;
            }
            // All batch members use exactly this one equity denominator.
            let notional_per_lot = policy
                .fixed_account_notional_per_lot
                .unwrap_or_else(|| self.close[row] / pip * ppv);
            let mut requests = Vec::new();
            let mut requested_lots = reserved_lots;
            let mut requested_stop_risk = reserved_risk;
            // Three simultaneous 30% requests fit the 90% budget to rounding precision.
            let stop_risk_limit =
                marked * (policy.max_entry_stop_risk_fraction + 8.0 * f64::EPSILON);
            let open_count = positions.iter().flatten().count();
            let mut slots = policy.max_open_positions.saturating_sub(open_count);
            if settings.max_trades_per_day > 0 {
                slots = slots.min(settings.max_trades_per_day.saturating_sub(day_entries));
            }
            if !session_block
                && !out.valuation.drawdown.capital_bound_breached
                && entry_before_deadline
                && !target_banked
            {
                for &i in &order {
                    let tape = lanes[i].tape;
                    let signal = row - 1;
                    if positions[i].is_some()
                        || just_closed[i]
                        || tape.signals[signal] == 0
                        || tape.ml_multipliers[signal] == 0.0
                    {
                        continue;
                    }
                    if tape.tp_pips[signal] > policy.max_target_pips {
                        out.admission.record("oversized_target");
                        continue;
                    }
                    if requests.len() >= slots {
                        out.admission.record("capacity");
                        continue;
                    }
                    let direction = f64::from(tape.signals[signal]);
                    let price = self.close[row] + direction * spread * 0.5 * pip;
                    let mut stop = tape.sl_pips[signal];
                    if let Some(levels) = price_stops {
                        let level = levels.signal_row_levels[signal];
                        let [min, max] = levels.distance_bounds_pips;
                        stop = direction * (price - level) / pip;
                        if !level.is_finite() || level <= 0.0 || !(min..=max).contains(&stop) {
                            out.admission.record("price_stop_distance");
                            continue;
                        }
                    }
                    let entry_risk_fraction = policy
                        .confidence_stop_risk_band
                        .map(|[min, max]| {
                            neoethos_core::domain::risky_mode::confidence_scaled_risk_fraction(
                                min,
                                max,
                                settings.high_quality_confidence,
                                tape.confidences[signal],
                            )
                        })
                        .or(policy.exact_stop_risk_per_entry)
                        .or(policy.rounded_order_stop_risk_ceiling);
                    let mut lots = if let Some(risk_fraction) = entry_risk_fraction {
                        let lots = (marked * risk_fraction) / (stop * ppv);
                        ensure!(
                            lots.is_finite() && lots > 0.0,
                            "Risky position size cannot be represented"
                        );
                        lots
                    } else {
                        super::super::risk_based_pos_lots(
                            tape.confidences[signal],
                            marked,
                            stop,
                            settings,
                        ) * tape.ml_multipliers[signal]
                    };
                    if lots > 0.0 {
                        let unrounded_request = lots;
                        let mut fit_adjusted = false;
                        if let Some(limit) = fixed_leverage {
                            if let Some(grid) = grid {
                                if policy.fit_to_buying_power == Some(true) {
                                    let risk_capacity = (stop_risk_limit - requested_stop_risk)
                                        .max(0.0)
                                        / (stop * ppv);
                                    let margin_capacity = remaining_fixed_margin_lots(
                                        marked,
                                        limit,
                                        notional_per_lot,
                                        requested_lots,
                                    );
                                    let resized = lots.min(risk_capacity).min(margin_capacity);
                                    fit_adjusted = resized < lots;
                                    lots = resized;
                                }
                                let Some(normalized) = grid.normalize_down(lots) else {
                                    let refusal = if fit_adjusted
                                        && grid.normalize_down(unrounded_request).is_some()
                                    {
                                        "buying_power_below_min_lots"
                                    } else {
                                        "below_min_lots"
                                    };
                                    out.admission.record(refusal);
                                    continue;
                                };
                                lots = normalized;
                            }
                            // A refused order leaves the slot available for the next request.
                            let order_risk = stop * ppv * lots;
                            if requested_stop_risk + order_risk > stop_risk_limit {
                                out.admission.record("aggregate_stop_risk_budget");
                                continue;
                            }
                            if (requested_lots + lots) * notional_per_lot > marked * limit {
                                out.admission.record("gross_notional_limit");
                                continue;
                            }
                            requested_stop_risk += order_risk;
                            requested_lots += lots;
                        }
                        requests.push(EntryRequest {
                            lane: i,
                            requested_lots: rounded_risky.then_some(unrounded_request),
                            lots,
                            stop,
                            price,
                            fit_adjusted,
                        });
                    }
                }
            }
            let scale = policy
                .max_gross_notional_multiple
                .filter(|_| fixed_leverage.is_none())
                .map_or(1.0, |notional_cap| {
                    let requested_risk: f64 = requests
                        .iter()
                        .map(|request| request.lots * request.stop * ppv)
                        .sum();
                    let requested_notional = requests
                        .iter()
                        .map(|request| request.lots * notional_per_lot)
                        .sum();
                    let ratio = |available: f64, requested: f64| {
                        if requested > 0.0 {
                            (available.max(0.0) / requested).min(1.0)
                        } else {
                            1.0
                        }
                    };
                    ratio(
                        marked * policy.max_entry_stop_risk_fraction - reserved_risk,
                        requested_risk,
                    )
                    .min(ratio(
                        marked * notional_cap - reserved_lots * notional_per_lot,
                        requested_notional,
                    ))
                });
            for request in requests {
                let i = request.lane;
                let mut lots = request.lots * scale;
                if lots <= 0.0 {
                    out.admission.record("capacity");
                    continue;
                }
                if let Some(grid) = grid.filter(|_| fixed_leverage.is_none()) {
                    let Some(normalized) = grid.normalize_down(lots) else {
                        out.admission.record("below_min_lots");
                        continue;
                    };
                    lots = normalized;
                }
                let tape = lanes[i].tape;
                let stop = request.stop;
                let order_risk = stop * ppv * lots;
                if full_size_risky
                    && fixed_leverage.is_none()
                    && reserved_risk + order_risk > stop_risk_limit
                {
                    out.admission.record("aggregate_stop_risk_budget");
                    continue;
                }
                let p = Position {
                    direction: tape.signals[row - 1],
                    entry: row,
                    price: request.price,
                    lots,
                    stop,
                    target: tape.tp_pips[row - 1],
                    trail: 0.0,
                    mfe: 0.0,
                    mae: 0.0,
                    entry_equity: marked,
                    signal_confidence: tape.confidences[row - 1],
                    requested_lots: request.requested_lots,
                };
                reserved_risk += order_risk;
                reserved_lots += lots;
                positions[i] = Some(p);
                out.admission.record("accepted");
                if request
                    .requested_lots
                    .is_some_and(|requested| lots < requested)
                {
                    let reason = if request.fit_adjusted {
                        "buying_power"
                    } else {
                        "lot_grid"
                    };
                    *out.admission
                        .accepted_size_adjustments
                        .entry(reason)
                        .or_default() += 1;
                }
                day_entries += 1;
            }
            if reserved_lots > 0.0
                && marked > 0.0
                && positions.iter().flatten().any(|p| p.entry == row)
            {
                out.admission.max_admitted_stop_risk_fraction = out
                    .admission
                    .max_admitted_stop_risk_fraction
                    .max(reserved_risk / marked);
                out.admission.max_admitted_gross_notional_multiple = out
                    .admission
                    .max_admitted_gross_notional_multiple
                    .max(reserved_lots * notional_per_lot / marked);
            }
            marked = balance
                + positions
                    .iter()
                    .flatten()
                    .map(|p| net(p, self.close[row]))
                    .sum::<f64>();
            ensure!(
                marked.is_finite() && notional_per_lot.is_finite(),
                "shared position valuation overflow"
            );
            if marked <= 0.0 {
                out.valuation.drawdown.capital_bound_breached = true;
            }
            peak = peak.max(marked);
            out.valuation.drawdown.max_close_equity_drawdown = out
                .valuation
                .drawdown
                .max_close_equity_drawdown
                .max(((peak - marked) / peak).max(0.0));
            out.admission.max_simultaneous_positions = out
                .admission
                .max_simultaneous_positions
                .max(positions.iter().flatten().count());
            out.valuation.ending_realized_balance = balance;
            out.valuation.ending_modeled_net_equity = marked;
            out.valuation.last_processed_timestamp_ms = timestamp;
            if record_trace {
                out.equity_trace.push(JointBarEquityPointV1 {
                    timestamp_ms: timestamp,
                    realized_balance: balance,
                    modeled_net_equity: marked,
                    open_positions: positions.iter().flatten().count(),
                });
            }
            if out.valuation.drawdown.capital_bound_breached || target_banked || row + 1 == end_row
            {
                for &i in &order {
                    if let Some(p) = &positions[i] {
                        out.terminal_open.push(JointBarMarkV1 {
                            expert_id: lanes[i].expert_id.to_owned(),
                            terms: p.report_terms(),
                            entry_timestamp_ms: self.timestamps[p.entry],
                            account_equity_at_entry: full_size_risky.then_some(p.entry_equity),
                            modeled_entry_price: p.price,
                            requested_lots: p.requested_lots,
                            gross_unrealized_account: f64::from(p.direction)
                                * (self.close[row] - p.price)
                                / pip
                                * ppv
                                * p.lots,
                            modeled_net_liquidation_pnl: net(p, self.close[row]),
                        });
                    }
                }
                break;
            }
        }
        if let Some(goal) = &mut out.risky_goal {
            goal.status = if out.valuation.drawdown.capital_bound_breached {
                "capital_bound_breached"
            } else if goal.banked_target_timestamp_ms.is_some() {
                "target_banked"
            } else if goal.full_horizon_available {
                "deadline_elapsed_without_banked_target"
            } else {
                "incomplete_history"
            };
        }
        Ok(out)
    }
}
