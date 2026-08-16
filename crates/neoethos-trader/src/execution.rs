//! Phase-1 `MockExecutionAdapter` — simulated fills, ZERO broker calls.
//!
//! Fills every `Open`/`Close` at the `mark_price` the engine observed at
//! decision time, ADJUSTED BY THE CONFIGURED COST MODEL, and hands back
//! synthetic position ids. It records a full fill log so the replay harness +
//! tests can assert what happened. This is the offline-dry-run adapter behind
//! the `ExecutionAdapter` trait; the real cTrader `broker_api` adapter
//! (Phase 5) implements the SAME trait — demo vs live is the connected account,
//! not separate code.
//!
//! ## Costs (audit #227, fixed 2026-08-09)
//!
//! Until 2026-08-09 this adapter filled at the mark with **zero spread, zero
//! commission and zero slippage**, so a replay reported the gross path of a
//! strategy that pays none of the three. That is the single largest reason a
//! replay number could not be compared with a live number.
//!
//! [`ReplayCostModel`] now carries the three charges. It defaults to ZERO —
//! deliberately, because inventing a spread would be a different lie — and the
//! zero case is reported as a fidelity warning by the replay harness rather
//! than passing silently. Front-ends that know the operator's real costs (they
//! hold `Settings`) build the model with [`ReplayCostModel::from_pips`] and
//! pass it through `EngineConfig::costs`.

use crate::contracts::{ExecReport, ExecStatus, ExecutionAdapter, TradeIntent};

/// Per-trade transaction costs for the offline replay, in PRICE units (already
/// multiplied by the symbol's pip size) except `commission_per_lot`, which is
/// account currency.
///
/// `half_spread_price` is charged on BOTH legs (buy at ask, sell at bid), which
/// is how a full round-turn spread gets paid exactly once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReplayCostModel {
    /// Half the quoted spread, in price units. Charged adversely on entry and
    /// on exit.
    pub half_spread_price: f64,
    /// Additional adverse price movement per fill, in price units.
    pub slippage_price: f64,
    /// Commission in ACCOUNT currency per lot, charged on entry and on exit.
    pub commission_per_lot: f64,
}

impl Default for ReplayCostModel {
    fn default() -> Self {
        Self::zero()
    }
}

impl ReplayCostModel {
    /// The historical behaviour: charge nothing. Kept as an explicit,
    /// NAMED state so a run that charges nothing has to say so.
    pub const fn zero() -> Self {
        Self {
            half_spread_price: 0.0,
            slippage_price: 0.0,
            commission_per_lot: 0.0,
        }
    }

    /// Build from pip-denominated costs. `spread_pips` is the FULL quoted
    /// spread — it is halved here so entry and exit each pay half.
    ///
    /// Non-finite or negative inputs are clamped to zero and logged; a cost
    /// model must never make a trade look BETTER than the mark.
    pub fn from_pips(
        spread_pips: f64,
        slippage_pips: f64,
        commission_per_lot: f64,
        pip_size: f64,
    ) -> Self {
        fn sane(name: &'static str, v: f64) -> f64 {
            if v.is_finite() && v >= 0.0 {
                v
            } else {
                tracing::warn!(
                    target: "neoethos_trader::execution",
                    field = name,
                    value = v,
                    "replay cost input is negative or non-finite — REFUSED, charging 0 for \
                     this component. The replay will look better than reality by that much."
                );
                0.0
            }
        }
        let pip = sane("pip_size", pip_size);
        Self {
            half_spread_price: sane("spread_pips", spread_pips) * 0.5 * pip,
            slippage_price: sane("slippage_pips", slippage_pips) * pip,
            commission_per_lot: sane("commission_per_lot", commission_per_lot),
        }
    }

    /// True when this model charges literally nothing.
    pub fn is_zero(&self) -> bool {
        self.half_spread_price == 0.0
            && self.slippage_price == 0.0
            && self.commission_per_lot == 0.0
    }

    /// Adverse price adjustment applied to one fill.
    fn adverse(&self) -> f64 {
        self.half_spread_price + self.slippage_price
    }
}

/// Simulates execution in-memory. Optionally rejects a fraction of intents to
/// let tests exercise the rejection path (default: fill everything).
#[derive(Debug, Default)]
pub(crate) struct MockExecutionAdapter {
    next_id: u64,
    costs: ReplayCostModel,
    commission_charged: f64,
    /// Direction + volume of each open position id, so a Close can be filled on
    /// the correct side of the spread and charged the right commission.
    open_legs: Vec<(String, f64, f64)>,
}

impl MockExecutionAdapter {
    /// Adapter that charges `costs` on every fill.
    pub fn with_costs(costs: ReplayCostModel) -> Self {
        Self {
            costs,
            ..Self::default()
        }
    }

    fn alloc_position_id(&mut self) -> String {
        self.next_id += 1;
        format!("mock-pos-{}", self.next_id)
    }
}

impl ExecutionAdapter for MockExecutionAdapter {
    fn execute(&mut self, intent: &TradeIntent, mark_price: f64) -> anyhow::Result<ExecReport> {
        let adverse = self.costs.adverse();
        let mut commission = 0.0;
        let report = match intent {
            TradeIntent::Open { dir, volume, .. } => {
                // Entry pays the adverse side: a buy fills above the mark, a
                // sell below it.
                let fill = mark_price + dir.sign() * adverse;
                commission = self.costs.commission_per_lot * volume.abs();
                let id = self.alloc_position_id();
                self.open_legs.push((id.clone(), dir.sign(), *volume));
                ExecReport {
                    status: ExecStatus::Filled,
                    fill_price: Some(fill),
                    position_id: Some(id),
                    detail: "mock open filled".to_string(),
                }
            }
            TradeIntent::Close {
                position_id,
                volume,
                ..
            } => {
                // Exit pays the adverse side of the OPPOSITE leg. If we never
                // saw the open (defensive — a manifest-driven position), charge
                // nothing rather than guess a side, and say so.
                let leg = self.open_legs.iter().find(|(id, _, _)| id == position_id);
                let (sign, open_vol) = match leg {
                    Some((_, s, v)) => (*s, *v),
                    None => {
                        tracing::warn!(
                            target: "neoethos_trader::execution",
                            position_id = %position_id,
                            "closing a position this adapter never opened — costs NOT charged \
                             on this leg (no known side). Counted, not dropped."
                        );
                        (0.0, 0.0)
                    }
                };
                let fill = mark_price - sign * adverse;
                let closed_vol = volume.unwrap_or(open_vol).abs();
                commission = self.costs.commission_per_lot * closed_vol;
                if volume.is_none() {
                    self.open_legs.retain(|(id, _, _)| id != position_id);
                }
                ExecReport {
                    status: ExecStatus::Filled,
                    fill_price: Some(fill),
                    position_id: Some(position_id.clone()),
                    detail: "mock close filled".to_string(),
                }
            }
            TradeIntent::Amend { position_id, .. } => ExecReport {
                status: ExecStatus::Filled,
                fill_price: None,
                position_id: Some(position_id.clone()),
                detail: "mock amend applied".to_string(),
            },
            TradeIntent::Cancel { order_id } => ExecReport {
                status: ExecStatus::Filled,
                fill_price: None,
                position_id: None,
                detail: format!("mock cancel {order_id}"),
            },
        };
        self.commission_charged += commission;
        Ok(report)
    }

    fn charged_costs(&self) -> f64 {
        self.commission_charged
    }
}
