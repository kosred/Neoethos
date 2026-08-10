//! In-memory open-position tracking + bar-driven SL/TP management.
//!
//! Phase 1 is a self-contained simulator: positions are opened/closed/amended
//! from executed `TradeIntent`s and marked against replayed bars. P&L here is a
//! simple `points × volume` proxy — the authoritative strategy P&L stays in the
//! GA backtest; this only needs to prove the loop mechanics + exposure tracking.

use serde::{Deserialize, Serialize};

use crate::contracts::{CloseReason, Direction, ExecReport, ExecStatus, LiveBar, SignalSource, TradeIntent};

/// One open position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub id: String,
    pub symbol: String,
    pub dir: Direction,
    pub volume: f64,
    pub entry_price: f64,
    pub sl: Option<f64>,
    pub tp: Option<f64>,
    pub source: SignalSource,
    /// Engine bar index at which this position was opened. Drives the
    /// `max_hold_bars` time stop — the GA evaluator's third exit, which the
    /// replay did not have at all (audit #228). `serde(default)` so manifests
    /// written before 2026-08-09 still load.
    #[serde(default)]
    pub opened_at_bar: u64,
}

impl Position {
    /// Unrealised P&L as `points × volume` (sign-aware). Pip value + contract
    /// size wire in with the real cost model later.
    pub fn unrealized(&self, price: f64) -> f64 {
        (price - self.entry_price) * self.dir.sign() * self.volume
    }

    /// Does `bar` cross this position's SL or TP? Returns the close reason + the
    /// price to close at (the level itself — a conservative fill assumption).
    /// SL is checked before TP so a bar that straddles both is treated as the
    /// adverse outcome (no intrabar-path optimism).
    pub fn exit_hit(&self, bar: &LiveBar) -> Option<(CloseReason, f64)> {
        match self.dir {
            Direction::Long => {
                if let Some(sl) = self.sl {
                    if bar.l <= sl {
                        return Some((CloseReason::StopLoss, sl));
                    }
                }
                if let Some(tp) = self.tp {
                    if bar.h >= tp {
                        return Some((CloseReason::TakeProfit, tp));
                    }
                }
            }
            Direction::Short => {
                if let Some(sl) = self.sl {
                    if bar.h >= sl {
                        return Some((CloseReason::StopLoss, sl));
                    }
                }
                if let Some(tp) = self.tp {
                    if bar.l <= tp {
                        return Some((CloseReason::TakeProfit, tp));
                    }
                }
            }
            Direction::Flat => {}
        }
        None
    }
}

/// Tracks open positions, applies executed intents, and emits SL/TP-driven
/// close intents per bar.
#[derive(Debug, Default)]
pub struct PositionManager {
    open: Vec<Position>,
    next_id: u64,
    realized_pnl: f64,
    opened_count: usize,
    closed_count: usize,
    /// Bar index of the bar currently being processed. Stamped onto every
    /// position opened on it.
    current_bar: u64,
    /// The GA evaluator's time stop. `None` ⇒ no time stop (the Phase-1
    /// behaviour); the real-gene replay paths set it from
    /// `EvaluationConfig::max_hold_bars` so a replayed trade cannot outlive
    /// every trade the backtest scored (audit #228).
    max_hold_bars: Option<u64>,
    /// Count of time-stop closes emitted, for the run report.
    max_hold_exits: usize,
}

impl PositionManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tell the manager which bar index is being processed. The engine calls
    /// this once per bar, BEFORE managing exits and before applying fills.
    pub fn set_bar(&mut self, bar_index: u64) {
        self.current_bar = bar_index;
    }

    /// Arm the `max_hold_bars` time stop. `None` disables it.
    pub fn set_max_hold_bars(&mut self, bars: Option<u64>) {
        self.max_hold_bars = bars;
    }

    /// How many positions were closed by the time stop.
    pub fn max_hold_exits(&self) -> usize {
        self.max_hold_exits
    }

    pub fn open_positions(&self) -> &[Position] {
        &self.open
    }

    /// Snapshot (clones) of the positions for one symbol — handed to the
    /// DecisionEngine so it can reason without holding a borrow on the manager.
    pub fn positions_for(&self, symbol: &str) -> Vec<Position> {
        self.open.iter().filter(|p| p.symbol == symbol).cloned().collect()
    }

    pub fn has_open(&self, symbol: &str) -> bool {
        self.open.iter().any(|p| p.symbol == symbol)
    }

    pub fn open_count(&self) -> usize {
        self.open.len()
    }

    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    pub fn opened_count(&self) -> usize {
        self.opened_count
    }

    pub fn closed_count(&self) -> usize {
        self.closed_count
    }

    /// Total unrealised P&L across all open positions marked at `mark`
    /// (a per-symbol price lookup; symbols with no mark contribute 0).
    pub fn unrealized_total(&self, mark: impl Fn(&str) -> Option<f64>) -> f64 {
        self.open
            .iter()
            .filter_map(|p| mark(&p.symbol).map(|px| p.unrealized(px)))
            .sum()
    }

    fn alloc_id(&mut self) -> String {
        self.next_id += 1;
        format!("sim-{}", self.next_id)
    }

    /// Reconcile an executed intent into the open set. No-op when the report is
    /// not `Filled` (a rejected/pending exec changes nothing on our books).
    pub fn apply(&mut self, intent: &TradeIntent, report: &ExecReport) {
        if report.status != ExecStatus::Filled {
            return;
        }
        match intent {
            TradeIntent::Open {
                symbol,
                dir,
                volume,
                sl,
                tp,
                source,
            } => {
                let price = report.fill_price.unwrap_or(0.0);
                let id = report.position_id.clone().unwrap_or_else(|| self.alloc_id());
                self.open.push(Position {
                    id,
                    symbol: symbol.clone(),
                    dir: *dir,
                    volume: *volume,
                    entry_price: price,
                    sl: *sl,
                    tp: *tp,
                    source: *source,
                    opened_at_bar: self.current_bar,
                });
                self.opened_count += 1;
            }
            TradeIntent::Close {
                position_id,
                volume,
                ..
            } => {
                if let Some(idx) = self.open.iter().position(|p| &p.id == position_id) {
                    let entry = self.open[idx].entry_price;
                    let sign = self.open[idx].dir.sign();
                    let fill = report.fill_price.unwrap_or(entry);
                    let pos_vol = self.open[idx].volume;
                    let close_vol = volume.unwrap_or(pos_vol).min(pos_vol);
                    self.realized_pnl += (fill - entry) * sign * close_vol;
                    if volume.is_none() || close_vol >= pos_vol {
                        self.open.remove(idx);
                        self.closed_count += 1;
                    } else {
                        self.open[idx].volume -= close_vol;
                    }
                }
            }
            TradeIntent::Amend {
                position_id,
                new_sl,
                new_tp,
            } => {
                if let Some(p) = self.open.iter_mut().find(|p| &p.id == position_id) {
                    if new_sl.is_some() {
                        p.sl = *new_sl;
                    }
                    if new_tp.is_some() {
                        p.tp = *new_tp;
                    }
                }
            }
            TradeIntent::Cancel { .. } => {}
        }
    }

    /// On each bar, produce `(Close intent, fill price)` pairs for every position
    /// of this symbol whose SL/TP the bar crossed — or whose `max_hold_bars`
    /// time stop has expired. The engine executes each at the returned level so
    /// realised P&L reflects the stop/target, not the bar close.
    ///
    /// ORDER OF PRECEDENCE: stop, then target, then the time stop. A bar that
    /// straddles both stop and target is already resolved adversely by
    /// [`Position::exit_hit`]; the time stop only fires on a bar where neither
    /// level was touched, and it fills at the bar CLOSE (the same price the GA
    /// evaluator uses for a `max_hold` exit).
    pub fn manage_on_bar(&mut self, bar: &LiveBar) -> Vec<(TradeIntent, f64)> {
        let max_hold = self.max_hold_bars;
        let current_bar = self.current_bar;
        let mut out = Vec::new();
        let mut timed_out = 0usize;
        for p in self.open.iter().filter(|p| p.symbol == bar.symbol) {
            if let Some((reason, price)) = p.exit_hit(bar) {
                out.push((
                    TradeIntent::Close {
                        position_id: p.id.clone(),
                        volume: None,
                        reason,
                    },
                    price,
                ));
                continue;
            }
            if let Some(limit) = max_hold {
                if current_bar.saturating_sub(p.opened_at_bar) >= limit {
                    timed_out += 1;
                    out.push((
                        TradeIntent::Close {
                            position_id: p.id.clone(),
                            volume: None,
                            reason: CloseReason::MaxHold,
                        },
                        bar.c,
                    ));
                }
            }
        }
        self.max_hold_exits += timed_out;
        out
    }
}
