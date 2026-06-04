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
}

impl PositionManager {
    pub fn new() -> Self {
        Self::default()
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
    /// of this symbol whose SL/TP the bar crossed. The engine executes each at
    /// the returned level so realised P&L reflects the stop/target, not the
    /// bar close.
    pub fn manage_on_bar(&self, bar: &LiveBar) -> Vec<(TradeIntent, f64)> {
        self.open
            .iter()
            .filter(|p| p.symbol == bar.symbol)
            .filter_map(|p| {
                p.exit_hit(bar).map(|(reason, price)| {
                    (
                        TradeIntent::Close {
                            position_id: p.id.clone(),
                            volume: None,
                            reason,
                        },
                        price,
                    )
                })
            })
            .collect()
    }
}
