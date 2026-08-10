//! The autonomous trading loop — design §3, wired over the trait seams.
//!
//! `on_bar` is the whole engine: per closed bar it (1) manages open positions
//! (SL/TP exits), then (2) on a base-TF bar evaluates the signal → decision →
//! risk gate → execution → position update. It is front-end-agnostic: the
//! replay harness drives it offline (Phase 1) and the live supervisor will drive
//! it from `BarClosed` events (Phase 2) — identical logic either way.

use std::collections::HashMap;

use serde::Serialize;

use crate::contracts::{
    AccountSnapshot, ExecStatus, ExecutionAdapter, LiveBar, RiskGate, SignalEngine, TradeIntent,
};
use crate::decision::DecisionEngine;
use crate::portfolio::PortfolioRegistry;
use crate::position::PositionManager;

/// The replay's SYNTHETIC starting balance (audit #265).
///
/// This is the third of three numbers in the tree for one concept:
///   - `neoethos-core config.rs:528` `risk.initial_balance` = 10_000.0
///   - `neoethos-core config.rs:1767` `backtest_runtime.initial_equity` = 100_000.0
///     — and THAT one is the GPU kernel's compounding denominator, so every
///     percentage return the search ranks on is computed against a balance
///     ten times the operator's real account
///   - this one, the replay's
///
/// It is NOT reconciled here: `neoethos-core/src/config.rs` is owned by another
/// workflow this wave, and silently changing which of three numbers the replay
/// uses would move the reported drawdown without anybody choosing it. What
/// changed is that the number is now NAMED, and a replay that runs on it says
/// so in its fidelity report instead of presenting it as the operator's
/// account.
pub const DEFAULT_REPLAY_STARTING_BALANCE: f64 = 10_000.0;

/// Engine-wide knobs (Phase 1).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Notional starting balance for the P&L/equity bookkeeping.
    /// Defaults to [`DEFAULT_REPLAY_STARTING_BALANCE`], which is NOT the
    /// operator's account balance — front-ends holding `Settings` should pass
    /// `risk.initial_balance` instead.
    pub starting_balance: f64,
    /// Max bars retained per (symbol, tf) rolling window (indicator warmup).
    pub window_cap: usize,
    /// Transaction costs charged by the mock execution adapter. Defaults to
    /// ZERO — see [`crate::execution::ReplayCostModel`]; the zero case is
    /// reported, not hidden.
    pub costs: crate::execution::ReplayCostModel,
    /// The GA evaluator's time stop, in bars. `None` ⇒ no time stop.
    pub max_hold_bars: Option<u64>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            starting_balance: DEFAULT_REPLAY_STARTING_BALANCE,
            window_cap: 512,
            costs: crate::execution::ReplayCostModel::zero(),
            max_hold_bars: None,
        }
    }
}

/// A point-in-time summary of the engine's activity (status API + replay report).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct EngineStats {
    pub bars_processed: usize,
    pub signals_evaluated: usize,
    pub intents_emitted: usize,
    pub intents_executed: usize,
    pub intents_blocked: usize,
    pub positions_opened: usize,
    pub positions_closed: usize,
    pub open_positions: usize,
    /// Realised P&L NET of commission (spread and slippage are already inside
    /// the fill prices). Before 2026-08-09 this was gross of all three.
    pub realized_pnl: f64,
    pub equity: f64,
    /// Commission charged across the run, in account currency.
    #[serde(default)]
    pub commission_paid: f64,
    /// Positions closed by the `max_hold_bars` time stop.
    #[serde(default)]
    pub max_hold_exits: usize,
    /// **Every way this run is NOT the operator's strategy** (audit #220–#231).
    ///
    /// A diagnostic tool that gives wrong diagnostics is worse than no tool, so
    /// the number now carries its own disclaimer: each entry names one stub,
    /// synthetic input or divergence from the GA evaluator that shaped the
    /// figures above. An EMPTY list is the only state in which these numbers
    /// may be compared with live results.
    #[serde(default)]
    pub fidelity_warnings: Vec<String>,
}

/// The trading loop, generic over the three trait seams so tests inject stubs
/// and production injects the real Gene/ensemble signal, the core RiskManager,
/// and the cTrader execution adapter — without the loop changing.
pub struct AutonomousEngine<S, R, E> {
    registry: PortfolioRegistry,
    signal: S,
    risk: R,
    exec: E,
    decision: DecisionEngine,
    positions: PositionManager,
    cfg: EngineConfig,
    windows: HashMap<(String, String), Vec<LiveBar>>,
    marks: HashMap<String, f64>,
    account: AccountSnapshot,
    bars: usize,
    signals: usize,
    intents: usize,
    executed: usize,
    blocked: usize,
}

impl<S: SignalEngine, R: RiskGate, E: ExecutionAdapter> AutonomousEngine<S, R, E> {
    pub fn new(
        registry: PortfolioRegistry,
        signal: S,
        risk: R,
        exec: E,
        decision: DecisionEngine,
        cfg: EngineConfig,
    ) -> Self {
        let account = AccountSnapshot {
            equity: cfg.starting_balance,
            balance: cfg.starting_balance,
            open_positions: 0,
            realized_pnl: 0.0,
        };
        let mut positions = PositionManager::new();
        positions.set_max_hold_bars(cfg.max_hold_bars);
        Self {
            registry,
            signal,
            risk,
            exec,
            decision,
            positions,
            cfg,
            windows: HashMap::new(),
            marks: HashMap::new(),
            account,
            bars: 0,
            signals: 0,
            intents: 0,
            executed: 0,
            blocked: 0,
        }
    }

    pub fn registry(&self) -> &PortfolioRegistry {
        &self.registry
    }

    pub fn positions(&self) -> &PositionManager {
        &self.positions
    }

    pub fn account(&self) -> &AccountSnapshot {
        &self.account
    }

    /// Borrow the execution adapter (e.g. to read the mock fill log in tests).
    pub fn execution(&self) -> &E {
        &self.exec
    }

    pub fn stats(&self) -> EngineStats {
        let commission = self.exec.charged_costs();
        EngineStats {
            bars_processed: self.bars,
            signals_evaluated: self.signals,
            intents_emitted: self.intents,
            intents_executed: self.executed,
            intents_blocked: self.blocked,
            positions_opened: self.positions.opened_count(),
            positions_closed: self.positions.closed_count(),
            open_positions: self.positions.open_count(),
            realized_pnl: self.positions.realized_pnl() - commission,
            equity: self.account.equity,
            commission_paid: commission,
            max_hold_exits: self.positions.max_hold_exits(),
            // Filled in by the caller that KNOWS what it stubbed — the replay
            // harness. The engine itself cannot see which signal engine or risk
            // gate it was handed.
            fidelity_warnings: Vec::new(),
        }
    }

    fn refresh_account(&mut self) {
        let unreal = {
            let marks = &self.marks;
            self.positions.unrealized_total(|s| marks.get(s).copied())
        };
        // Commission is a real charge against the account; spread and slippage
        // are already inside the fill prices, so counting them here too would
        // double-charge.
        let realized = self.positions.realized_pnl() - self.exec.charged_costs();
        self.account.realized_pnl = realized;
        self.account.open_positions = self.positions.open_count();
        self.account.balance = self.cfg.starting_balance + realized;
        self.account.equity = self.account.balance + unreal;
    }

    fn execute_intent(&mut self, intent: &TradeIntent, mark: f64) {
        self.intents += 1;
        match self.exec.execute(intent, mark) {
            Ok(report) => {
                if report.status == ExecStatus::Filled {
                    self.executed += 1;
                }
                self.positions.apply(intent, &report);
            }
            Err(e) => {
                tracing::error!(
                    target: "neoethos_trader::engine",
                    intent = intent.kind(),
                    error = %e,
                    "execution failed"
                );
            }
        }
    }

    /// Drive one closed bar through the loop.
    pub fn on_bar(&mut self, bar: &LiveBar) {
        self.bars += 1;
        // Stamp the bar index BEFORE any fill so a position opened on this bar
        // records this bar as its origin (the time stop counts from here).
        self.positions.set_bar(self.bars as u64);
        self.marks.insert(bar.symbol.clone(), bar.c);

        // Rolling window (one per symbol/tf), capped for warmup memory.
        let key = (bar.symbol.clone(), bar.tf.clone());
        {
            let buf = self.windows.entry(key.clone()).or_default();
            buf.push(bar.clone());
            if buf.len() > self.cfg.window_cap {
                let excess = buf.len() - self.cfg.window_cap;
                buf.drain(0..excess);
            }
        }

        // 1. Manage existing positions first (SL/TP exits at the level hit).
        let managed = self.positions.manage_on_bar(bar);
        for (intent, fill_price) in managed {
            self.execute_intent(&intent, fill_price);
        }
        self.refresh_account();

        // 2. Signal → decision → risk → execution, only on a base-TF bar.
        if let Some(entry) = self.registry.entry_for(&bar.symbol, &bar.tf).cloned() {
            self.signals += 1;
            let window = self.windows.get(&key).cloned().unwrap_or_default();
            let signal = self.signal.evaluate(&entry, &window);
            let open = self.positions.positions_for(&bar.symbol);
            // Bound to a local first: `intent` now takes `&mut self` (it latches
            // the one-shot synthetic-bracket warning), and `execute_intent`
            // below takes `&mut self` too.
            let decided = self.decision.intent(&signal, &open, bar.c);
            if let Some(intent) = decided {
                match self.risk.check(&intent, &self.account) {
                    Ok(()) => self.execute_intent(&intent, bar.c),
                    Err(tier) => {
                        self.blocked += 1;
                        tracing::debug!(
                            target: "neoethos_trader::engine",
                            symbol = %bar.symbol,
                            intent = intent.kind(),
                            ?tier,
                            "intent blocked by risk gate"
                        );
                    }
                }
            }
            self.refresh_account();
        }
    }
}
