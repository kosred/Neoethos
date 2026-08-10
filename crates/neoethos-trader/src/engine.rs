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

/// The replay's balance when there is no account to read (audit #265).
///
/// **#265 is CLOSED on the config side as of 2026-08-10.** There used to be
/// three numbers in the tree for one concept:
///   - `risk.initial_balance` = 10 000 — the ACCOUNT
///   - `models.backtest_runtime.initial_equity` = 100 000 — the denominator
///     every percentage the search ranked on was computed against
///   - this one, the replay's
///
/// The operator's decision: **the balance is READ FROM THE REAL ACCOUNT at
/// demo/live time, and there is no second constant.** So
/// `models.backtest_runtime.initial_equity` is DELETED (it is in
/// `load_seal::RETIRED_KEYS`), and
/// `neoethos_search::eval::BacktestRuntimeOverrides::from_settings` takes the
/// search's denominator from `risk.initial_balance` — the same field the live
/// drawdown floor reads, and the field the broker's reported balance belongs in.
///
/// This constant is NOT a third balance. Both replay front-ends build their
/// config through [`EngineConfig::for_replay_from_settings`], which takes
/// `risk.initial_balance`, so a replay reaches this value only when `Settings`
/// could not be read at all or the configured balance is unusable — and in both
/// cases it says so at WARN and `common_warnings` marks the run synthetic. It is
/// the "there was no account" value, not an alternative to one.
pub const DEFAULT_REPLAY_STARTING_BALANCE: f64 = 10_000.0;

/// Engine-wide knobs (Phase 1).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Starting balance for the P&L/equity bookkeeping.
    ///
    /// **#265: this is `risk.initial_balance` — the ACCOUNT — on every path a
    /// front-end can actually take.** [`EngineConfig::for_replay_from_settings`]
    /// fills it, and at demo/live time that field carries the balance the broker
    /// reports, so a demo account and a live account run the SAME engine on the
    /// SAME number and nothing about the models or strategies changes between
    /// them. `Default` falls back to [`DEFAULT_REPLAY_STARTING_BALANCE`], which
    /// means "no account could be read" and is reported as a synthetic run.
    pub starting_balance: f64,
    /// Max bars retained per (symbol, tf) rolling window (indicator warmup).
    pub window_cap: usize,
    /// Transaction costs charged by the mock execution adapter. Defaults to
    /// ZERO — see [`crate::execution::ReplayCostModel`]; the zero case is
    /// reported, not hidden.
    pub costs: crate::execution::ReplayCostModel,
    /// The GA evaluator's time stop, in bars. `None` ⇒ no time stop.
    pub max_hold_bars: Option<u64>,
    /// The GA evaluator's / live loop's break-even + trailing stop (audit
    /// #227). `None` ⇒ no trail, which is what this engine did unconditionally
    /// before 2026-08-10 — stop, target and time stop were its only exits while
    /// both of the paths it claims to mirror also move the stop after `+1R`.
    ///
    /// The replay helpers resolve it from `models.exit_policy` via
    /// `EvaluationConfig`, the same single recipient discovery and live read, so
    /// a run with the policy OFF is byte-identical to the old behaviour and a run
    /// with it ON models what the strategy was actually scored under.
    pub trailing: Option<crate::position::TrailingPolicy>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            starting_balance: DEFAULT_REPLAY_STARTING_BALANCE,
            window_cap: 512,
            costs: crate::execution::ReplayCostModel::zero(),
            max_hold_bars: None,
            trailing: None,
        }
    }
}

impl EngineConfig {
    /// The replay config a front-end that holds the operator's `Settings`
    /// should use: HIS starting balance and HIS broker costs, not the
    /// synthetic defaults (audit #224 / #265).
    ///
    /// This exists so the policy lives ONCE. Both front-ends —
    /// `neoethos-cli`'s `trader-replay` family and `neoethos-app`'s
    /// `POST /autonomous/replay` — passed `EngineConfig::default()`, which is
    /// a 10 000 balance and `ReplayCostModel::zero()`: every replay an operator
    /// could actually run reported percentage drawdown against a balance that
    /// was not his, on fills that charged nothing. This entry point takes the
    /// four numbers and enforces the RULES about them; front-ends holding a
    /// `Settings` should call [`EngineConfig::for_replay_from_settings`] below,
    /// which is the one adapter that reads those four fields off the config.
    ///
    /// * `initial_balance` — `risk.initial_balance`. A non-finite or
    ///   non-positive value keeps [`DEFAULT_REPLAY_STARTING_BALANCE`], because
    ///   the equity curve divides by it; the substitution is logged, never
    ///   silent, and `data_replay::common_warnings` then reports the run as
    ///   synthetic exactly as before.
    /// * `commission_per_lot_per_side` — the ONE-SIDE charge. The replay's
    ///   execution adapter charges it on entry AND on exit
    ///   (`execution.rs:173`/`:206`), so handing it a round-trip number bills
    ///   the operator twice. Callers convert with
    ///   `RiskConfig::round_trip_commission_per_lot() / 2.0`.
    /// * `pip_size` — `None` when the symbol is not in the metadata table. The
    ///   spread cannot be converted to price units without it, so the costs
    ///   stay ZERO and say so: a wrong cost is worse than a declared absent
    ///   one, because it looks like it was charged.
    pub fn for_replay(
        initial_balance: f64,
        spread_pips: f64,
        slippage_pips: f64,
        commission_per_lot_per_side: f64,
        pip_size: Option<f64>,
    ) -> Self {
        let starting_balance = if initial_balance.is_finite() && initial_balance > 0.0 {
            initial_balance
        } else {
            tracing::warn!(
                target: "neoethos_trader::engine",
                configured = initial_balance,
                fallback = DEFAULT_REPLAY_STARTING_BALANCE,
                "risk.initial_balance is not a usable balance — the replay's equity curve \
                 divides by it, so the synthetic default is used and the run is reported as \
                 synthetic"
            );
            DEFAULT_REPLAY_STARTING_BALANCE
        };
        let costs = match pip_size {
            Some(pip) => crate::execution::ReplayCostModel::from_pips(
                spread_pips,
                slippage_pips,
                commission_per_lot_per_side,
                pip,
            ),
            None => {
                tracing::warn!(
                    target: "neoethos_trader::engine",
                    "the symbol's pip size is unknown, so the spread cannot be converted to \
                     price units — this replay charges NOTHING. Add the symbol to the \
                     metadata table rather than trusting the result"
                );
                crate::execution::ReplayCostModel::zero()
            }
        };
        Self {
            starting_balance,
            costs,
            ..Self::default()
        }
    }

    /// The ONE `Settings` → [`EngineConfig`] adapter for every replay front-end.
    ///
    /// **Moved here 2026-08-10 (#229).** It used to be `replay_engine_config`
    /// in `neoethos-cli/src/main.rs`, private to the CLI. `neoethos-app`'s
    /// `POST /autonomous/replay` therefore could not call it and passed
    /// [`EngineConfig::default`] — `ReplayCostModel::zero()` and the synthetic
    /// [`DEFAULT_REPLAY_STARTING_BALANCE`] — so the operator's Replay button
    /// filled at the mark, charged nothing, and reported drawdown against a
    /// balance that was not his, while `data_replay`'s module header claimed
    /// the two front-ends produce byte-identical `EngineStats`. The CLI copy is
    /// DELETED; this is the only one.
    ///
    /// `None` settings is not an error — it is a replay whose costs and balance
    /// could not be resolved, which returns the synthetic default and says so
    /// at WARN. `data_replay::common_warnings` then reports the run as
    /// synthetic, so the disclosure survives.
    ///
    /// Commission is converted from the operator's round trip to the ONE-SIDE
    /// charge [`EngineConfig::for_replay`] documents, because the replay's
    /// execution adapter bills it on entry AND on exit.
    pub fn for_replay_from_settings(
        settings: Option<&neoethos_core::Settings>,
        symbol: &str,
    ) -> Self {
        let Some(settings) = settings else {
            tracing::warn!(
                target: "neoethos_trader::replay",
                symbol,
                "no config resolved — this replay fills at the mark, charging nothing, on a \
                 synthetic balance"
            );
            return Self::default();
        };
        let pip_size = neoethos_core::symbol_metadata::global_table()
            .lookup(symbol)
            .map(|meta| meta.pip_size);
        let risk = &settings.risk;
        let commission_per_side = risk.round_trip_commission_per_lot() / 2.0;
        tracing::info!(
            target: "neoethos_trader::replay",
            symbol,
            initial_balance = risk.initial_balance,
            spread_pips = risk.backtest_spread_pips,
            slippage_pips = risk.slippage_pips,
            commission_per_lot_per_side = commission_per_side,
            pip_size = ?pip_size,
            "replay balance and costs taken from the operator's config"
        );
        let mut cfg = Self::for_replay(
            risk.initial_balance,
            risk.backtest_spread_pips,
            risk.slippage_pips,
            commission_per_side,
            pip_size,
        );
        // ── The exit the strategy was SCORED under (audit #227) ──────────────
        //
        // `models.exit_policy` is the single recipient for the break-even /
        // trailing geometry: discovery reads it through `EvaluationConfig`
        // (`strategy_gene.rs:905-913`) and the live loop reads it directly
        // (`live_trading.rs:762`, `:1479-1493`). The replay read it NOWHERE, so
        // it modelled a strategy whose stop never moves against two paths on
        // which it does — while its own module header claimed parity.
        //
        // Same field, third reader. `trailing_enabled: false` (the shipped
        // default) leaves `trailing: None`, i.e. byte-identical to every replay
        // run before 2026-08-10. A pip size the metadata table does not have
        // also leaves it `None`, because the minimum lock is quoted in pips and
        // inventing a pip size would put the stop at a level no broker has.
        let exit = settings.models.exit_policy;
        cfg.trailing = if exit.trailing_enabled {
            let armed = pip_size.and_then(|pip| {
                crate::position::TrailingPolicy::new(
                    exit.trailing_be_trigger_r,
                    exit.trailing_stop_multiplier,
                    exit.trailing_min_lock_pips,
                    pip,
                )
            });
            if armed.is_none() {
                tracing::warn!(
                    target: "neoethos_trader::replay",
                    symbol,
                    be_trigger_r = exit.trailing_be_trigger_r,
                    stop_multiplier = exit.trailing_stop_multiplier,
                    min_lock_pips = exit.trailing_min_lock_pips,
                    pip_size = ?pip_size,
                    "models.exit_policy.trailing_enabled is ON but the geometry cannot move a \
                     stop (unknown pip size, or a non-finite / non-positive trigger or \
                     multiplier) — this replay runs WITHOUT the trail and says so in its \
                     fidelity report. It does not describe what live will do"
                );
            }
            armed
        } else {
            None
        };
        cfg
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
        positions.set_trailing(cfg.trailing);
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
