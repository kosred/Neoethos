//! Phase-1 contract types + trait seams for the autonomous trader.
//!
//! These are the stable boundaries the later phases plug real implementations
//! into (live data, the discovered `Gene` / `SoftVotingEnsemble`, the cTrader
//! `ExecutionAdapter`, the core `RiskManager`). Phase 1 ships lightweight types
//! + stub impls so the end-to-end loop is provable OFFLINE with zero broker
//! calls — a pure dry-run, NOT a parallel "paper" product.

use serde::{Deserialize, Serialize};

/// Trade direction a signal or position can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Long,
    Short,
    Flat,
}

impl Direction {
    pub fn opposite(self) -> Direction {
        match self {
            Direction::Long => Direction::Short,
            Direction::Short => Direction::Long,
            Direction::Flat => Direction::Flat,
        }
    }

    /// +1 long, -1 short, 0 flat — the multiplier for signed P&L.
    pub fn sign(self) -> f64 {
        match self {
            Direction::Long => 1.0,
            Direction::Short => -1.0,
            Direction::Flat => 0.0,
        }
    }
}

/// Risk/sizing regime. Mirrors `neoethos_core` trading_mode; kept local for the
/// Phase-1 skeleton and mapped to the core enum when the real RiskManager is
/// wired (Phase 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeMode {
    PropFirm,
    Risky,
}

/// Where a portfolio entry's signal comes from. Phase 1 carries lightweight
/// descriptors (ids/paths) rather than the heavy `Gene` / ensemble objects; the
/// SignalEngine resolves them to a real signal in Phase 4. The variants mirror
/// design §6 (`Gene` / `Ensemble` / `Blend`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategySource {
    Gene { id: String },
    Ensemble { dir: String },
    Blend { gene_id: String, ensemble_dir: String },
}

/// One thing the trader watches: a (symbol, base_tf, higher_tfs) tuple with the
/// signal source + risk mode. Produced by `PortfolioRegistry` — the source of
/// truth for "what to watch" (NOT the watchlist).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortfolioEntry {
    pub symbol: String,
    pub base_tf: String,
    #[serde(default)]
    pub higher_tfs: Vec<String>,
    pub source: StrategySource,
    pub mode: TradeMode,
}

/// A closed OHLCV bar for (symbol, tf). Replayed from history in Phase 1;
/// emitted by `LiveMarketData` on bar-close in Phase 2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveBar {
    pub symbol: String,
    pub tf: String,
    pub o: f64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    #[serde(default)]
    pub volume: f64,
    /// Bar OPEN time, milliseconds (candle-open indexing — the canonical
    /// timestamp policy, see `neoethos_core::contracts::temporal`).
    pub ts: i64,
}

/// Which engine produced a signal (for attribution + the blend gate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalSource {
    Strategy,
    Ensemble,
    Blend,
}

/// A directional call with confidence, for one (symbol, base_tf).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signal {
    pub symbol: String,
    pub dir: Direction,
    /// `0.0..=1.0` strategy/model confidence; sizing scales with this.
    pub confidence: f64,
    pub source: SignalSource,
}

/// Why a position is being closed (for the journal + attribution).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloseReason {
    StopLoss,
    TakeProfit,
    Signal,
    Manual,
}

/// A concrete action the DecisionEngine wants taken. `volume` is in lots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TradeIntent {
    Open {
        symbol: String,
        dir: Direction,
        volume: f64,
        sl: Option<f64>,
        tp: Option<f64>,
        source: SignalSource,
    },
    /// `volume: None` closes the whole position; `Some(v)` is a partial close.
    Close {
        position_id: String,
        volume: Option<f64>,
        reason: CloseReason,
    },
    Amend {
        position_id: String,
        new_sl: Option<f64>,
        new_tp: Option<f64>,
    },
    Cancel {
        order_id: String,
    },
}

impl TradeIntent {
    /// The symbol this intent concerns, when it is positionable (`Open`).
    pub fn symbol(&self) -> Option<&str> {
        match self {
            TradeIntent::Open { symbol, .. } => Some(symbol),
            _ => None,
        }
    }

    /// Short stable label for logs/telemetry.
    pub fn kind(&self) -> &'static str {
        match self {
            TradeIntent::Open { .. } => "open",
            TradeIntent::Close { .. } => "close",
            TradeIntent::Amend { .. } => "amend",
            TradeIntent::Cancel { .. } => "cancel",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecStatus {
    Filled,
    Rejected,
    Pending,
}

/// The outcome of executing an intent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecReport {
    pub status: ExecStatus,
    pub fill_price: Option<f64>,
    pub position_id: Option<String>,
    pub detail: String,
}

/// A snapshot of account state the RiskGate reasons over.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub equity: f64,
    pub balance: f64,
    pub open_positions: usize,
    pub realized_pnl: f64,
}

/// Risk-gate rejection tiers (Phase-1 placeholder; the real `RiskManager` /
/// `RiskyModeManager` tiers wire in at Phase 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KillSwitchTier {
    DailyLoss,
    MaxDrawdown,
    EquityFloor,
    ExposureCap,
}

// ── Trait seams ──────────────────────────────────────────────────────────────

/// Produces a signal for a portfolio entry from its rolling bar window. Phase 1
/// = a deterministic stub; Phase 4 wires the `Gene` + `SoftVotingEnsemble`
/// regime-conditional blend (design §7).
pub trait SignalEngine {
    fn evaluate(&mut self, entry: &PortfolioEntry, window: &[LiveBar]) -> Signal;
}

/// Gates every intent before execution. Phase 1 = permissive stub; Phase 5
/// wires `RiskManager::check_trade_allowed` / `RiskyModeManager`.
pub trait RiskGate {
    fn check(&self, intent: &TradeIntent, account: &AccountSnapshot) -> Result<(), KillSwitchTier>;
}

/// Executes an intent at the observed `mark_price` (the bar close / SL-TP level
/// at decision time — the Phase-1 mock fills there; the real cTrader adapter
/// ignores it and lets the broker fill). Phase 1 = `MockExecutionAdapter` (sim
/// fills, ZERO broker calls); Phase 5 wires the single cTrader `broker_api`
/// path — demo vs live is the connected ACCOUNT, not separate code.
pub trait ExecutionAdapter {
    fn execute(&mut self, intent: &TradeIntent, mark_price: f64) -> anyhow::Result<ExecReport>;
}
