//! # neoethos-trader — the autonomous live-trading engine (Phase 1)
//!
//! The single library service both the UI (`neoethos-app`) and the CLI
//! (`neoethos-cli`) drive as thin front-ends — there is no UI-only or CLI-only
//! trading logic, ever (design `docs/v0.5-autonomous-trader-design.md` §1.1).
//!
//! ## What Phase 1 delivers
//! The complete bar→signal→decision→risk→execution→position loop ([`engine`]),
//! provable end-to-end OFFLINE via the [`replay`] harness with **zero broker
//! calls** — a pure dry-run, not a parallel "paper" product. Everything heavy is
//! stubbed behind a trait seam so later phases plug in the real pieces without
//! the loop changing:
//!
//! | Seam ([`contracts`]) | Phase 1 stub | Wired later |
//! |---|---|---|
//! | [`SignalEngine`] | [`signal::MomentumStubSignal`] | Gene + `SoftVotingEnsemble` blend (P4) |
//! | [`RiskGate`] | [`risk::PermissiveRiskGate`] / [`risk::MaxOpenPositionsGate`] | core `RiskManager` / `RiskyModeManager` (P5) |
//! | [`ExecutionAdapter`] | [`execution::MockExecutionAdapter`] | cTrader `broker_api` (P5; demo vs live = the account) |
//! | [`portfolio::PortfolioRegistry`] | explicit list / JSON manifest | promotion-artifact scan + hot-reload (P2) |
//!
//! Live market data (P2) and the rolling multi-TF feature cube (P3) feed
//! [`contracts::LiveBar`]s and the SignalEngine respectively.

pub mod contracts;
pub mod data_replay;
pub mod decision;
pub mod engine;
pub mod execution;
pub mod portfolio;
pub mod position;
pub mod replay;
pub mod risk;
pub mod signal;

// Curated surface so front-ends can `use neoethos_trader::*` ergonomically.
pub use contracts::{
    AccountSnapshot, CloseReason, Direction, ExecReport, ExecStatus, ExecutionAdapter,
    KillSwitchTier, LiveBar, PortfolioEntry, RiskGate, Signal, SignalEngine, SignalSource,
    StrategySource, TradeIntent, TradeMode,
};
pub use data_replay::{load_bars_from_dir, replay_symbol_from_dir};
pub use decision::{DecisionConfig, DecisionEngine};
pub use engine::{AutonomousEngine, EngineConfig, EngineStats};
pub use execution::MockExecutionAdapter;
pub use portfolio::PortfolioRegistry;
pub use position::{Position, PositionManager};
pub use replay::replay;
pub use risk::{MaxOpenPositionsGate, PermissiveRiskGate};
pub use signal::MomentumStubSignal;
