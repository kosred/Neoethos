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
//! | [`RiskGate`] | [`risk::PermissiveRiskGate`] / [`risk::MaxOpenPositionsGate`] | `RiskyModeManager` (P5). NOT the core `RiskManager` — it has no production constructor; see [`risk`] |
//! | [`ExecutionAdapter`] | [`execution::MockExecutionAdapter`] | cTrader `broker_api` (P5; demo vs live = the account) |
//! | [`portfolio::PortfolioRegistry`] | explicit list / JSON manifest | promotion-artifact scan + hot-reload (P2) |
//!
//! Live market data (P2) and the rolling multi-TF feature cube (P3) feed
//! [`contracts::LiveBar`]s and the SignalEngine respectively.
//!
//! ## Read this before believing a replay number (2026-08-09, audit #220–#231)
//!
//! Several of the seams above are STILL stubbed, and the replay screen has been
//! reporting their output as if it were the operator's strategy. Every
//! `data_replay::replay_*` entry point now returns
//! [`engine::EngineStats::fidelity_warnings`] — an explicit list of every stub,
//! synthetic input and divergence-from-the-GA-evaluator in the path that
//! produced the numbers — and logs the same list at `warn`. **A run whose list
//! is non-empty must not be compared with live results.**
//!
//! What changed with it: the real-gene replay paths now place the GENE'S OWN
//! bracket instead of a 0.5 %-of-price synthetic stop, exit on stop / target /
//! `max_hold_bars` like the evaluator rather than on a flat or reversed signal,
//! and charge whatever [`execution::ReplayCostModel`] the caller supplies. That
//! model still defaults to zero — inventing a spread would be a different lie —
//! so the zero case is the first line of the fidelity report.
//!
//! 2026-08-10 (audit #227): the fourth exit arrived. The crate had NO
//! break-even move and NO trailing stop anywhere, while both paths it claims to
//! mirror move the stop once a trade reaches `+be_trigger_r × R`. It now reads
//! the same `models.exit_policy` they do, through the one
//! [`engine::EngineConfig::for_replay_from_settings`] adapter, and reports the
//! armed geometry (or its absence) in the fidelity list. With the shipped
//! default — `trailing_enabled: false` — the numbers are unchanged.

pub mod blend_signal;
pub mod contracts;
pub mod data_replay;
pub mod decision;
pub mod engine;
pub mod execution;
pub mod gene_signal;
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
pub use blend_signal::{
    BlendConfig, BlendMode, BlendedSignalEngine, DEFAULT_BLEND_GATE_FLOOR,
    DEFAULT_BLEND_VETO_BELOW, MlDecision, blend_decision,
};
pub use data_replay::{
    load_bars_from_dir, ohlcv_to_livebars, replay_portfolio_from_dir, replay_symbol_from_dir,
};
#[cfg(feature = "ml-blend")]
pub use data_replay::replay_blend_from_dir;
pub use gene_signal::{
    PrecomputedSignalEngine, combine_gene_signals, combine_gene_signals_with_brackets,
    combine_gene_signals_with_confidence,
};
pub use decision::{DEFAULT_SYNTHETIC_STOP_FRAC, DecisionConfig, DecisionEngine};
pub use engine::{
    AutonomousEngine, DEFAULT_REPLAY_STARTING_BALANCE, EngineConfig, EngineStats,
};
pub use execution::{MockExecutionAdapter, ReplayCostModel};
pub use portfolio::PortfolioRegistry;
pub use position::{Position, PositionManager, TrailingPolicy};
pub use replay::replay;
pub use risk::{MaxOpenPositionsGate, PermissiveRiskGate};
pub use signal::MomentumStubSignal;
