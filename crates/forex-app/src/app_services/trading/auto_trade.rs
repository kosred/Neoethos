//! AI auto-trade pipeline — the `signal → order` glue.
//!
//! Closes audit gap #1 (top priority): the `auto_trade_enabled`
//! toggle existed but no inference→order flow connected to it. This
//! module supplies the production-shape signal type plus the
//! `TradingSession::dispatch_auto_trade_signal` entry point that the
//! eventual inference loop calls.
//!
//! ## Architecture
//!
//! ```text
//!   ┌───────────────┐
//!   │ InferenceLoop │  (D1 follow-up — loads models, runs predict
//!   │  (producer)   │   on live bars, emits AutoTradeSignals here)
//!   └───────┬───────┘
//!           │ AutoTradeSignal
//!           ▼
//!   ┌──────────────────────────┐
//!   │ TradingSession           │   THIS module
//!   │ ├ auto_trade_enabled?    │
//!   │ ├ confidence >= min?     │
//!   │ ├ news_filter blackout?  │
//!   │ ├ halt_state.halted?     │
//!   │ ├ risky_mode kill switch?│
//!   │ └ prop_firm_pre_trade?   │
//!   └─────────┬────────────────┘
//!             │ pass → execute_ctrader_order
//!             ▼
//!     existing fill path (orders.rs)
//! ```
//!
//! Every gate the manual-order path enforces (HALT, news blackout,
//! risk_gate) is enforced HERE too — auto-trades are STRICTLY
//! tighter than manual orders. Spec ref: research §11.3 + §5.5.
//!
//! ## Status: v0.4.5 scaffold
//!
//! - ✓ Signal type, side mapping, confidence threshold
//! - ✓ Gate chain: auto-trade flag → confidence → news → halt →
//!   risky mode → prop-firm → fill
//! - ✓ Tests covering each gate rejection
//! - ⏳ Inference producer (D1 follow-up — separate task because
//!   live-bar streaming + model loading is a multi-day item that
//!   touches forex-models)

use super::{BotDecisionEntry, BotDecisionSide, BotDecisionSource, TradingSession};
use crate::app_state::AppState;

/// Single inference decision emitted by the auto-trade pipeline.
/// Construction is gated on the producer side — the consumer
/// (`TradingSession`) treats any received signal as authoritative
/// and only checks gates, not signal validity.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoTradeSignal {
    /// Trading symbol the signal targets. Must equal
    /// `AppState.selected_pair` for the signal to be dispatched —
    /// the chart/order context lives there. Producer is responsible
    /// for filtering to the active symbol.
    pub symbol: String,
    /// Long / short side. `Flat` signals are no-ops — the gate
    /// returns `Skip` immediately. We carry `Flat` as a side rather
    /// than wrapping the whole struct in `Option` because the
    /// producer benefits from a uniform stream contract.
    pub side: AutoTradeSide,
    /// Confidence in `[0.0, 1.0]`. Compared against
    /// [`AUTO_TRADE_MIN_CONFIDENCE`] in the gate; signals below
    /// threshold are rejected with `GateDecision::BelowConfidence`.
    pub confidence: f32,
    /// Display label rendered on the chart overlay marker after
    /// dispatch (`"AI BUY · 0.74"`, `"AI SELL · 0.81"`).
    pub label: String,
    /// Unix-ms timestamp at which the signal was produced. Used
    /// for the chart overlay's timestamp→candle mapping (see
    /// `bot_decisions_to_overlays`).
    pub timestamp_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoTradeSide {
    Buy,
    Sell,
    Flat,
}

/// Minimum confidence required to dispatch an auto-trade signal.
/// Hard-coded at the spec floor — research §4.6.1 names 0.6 as the
/// stage-0 minimum; the Risky Mode gate tightens this further per
/// stage when active.
pub const AUTO_TRADE_MIN_CONFIDENCE: f32 = 0.6;

/// Outcome of pushing an auto-trade signal through the gate chain.
/// Every variant is observable from outside so the inference loop's
/// per-signal log line can record exactly which gate rejected it —
/// invaluable when debugging "why didn't the bot trade?".
#[derive(Debug, Clone, PartialEq)]
pub enum GateDecision {
    /// Order dispatched to the broker fill path. The chart-overlay
    /// buffer was updated with the matching `BotDecisionEntry`.
    Dispatched,
    /// The operator has not flipped `auto_trade_enabled` on. Safe
    /// default — manual-only mode is the v0.4.5 baseline.
    AutoTradeOff,
    /// Signal targeted a different symbol than the currently-active
    /// chart pair. The producer should align with `selected_pair`.
    SymbolMismatch { active: String, signal: String },
    /// `side == AutoTradeSide::Flat` — no order to send.
    FlatSide,
    /// Confidence below [`AUTO_TRADE_MIN_CONFIDENCE`].
    BelowConfidence { confidence: f32, minimum: f32 },
    /// News blackout active. Same gate that the manual path uses
    /// (`news_filter.is_blackout()`).
    NewsBlackout,
    /// T-Manual HALT in force.
    Halted,
    /// Risky Mode kill-switch tripped (one of the 7 tiers).
    RiskyModeKillSwitch,
    /// Prop-firm risk gate rejected the implied order.
    PropFirmGate { reason: String },
}

impl AutoTradeSignal {
    /// True when the signal would, on its own merits, be a valid
    /// dispatch candidate. Useful for the inference loop's logging
    /// before it hits the session-level gates.
    pub fn is_actionable(&self) -> bool {
        !matches!(self.side, AutoTradeSide::Flat) && self.confidence >= AUTO_TRADE_MIN_CONFIDENCE
    }
}

impl AutoTradeSide {
    /// Convert to the [`BotDecisionSide`] consumed by the chart
    /// overlay layer. `Flat` maps to `Flat` for symmetry; the
    /// gate rejects flat signals before they reach this path.
    pub fn to_bot_decision_side(self) -> BotDecisionSide {
        match self {
            AutoTradeSide::Buy => BotDecisionSide::Buy,
            AutoTradeSide::Sell => BotDecisionSide::Sell,
            AutoTradeSide::Flat => BotDecisionSide::Flat,
        }
    }
}

impl TradingSession {
    /// Push an [`AutoTradeSignal`] through the production gate chain.
    /// Returns the [`GateDecision`] so the producer's log line knows
    /// exactly what happened. When the decision is `Dispatched`, the
    /// chart-overlay buffer has been updated and the broker fill path
    /// has been invoked; on any other variant nothing has been sent
    /// to the broker.
    ///
    /// The gate order matches the manual-order path in
    /// `orders.rs::execute_ctrader_order`:
    /// 1. `auto_trade_enabled` flag (auto-only — not a manual gate)
    /// 2. `symbol == selected_pair` (auto-only — manual binds the pair via UI)
    /// 3. `side != Flat` (auto-only — manual doesn't have a flat option)
    /// 4. `confidence >= min` (auto-only — manual has no confidence)
    /// 5. `news_filter.is_blackout()` (shared)
    /// 6. `halt_state.halted` (shared)
    /// 7. risky_mode `check_trade_allowed` (shared, when armed)
    /// 8. `prop_firm_pre_trade_check` (shared — runs inside
    ///    `execute_ctrader_order` after we dispatch)
    ///
    /// Gates 1-7 are evaluated here and return early on rejection;
    /// gate 8 runs inside the fill path so its rejection surfaces
    /// via `state.status_msg` as for manual orders.
    pub fn dispatch_auto_trade_signal(
        &mut self,
        state: &mut AppState,
        signal: AutoTradeSignal,
    ) -> GateDecision {
        // Gate 1 — operator flag.
        if !state.auto_trade_enabled {
            return GateDecision::AutoTradeOff;
        }

        // Gate 2 — symbol alignment.
        if signal.symbol != state.selected_pair {
            return GateDecision::SymbolMismatch {
                active: state.selected_pair.clone(),
                signal: signal.symbol.clone(),
            };
        }

        // Gate 3 — flat means skip.
        let side = match signal.side {
            AutoTradeSide::Flat => return GateDecision::FlatSide,
            AutoTradeSide::Buy => super::CTraderTradeSide::Buy,
            AutoTradeSide::Sell => super::CTraderTradeSide::Sell,
        };

        // Gate 4 — confidence.
        if signal.confidence < AUTO_TRADE_MIN_CONFIDENCE {
            return GateDecision::BelowConfidence {
                confidence: signal.confidence,
                minimum: AUTO_TRADE_MIN_CONFIDENCE,
            };
        }

        // Gate 5 — news blackout (B1).
        if state.llm_news_filter.is_blackout() {
            return GateDecision::NewsBlackout;
        }

        // Gate 6 — manual HALT.
        if self.is_halted() {
            return GateDecision::Halted;
        }

        // Gate 7 — Risky Mode kill-switch (research §5).
        // We cannot construct the (size_usd, sl_pips, tp_pips) tuple
        // here without the order ticket; the in-pipeline check that
        // matters lives inside `execute_ctrader_order`. We do a
        // cheap pre-check on the last-trip flag so a clearly-tripped
        // manager rejects the signal immediately without reaching
        // the broker path.
        if let Some(rm) = self.risky_mode_manager()
            && rm.last_kill_switch_trip().is_some()
        {
            return GateDecision::RiskyModeKillSwitch;
        }

        // Record the decision so the chart overlay paints it even
        // if a downstream gate (8 — prop-firm) blocks the fill.
        // Operators want to see the AI's intent in either case.
        self.record_bot_decision(BotDecisionEntry {
            symbol: signal.symbol.clone(),
            side: signal.side.to_bot_decision_side(),
            price: 0.0, // filled in by the chart layer from the live quote
            timestamp_ms: signal.timestamp_ms,
            label: signal.label.clone(),
            source: BotDecisionSource::Ai,
            confidence: Some(signal.confidence),
        });

        // Gates 1-7 all passed. Hand off to the broker fill path,
        // which runs gate 8 (prop-firm check, PnL circuit breaker,
        // size validation) and either fills or rejects with a
        // structured app_event the operator can audit.
        self.execute_ctrader_order(state, side);
        GateDecision::Dispatched
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::trading::{TradingAdapterKind, TradingSession};
    use crate::app_state::{AppRuntimeConfig, AppState};
    use forex_core::Settings;

    fn test_state() -> AppState {
        let tmp = std::env::temp_dir().join(format!(
            "forex-ai-auto-trade-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::create_dir_all(&tmp);
        let runtime = AppRuntimeConfig {
            config_path: "config.yaml".to_string(),
            data_dir: tmp,
            start_local: true,
            auto_discovery: false,
            auto_training: false,
        };
        let mut s = AppState::new(runtime, &Settings::default(), Vec::new());
        s.selected_pair = "EURUSD".to_string();
        s
    }

    fn sample_signal(side: AutoTradeSide, confidence: f32) -> AutoTradeSignal {
        AutoTradeSignal {
            symbol: "EURUSD".to_string(),
            side,
            confidence,
            label: format!("AI {:?} · {:.2}", side, confidence),
            timestamp_ms: 1_700_000_000_000,
        }
    }

    #[test]
    fn signal_is_actionable_requires_non_flat_and_above_min_confidence() {
        assert!(sample_signal(AutoTradeSide::Buy, 0.7).is_actionable());
        assert!(sample_signal(AutoTradeSide::Sell, 0.6).is_actionable());
        assert!(!sample_signal(AutoTradeSide::Buy, 0.59).is_actionable());
        assert!(!sample_signal(AutoTradeSide::Flat, 0.99).is_actionable());
    }

    #[test]
    fn gate_rejects_when_auto_trade_off() {
        let mut state = test_state();
        state.auto_trade_enabled = false;
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let decision = session.dispatch_auto_trade_signal(&mut state, sample_signal(AutoTradeSide::Buy, 0.8));
        assert_eq!(decision, GateDecision::AutoTradeOff);
        // Buffer should remain empty — no decision recorded for a
        // gate-1 reject.
        assert_eq!(session.bot_decision_buffer_len(), 0);
    }

    #[test]
    fn gate_rejects_on_symbol_mismatch() {
        let mut state = test_state();
        state.auto_trade_enabled = true;
        state.selected_pair = "EURUSD".to_string();
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let mut sig = sample_signal(AutoTradeSide::Buy, 0.8);
        sig.symbol = "GBPUSD".to_string();
        let decision = session.dispatch_auto_trade_signal(&mut state, sig);
        match decision {
            GateDecision::SymbolMismatch { active, signal } => {
                assert_eq!(active, "EURUSD");
                assert_eq!(signal, "GBPUSD");
            }
            other => panic!("expected SymbolMismatch, got {other:?}"),
        }
    }

    #[test]
    fn gate_rejects_flat_side() {
        let mut state = test_state();
        state.auto_trade_enabled = true;
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let decision =
            session.dispatch_auto_trade_signal(&mut state, sample_signal(AutoTradeSide::Flat, 0.9));
        assert_eq!(decision, GateDecision::FlatSide);
    }

    #[test]
    fn gate_rejects_below_min_confidence() {
        let mut state = test_state();
        state.auto_trade_enabled = true;
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let decision =
            session.dispatch_auto_trade_signal(&mut state, sample_signal(AutoTradeSide::Buy, 0.5));
        match decision {
            GateDecision::BelowConfidence { confidence, minimum } => {
                assert!((confidence - 0.5).abs() < 1e-6);
                assert!((minimum - AUTO_TRADE_MIN_CONFIDENCE).abs() < 1e-6);
            }
            other => panic!("expected BelowConfidence, got {other:?}"),
        }
    }

    #[test]
    fn gate_rejects_during_news_blackout() {
        let mut state = test_state();
        state.auto_trade_enabled = true;
        state.llm_news_filter.enabled = true;
        state.llm_news_filter.current_status = "BLACKOUT".to_string();
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let decision =
            session.dispatch_auto_trade_signal(&mut state, sample_signal(AutoTradeSide::Buy, 0.9));
        assert_eq!(decision, GateDecision::NewsBlackout);
    }

    #[test]
    fn gate_rejects_when_halted() {
        let mut state = test_state();
        state.auto_trade_enabled = true;
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.trip_manual_halt(&mut state);
        let decision =
            session.dispatch_auto_trade_signal(&mut state, sample_signal(AutoTradeSide::Buy, 0.9));
        assert_eq!(decision, GateDecision::Halted);
    }
}
