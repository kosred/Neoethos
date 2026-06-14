//! Offline replay harness — Phase 1's proof that the loop works end-to-end with
//! ZERO broker calls. Feed a chronological slice of historical `LiveBar`s and
//! the engine runs exactly as it will live (design §4 Phase 1: "a pure dry-run,
//! not a parallel paper product"). This is a developer/CI seam, NOT a user-
//! facing mode.

use crate::contracts::{ExecutionAdapter, RiskGate, SignalEngine};
use crate::engine::{AutonomousEngine, EngineStats};

/// Run `bars` (assumed already in ascending timestamp order) through `engine`
/// and return the resulting stats snapshot.
pub fn replay<S, R, E>(engine: &mut AutonomousEngine<S, R, E>, bars: &[crate::contracts::LiveBar]) -> EngineStats
where
    S: SignalEngine,
    R: RiskGate,
    E: ExecutionAdapter,
{
    for bar in bars {
        engine.on_bar(bar);
    }
    engine.stats()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{LiveBar, PortfolioEntry, StrategySource, TradeMode};
    use crate::decision::{DecisionConfig, DecisionEngine};
    use crate::engine::EngineConfig;
    use crate::execution::MockExecutionAdapter;
    use crate::portfolio::PortfolioRegistry;
    use crate::risk::{MaxOpenPositionsGate, PermissiveRiskGate};
    use crate::signal::MomentumStubSignal;

    fn entry(symbol: &str, base_tf: &str) -> PortfolioEntry {
        PortfolioEntry {
            symbol: symbol.to_string(),
            base_tf: base_tf.to_string(),
            higher_tfs: vec!["H1".to_string()],
            source: StrategySource::Gene { id: "stub".to_string() },
            mode: TradeMode::PropFirm,
        }
    }

    fn bar(symbol: &str, tf: &str, ts: i64, o: f64, h: f64, l: f64, c: f64) -> LiveBar {
        LiveBar {
            symbol: symbol.to_string(),
            tf: tf.to_string(),
            o,
            h,
            l,
            c,
            volume: 1.0,
            ts,
        }
    }

    /// A deterministic up-then-down price path on the base TF: rising long enough
    /// to trigger a Long, then falling to flip Short — exercising open + reversal
    /// + (likely) an SL/TP exit.
    fn ramp_series(symbol: &str, tf: &str) -> Vec<LiveBar> {
        let mut bars = Vec::new();
        let mut ts = 0i64;
        // Rising leg: 1.0000 → 1.0100
        let mut price = 1.0000;
        for _ in 0..20 {
            let next = price + 0.0005;
            bars.push(bar(symbol, tf, ts, price, next + 0.0002, price - 0.0002, next));
            price = next;
            ts += 60_000;
        }
        // Falling leg: back down past the entry to force a reversal/stop.
        for _ in 0..20 {
            let next = price - 0.0005;
            bars.push(bar(symbol, tf, ts, price, price + 0.0002, next - 0.0002, next));
            price = next;
            ts += 60_000;
        }
        bars
    }

    #[test]
    fn replay_runs_loop_end_to_end_with_zero_broker_calls() {
        let registry = PortfolioRegistry::from_entries(vec![entry("EURUSD", "M1")]);
        let mut engine = AutonomousEngine::new(
            registry,
            MomentumStubSignal::new(3),
            PermissiveRiskGate,
            MockExecutionAdapter::new(),
            DecisionEngine::new(DecisionConfig::default()),
            EngineConfig::default(),
        );

        let bars = ramp_series("EURUSD", "M1");
        let stats = replay(&mut engine, &bars);

        // Every bar processed; the momentum stub fired on base-TF bars.
        assert_eq!(stats.bars_processed, bars.len());
        assert!(stats.signals_evaluated > 0, "stub signal should evaluate on base-TF bars");
        // The up-then-down ramp must have opened at least one position and then
        // closed it (reversal or SL/TP) — proving the full open→manage→close path.
        assert!(stats.positions_opened > 0, "expected at least one open");
        assert!(stats.positions_closed > 0, "expected at least one close");
        // Mock adapter recorded fills (and made ZERO real broker calls — it
        // cannot; it has no transport).
        assert!(engine.execution().fill_count() > 0, "mock should record fills");
        // Equity bookkeeping stayed finite.
        assert!(stats.equity.is_finite());
    }

    #[test]
    fn non_base_tf_bars_do_not_trigger_signals() {
        // Registry watches EURUSD M1; feed only H4 bars → no signal evaluation,
        // no positions, but the bars are still ingested (for the higher-TF cube
        // later). Proves the base-TF gate in the loop.
        let registry = PortfolioRegistry::from_entries(vec![entry("EURUSD", "M1")]);
        let mut engine = AutonomousEngine::new(
            registry,
            MomentumStubSignal::new(2),
            PermissiveRiskGate,
            MockExecutionAdapter::new(),
            DecisionEngine::default(),
            EngineConfig::default(),
        );

        let bars: Vec<LiveBar> = (0..10)
            .map(|i| bar("EURUSD", "H4", i * 14_400_000, 1.0, 1.001, 0.999, 1.0005))
            .collect();
        let stats = replay(&mut engine, &bars);

        assert_eq!(stats.bars_processed, 10);
        assert_eq!(stats.signals_evaluated, 0, "H4 is not the base TF → no signal");
        assert_eq!(stats.positions_opened, 0);
    }

    #[test]
    fn risk_gate_blocks_opens_over_the_cap() {
        // Two symbols, both base M1, both ramping up so each wants to open. With
        // a 1-position cap, the second symbol's open must be blocked.
        let registry = PortfolioRegistry::from_entries(vec![
            entry("EURUSD", "M1"),
            entry("GBPUSD", "M1"),
        ]);
        let mut engine = AutonomousEngine::new(
            registry,
            MomentumStubSignal::new(2),
            MaxOpenPositionsGate::new(1),
            MockExecutionAdapter::new(),
            DecisionEngine::default(),
            EngineConfig::default(),
        );

        // Interleave two rising series so both fire opens on the same window.
        let mut bars = Vec::new();
        let mut price = 1.0;
        for i in 0..12i64 {
            let next = price + 0.001;
            bars.push(bar("EURUSD", "M1", i * 60_000, price, next + 0.0002, price, next));
            bars.push(bar("GBPUSD", "M1", i * 60_000, price, next + 0.0002, price, next));
            price = next;
        }
        let stats = replay(&mut engine, &bars);

        assert!(stats.intents_blocked > 0, "the 1-position cap must block the 2nd symbol's open");
        assert!(stats.open_positions <= 1, "cap must hold open positions at 1");
    }
}
