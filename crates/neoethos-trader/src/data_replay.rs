//! Load real on-disk `.vortex` history and dry-run it through the Phase-1 engine.
//!
//! This is the single helper that makes the offline replay REACHABLE from both
//! front-ends: `neoethos-cli trader-replay` and the app `POST /autonomous/replay`
//! both call [`replay_symbol_from_dir`], so they produce byte-identical
//! [`EngineStats`] from the same data — the UI↔CLI parity mandate, applied to the
//! trader from day one. ZERO broker calls (mock execution), real bars in.

use std::path::Path;

use crate::contracts::{LiveBar, PortfolioEntry, StrategySource, TradeMode};
use crate::decision::DecisionEngine;
use crate::engine::{AutonomousEngine, EngineConfig, EngineStats};
use crate::execution::MockExecutionAdapter;
use crate::portfolio::PortfolioRegistry;
use crate::risk::PermissiveRiskGate;
use crate::signal::MomentumStubSignal;

/// Load `(symbol, base_tf)` OHLCV from the data directory and map each bar to a
/// [`LiveBar`]. Bars come back in ascending-timestamp order (the loader
/// normalises that). Errors if the timeframe isn't present on disk.
pub fn load_bars_from_dir(
    data_dir: impl AsRef<Path>,
    symbol: &str,
    base_tf: &str,
) -> anyhow::Result<Vec<LiveBar>> {
    let ohlcv = neoethos_data::load_symbol_timeframe(data_dir, symbol, base_tf)?;
    let n = ohlcv.len();
    let mut bars = Vec::with_capacity(n);
    for i in 0..n {
        bars.push(LiveBar {
            symbol: symbol.to_string(),
            tf: base_tf.to_string(),
            o: ohlcv.open[i],
            h: ohlcv.high[i],
            l: ohlcv.low[i],
            c: ohlcv.close[i],
            volume: ohlcv.volume.as_ref().map(|v| v[i]).unwrap_or(0.0),
            ts: ohlcv.timestamp.as_ref().map(|v| v[i]).unwrap_or(0),
        });
    }
    Ok(bars)
}

/// Offline dry-run of `(symbol, base_tf)` real history through the Phase-1 engine
/// (momentum stub signal + permissive risk gate + mock execution). Returns the
/// resulting [`EngineStats`].
///
/// Phase 1.5 wires only the base timeframe; the higher-TF cube + the real Gene /
/// ensemble signal arrive in Phases 3–4 (the registry entry already carries the
/// `higher_tfs` slot for when they do).
pub fn replay_symbol_from_dir(
    data_dir: impl AsRef<Path>,
    symbol: &str,
    base_tf: &str,
    cfg: EngineConfig,
) -> anyhow::Result<EngineStats> {
    let bars = load_bars_from_dir(&data_dir, symbol, base_tf)?;
    if bars.is_empty() {
        anyhow::bail!(
            "no bars loaded for {symbol} {base_tf} — is the data folder populated for this pair/timeframe?"
        );
    }

    let registry = PortfolioRegistry::from_entries(vec![PortfolioEntry {
        symbol: symbol.to_string(),
        base_tf: base_tf.to_string(),
        higher_tfs: Vec::new(),
        source: StrategySource::Gene {
            id: format!("{symbol}-{base_tf}-stub"),
        },
        mode: TradeMode::PropFirm,
    }]);

    let mut engine = AutonomousEngine::new(
        registry,
        MomentumStubSignal::default(),
        PermissiveRiskGate,
        MockExecutionAdapter::new(),
        DecisionEngine::default(),
        cfg,
    );

    Ok(crate::replay::replay(&mut engine, &bars))
}
