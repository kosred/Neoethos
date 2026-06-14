//! Shared real-data test fixtures for the workspace.
//!
//! GROUP F remediation (operator directive 2026-05-25 "απαγορευονται
//! παντου συνθετικα δεδομενα"): replaces ~19 hand-rolled synthetic
//! OHLCV/feature generators scattered across the test code with a
//! single canonical fixture seeded by REAL cTrader historical data.
//!
//! ## Why this lives in `neoethos-data`
//!
//! `Ohlcv` and `FeatureFrame` are owned by `neoethos-data`. A `test_fixtures`
//! sub-module here is the natural home for canonical sample data, and it
//! keeps the workspace from sprouting yet another tiny crate.
//!
//! ## Access pattern
//!
//! The module is **always compiled** (no `#[cfg(test)]` gate) so non-test
//! callers — the `--api-test` smoke harness, the wizard's
//! "load demo data" button, the operator's first-time-run-without-data
//! recovery path — can pull the same sample as the unit tests do. This
//! keeps the test-vs-production drift surface zero by construction.
//!
//! ## Fixture source
//!
//! The seed JSON lives at `crates/neoethos-data/test_fixtures/eurusd_m1_100bars.json`
//! and was generated from a real cTrader Open API
//! `ProtoOAGetTrendbarsReq` response for EURUSD M1 (the operator's
//! preferred default pair) on the most recent week available at
//! capture time. To refresh:
//!
//! 1. `neoethos-cli capture-fixture --symbol EURUSD --timeframe M1 --bars 100`
//! 2. Replace `eurusd_m1_100bars.json` with the new capture.
//! 3. Re-run `cargo test --workspace test_fixtures` — the round-trip
//!    self-check tests below will reject malformed input.
//!
//! Until that CLI subcommand lands, the fixture is hand-curated from a
//! prior cTrader capture and ships in the repo. The 100-bar window is
//! enough for the warm-up of every existing indicator (longest is the
//! Hurst-100 window used by the feature builder).

use crate::{FeatureFrame, Ohlcv};
use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Embedded JSON dump of the canonical EURUSD M1 sample. The path
/// is relative to this source file via `include_str!` so the data
/// ships in every build artifact — no filesystem dependency at
/// runtime.
const EURUSD_M1_100BARS_JSON: &str = include_str!("../test_fixtures/eurusd_m1_100bars.json");

/// Wire-shape mirror of the captured cTrader bars payload. One row
/// per bar; matches the ProtoOATrendbar fields we care about for
/// OHLCV reconstruction. Timestamps are Unix-ms UTC (the canonical
/// workspace convention — see `neoethos_core::utils::clock`).
#[derive(Debug, Clone, Deserialize, Serialize)]
struct CTraderBarRow {
    /// Bar-close Unix ms (UTC).
    t: i64,
    /// Open price.
    o: f64,
    /// High price.
    h: f64,
    /// Low price.
    l: f64,
    /// Close price.
    c: f64,
    /// Volume (broker units; for FX this is tick count, not lots).
    #[serde(default)]
    v: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CTraderBarsFixture {
    /// Symbol the bars are for (e.g. "EURUSD"). Surfaced via
    /// [`ctrader_sample_symbol`].
    symbol: String,
    /// Timeframe label (e.g. "M1"). Surfaced via
    /// [`ctrader_sample_timeframe`].
    timeframe: String,
    /// The bars themselves. At least 100 rows for the fixture to
    /// satisfy the longest indicator warm-up (Hurst-100).
    bars: Vec<CTraderBarRow>,
}

fn parse_fixture() -> Result<CTraderBarsFixture> {
    serde_json::from_str(EURUSD_M1_100BARS_JSON)
        .context("parse embedded eurusd_m1_100bars.json fixture")
}

/// Symbol the canonical fixture is for. Always `"EURUSD"`.
pub fn ctrader_sample_symbol() -> &'static str {
    "EURUSD"
}

/// Timeframe of the canonical fixture. Always `"M1"`.
pub fn ctrader_sample_timeframe() -> &'static str {
    "M1"
}

/// Return the canonical real-data OHLCV sample as an
/// [`Ohlcv`]. Suitable for any test that previously hand-rolled a
/// 5-10 bar synthetic ramp.
///
/// Panics if the embedded JSON is corrupt — that would be a build
/// error worth catching loudly (the fixture lives in git, so this
/// can only fail during repo refresh).
pub fn ctrader_sample_ohlcv() -> Ohlcv {
    let fixture = parse_fixture().expect("embedded EURUSD M1 fixture must parse");
    let n = fixture.bars.len();
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    for row in fixture.bars {
        timestamps.push(row.t);
        open.push(row.o);
        high.push(row.h);
        low.push(row.l);
        close.push(row.c);
        volume.push(row.v);
    }
    Ohlcv {
        timestamp: Some(timestamps),
        open,
        high,
        low,
        close,
        volume: Some(volume),
    }
}

/// Return a small canonical [`FeatureFrame`] derived from the
/// OHLCV sample. Two synthetic-but-shape-faithful columns:
///
/// - `close_minus_open` — bar body sign, useful as a directional
///   sentinel in tests that don't run the full HPC feature builder
/// - `range_pips` — `(high − low) * 1e4`, a per-bar volatility proxy
///
/// Tests that need the full ~60-column HPC feature surface should
/// pull `Ohlcv` from [`ctrader_sample_ohlcv`] and run
/// `compute_hpc_feature_frame` themselves. This helper is for the
/// minimal-surface tests that previously hand-rolled a 1-3 column
/// FeatureFrame from scratch.
pub fn ctrader_sample_feature_frame() -> FeatureFrame {
    let ohlcv = ctrader_sample_ohlcv();
    let n = ohlcv.close.len();
    let mut data = Array2::<f32>::zeros((n, 2));
    for i in 0..n {
        data[(i, 0)] = (ohlcv.close[i] - ohlcv.open[i]) as f32;
        data[(i, 1)] = ((ohlcv.high[i] - ohlcv.low[i]) * 1e4) as f32;
    }
    FeatureFrame {
        timestamps: ohlcv.timestamp.clone().unwrap_or_default(),
        names: vec!["close_minus_open".to_string(), "range_pips".to_string()],
        data: crate::FeatureData::InMemory(data),
    }
}

/// Convenience helper for tests that want just the first `n` bars.
/// Saturates at the fixture's actual length so callers don't have
/// to check.
pub fn ctrader_sample_ohlcv_first(n: usize) -> Ohlcv {
    let full = ctrader_sample_ohlcv();
    let count = n.min(full.close.len());
    Ohlcv {
        timestamp: full
            .timestamp
            .as_ref()
            .map(|ts| ts[..count].to_vec()),
        open: full.open[..count].to_vec(),
        high: full.high[..count].to_vec(),
        low: full.low[..count].to_vec(),
        close: full.close[..count].to_vec(),
        volume: full
            .volume
            .as_ref()
            .map(|v| v[..count].to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_parses_and_has_minimum_bars() {
        let ohlcv = ctrader_sample_ohlcv();
        // Longest indicator warm-up in the workspace is Hurst at 100;
        // the fixture must have >= 100 bars to satisfy every caller.
        assert!(
            ohlcv.close.len() >= 100,
            "EURUSD M1 fixture must have >= 100 bars (got {})",
            ohlcv.close.len()
        );
    }

    #[test]
    fn fixture_ohlcv_invariants_hold() {
        let ohlcv = ctrader_sample_ohlcv();
        assert_eq!(ohlcv.open.len(), ohlcv.close.len());
        assert_eq!(ohlcv.high.len(), ohlcv.close.len());
        assert_eq!(ohlcv.low.len(), ohlcv.close.len());
        let ts = ohlcv
            .timestamp
            .as_ref()
            .expect("fixture must carry timestamps");
        assert_eq!(ts.len(), ohlcv.close.len());
        for i in 0..ohlcv.close.len() {
            // High >= max(open, close, low), Low <= min(...)
            let max_oc = ohlcv.open[i].max(ohlcv.close[i]);
            let min_oc = ohlcv.open[i].min(ohlcv.close[i]);
            assert!(
                ohlcv.high[i] >= max_oc.max(ohlcv.low[i]) - 1e-9,
                "bar {i}: high {} must be >= max(open, close, low)",
                ohlcv.high[i]
            );
            assert!(
                ohlcv.low[i] <= min_oc.min(ohlcv.high[i]) + 1e-9,
                "bar {i}: low {} must be <= min(open, close, high)",
                ohlcv.low[i]
            );
        }
        // Timestamps strictly monotonic.
        for i in 1..ts.len() {
            assert!(
                ts[i] > ts[i - 1],
                "timestamps must be strictly monotonic at index {i}: {} <= {}",
                ts[i],
                ts[i - 1]
            );
        }
    }

    #[test]
    fn fixture_first_n_truncates() {
        let small = ctrader_sample_ohlcv_first(10);
        assert_eq!(small.close.len(), 10);
        let huge = ctrader_sample_ohlcv_first(10_000);
        assert!(huge.close.len() < 10_000); // saturates at fixture size
    }

    #[test]
    fn feature_frame_shape_matches_ohlcv() {
        let frame = ctrader_sample_feature_frame();
        let ohlcv = ctrader_sample_ohlcv();
        assert_eq!(frame.n_samples(), ohlcv.close.len());
        assert_eq!(frame.n_features(), 2);
        assert_eq!(frame.names.len(), 2);
        assert_eq!(frame.timestamps.len(), ohlcv.close.len());
    }
}
