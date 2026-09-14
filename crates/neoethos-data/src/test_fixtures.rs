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
//! The module is **always compiled** (no `#[cfg(test)]` gate) because integration
//! tests in other workspace crates consume it as an ordinary dependency. It is
//! test support only; no operator or production-data path loads this fixture.
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

use crate::{FeatureCellValidity, FeatureColumnF64, FeatureFrame, Ohlcv};
use anyhow::{Context, Result};
use ndarray::Array2;
use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOutputV1, FeaturePlanV1,
    SourceArtifactBindingV1, SourceSegmentV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    /// Bar-open Unix ms (UTC). cTrader defines `utcTimestampInMinutes` as
    /// the timestamp of the bar's opening tick.
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
    let mut close_minus_open = Vec::with_capacity(n);
    let mut range_pips = Vec::with_capacity(n);
    for i in 0..n {
        close_minus_open.push(ohlcv.close[i] - ohlcv.open[i]);
        range_pips.push((ohlcv.high[i] - ohlcv.low[i]) * 1e4);
    }
    let columns = vec![
        FeatureColumnF64::new(
            "close_minus_open",
            close_minus_open,
            vec![FeatureCellValidity::Valid; n],
        )
        .expect("fixture body column"),
        FeatureColumnF64::new(
            "range_pips",
            range_pips,
            vec![FeatureCellValidity::Valid; n],
        )
        .expect("fixture range column"),
    ];
    ctrader_test_feature_frame_from_columns(
        ohlcv
            .timestamp
            .clone()
            .expect("embedded fixture has timestamps"),
        columns,
    )
    .expect("embedded f64 feature frame")
}

/// Build an f64/validity-aware feature frame for adversarial search tests while
/// retaining the same typed plan and dataset provenance as production frames.
///
/// This deliberately lives under `test_fixtures`: it is not a compatibility
/// replacement for the removed `FeatureFrame::from_array` runtime API. Callers
/// must provide canonical Unix-millisecond timestamps and explicit validity.
pub fn ctrader_test_feature_frame_from_columns(
    timestamps: Vec<i64>,
    columns: Vec<FeatureColumnF64>,
) -> Result<FeatureFrame> {
    anyhow::ensure!(
        !timestamps.is_empty(),
        "test feature timestamps must not be empty"
    );
    anyhow::ensure!(
        !columns.is_empty(),
        "test feature frame must contain columns"
    );
    for column in &columns {
        anyhow::ensure!(
            column.len() == timestamps.len(),
            "test feature column `{}` has {} rows; timestamps have {}",
            column.name,
            column.len(),
            timestamps.len()
        );
    }

    let identity = CanonicalDatasetIdentity::external(
        "embedded-ctrader-fixture-unverified",
        ctrader_sample_symbol(),
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("fixture dataset identity");
    let fixture_hash: [u8; 32] = Sha256::digest(EURUSD_M1_100BARS_JSON.as_bytes()).into();
    let source_node_id = "source:embedded-ctrader-fixture";
    let outputs = columns
        .iter()
        .map(|column| FeatureOutputV1::f64(column.name.clone(), 1))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let source = FeatureNodeV1::source(
        source_node_id,
        identity.clone(),
        "neoethos.test-fixture-derived-features.f64.v1",
        1,
        outputs,
        fixture_hash,
    )?;
    let names = columns
        .iter()
        .map(|column| column.name.clone())
        .collect::<Vec<_>>();
    let plan = FeaturePlanV1::new(vec![source], names)?;
    let provenance = DatasetFeatureArtifactProvenanceV1::new(
        &plan,
        vec![SourceArtifactBindingV1::new(
            source_node_id,
            identity,
            "neoethos.embedded-test-fixture.v1",
            fixture_hash,
            "embedded-fixture-v1",
            fixture_hash,
            BarTimestampConvention::BarOpen,
            vec![SourceSegmentV1::new(
                0,
                timestamps.len() as u64,
                timestamps[0],
                timestamps[timestamps.len() - 1],
            )?],
        )?],
    )?;
    FeatureFrame::from_columns(timestamps, columns, plan, provenance)
}

/// Explicit fixture seam for testing persisted preprocessing across crate
/// boundaries. This is not market-data publication or trading evidence.
/// `None` fits the supplied training range; `Some` replays that saved state on
/// these fixture rows without inspecting them to estimate new parameters.
pub fn ctrader_test_feature_frame_with_normalization(
    raw: &FeatureFrame,
    training_rows: std::ops::Range<usize>,
    fitted: Option<&crate::SearchNormalizationFittedStateV1>,
) -> Result<FeatureFrame> {
    anyhow::ensure!(
        raw.normalization_fitted_state().is_none(),
        "fixture input must be raw"
    );
    anyhow::ensure!(
        raw.provenance().bindings().len() == 1,
        "single-source fixture required"
    );
    let columns = raw
        .names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let cells = (0..raw.n_samples())
                .map(|row| raw.cell(row, index))
                .collect::<Result<Vec<_>>>()?;
            FeatureColumnF64::new(
                name.clone(),
                cells.iter().map(|cell| cell.value).collect(),
                cells.iter().map(|cell| cell.validity).collect(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    normalized_fixture_from_columns(
        raw.timestamps.clone(),
        columns,
        crate::FeatureBuildOptions {
            normalization_training_rows: Some(training_rows),
            ..Default::default()
        },
        fitted,
    )
}

/// Raw, explicitly unverified fixture with a recorded producer recipe. This
/// preserves real raw backing for cross-crate lazy-fit/prefix tests; it grants
/// no canonical publication, candidate or trading authority.
pub fn ctrader_test_feature_frame_from_columns_with_options(
    timestamps: Vec<i64>,
    columns: Vec<FeatureColumnF64>,
    options: crate::FeatureBuildOptions,
) -> Result<FeatureFrame> {
    Ok(
        ctrader_test_feature_frame_from_columns(timestamps, columns)?
            .with_feature_build_options(options),
    )
}

/// Bounded integration fixture for persisted CPU normalization. Uses the real
/// train-only fitter and its exact fitted-state node; the source remains the
/// explicitly unverified embedded fixture, never a production dataset claim.
/// Supplied column names are already the producer names (including prefixes).
pub fn ctrader_test_normalized_feature_frame_from_columns(
    timestamps: Vec<i64>,
    columns: Vec<FeatureColumnF64>,
    options: crate::FeatureBuildOptions,
) -> Result<FeatureFrame> {
    normalized_fixture_from_columns(timestamps, columns, options, None)
}

fn normalized_fixture_from_columns(
    timestamps: Vec<i64>,
    mut columns: Vec<FeatureColumnF64>,
    options: crate::FeatureBuildOptions,
    fitted: Option<&crate::SearchNormalizationFittedStateV1>,
) -> Result<FeatureFrame> {
    if let Some(state) = fitted {
        anyhow::ensure!(
            Some(state.training_rows()?) == options.normalization_training_rows,
            "fixture fit scope mismatch"
        );
    }
    let mut raw_columns = columns.clone();
    for column in &mut raw_columns {
        column.name = format!("pre-normalize:0:{}", column.name);
    }
    let raw = ctrader_test_feature_frame_from_columns(timestamps.clone(), raw_columns)?;
    let (fits, _) = crate::prepare_multitimeframe_feature_columns(
        &mut columns,
        true,
        options.normalization_training_rows.clone(),
        options.drop_columns_without_normalization_training_support,
        1,
        &crate::FeatureBuildControl::default(),
        fitted,
    )?;
    let names = columns
        .iter()
        .map(|column| column.name.clone())
        .collect::<Vec<_>>();
    let state = crate::SearchNormalizationFittedStateV1::new(names.clone(), fits)?;
    if let Some(expected) = fitted {
        anyhow::ensure!(&state == expected, "fixture changed the saved fit");
    }
    let mut nodes = raw.plan().nodes().to_vec();
    let normalization_hash =
        crate::semantic_source_hash(&[include_bytes!("core/normalization.rs")]);
    nodes.push(FeatureNodeV1::transform(
        "normalization:robust-f64",
        neoethos_feature_contracts::FeatureOperationTagV1::Normalization,
        crate::SEARCH_NORMALIZATION_POLICY_VERSION,
        nodes.iter().map(|node| node.id().to_owned()).collect(),
        names
            .iter()
            .map(|name| FeatureOutputV1::f64(name, crate::SEARCH_NORMALIZATION_POLICY_VERSION))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        Vec::new(),
        normalization_hash,
        normalization_hash,
        Some(state.fitted_state_hash()?),
    )?);
    let plan = FeaturePlanV1::new(nodes, names)?;
    let provenance =
        DatasetFeatureArtifactProvenanceV1::new(&plan, raw.provenance().bindings().to_vec())?;
    FeatureFrame::from_columns(timestamps, columns, plan, provenance)?
        .with_feature_build_options(options)
        .with_normalization_fitted_state(state)
}

/// Convenience adapter for legacy test matrices. The matrix is f64-only and
/// every non-finite cell becomes explicitly invalid; it cannot reintroduce the
/// removed f32 production contract.
pub fn ctrader_test_feature_frame_from_matrix(
    timestamps: Vec<i64>,
    names: Vec<String>,
    matrix: Array2<f64>,
) -> Result<FeatureFrame> {
    anyhow::ensure!(
        matrix.nrows() == timestamps.len(),
        "test feature matrix has {} rows; timestamps have {}",
        matrix.nrows(),
        timestamps.len()
    );
    anyhow::ensure!(
        matrix.ncols() == names.len(),
        "test feature matrix has {} columns; names have {}",
        matrix.ncols(),
        names.len()
    );
    let columns = names
        .into_iter()
        .enumerate()
        .map(|(column_index, name)| {
            let values = matrix.column(column_index).to_vec();
            let validity = values
                .iter()
                .map(|value| {
                    if value.is_finite() {
                        FeatureCellValidity::Valid
                    } else {
                        FeatureCellValidity::NonFinite
                    }
                })
                .collect();
            FeatureColumnF64::new(name, values, validity)
        })
        .collect::<Result<Vec<_>>>()?;
    ctrader_test_feature_frame_from_columns(timestamps, columns)
}

/// Produce a canonical M1 Unix-millisecond grid anchored at the captured
/// cTrader fixture. Tests may vary values and row count without falling back to
/// the removed seconds/nanoseconds inference path.
pub fn canonical_test_timestamps(rows: usize) -> Vec<i64> {
    let start = ctrader_sample_ohlcv()
        .timestamp
        .and_then(|timestamps| timestamps.first().copied())
        .expect("embedded fixture has a first timestamp");
    (0..rows)
        .map(|row| {
            start
                .checked_add((row as i64) * 60_000)
                .expect("test timestamp grid must fit i64")
        })
        .collect()
}

/// Convenience helper for tests that want just the first `n` bars.
/// Saturates at the fixture's actual length so callers don't have
/// to check.
pub fn ctrader_sample_ohlcv_first(n: usize) -> Ohlcv {
    let full = ctrader_sample_ohlcv();
    let count = n.min(full.close.len());
    Ohlcv {
        timestamp: full.timestamp.as_ref().map(|ts| ts[..count].to_vec()),
        open: full.open[..count].to_vec(),
        high: full.high[..count].to_vec(),
        low: full.low[..count].to_vec(),
        close: full.close[..count].to_vec(),
        volume: full.volume.as_ref().map(|v| v[..count].to_vec()),
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
