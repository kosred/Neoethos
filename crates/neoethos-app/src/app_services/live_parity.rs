//! Live↔backtest parity harness — window-invariance check for live signals.
//!
//! The live autopilot recomputes features on a SHORT window (default 1000
//! bars per timeframe) while discovery/backtest computed them on the full
//! history. Any indicator whose value depends on how much history it warmed
//! up on (long EMAs, rolling normalisations, cross-TF alignment edges) makes
//! live signals silently diverge from the validated backtest — the exact
//! class of bug that killed live performance before (missing trailing was
//! execution-level; this harness guards the SIGNAL level).
//!
//! Method: fetch a long and a short broker response per timeframe with the
//! exact same cTrader `toTimestamp`, prove the short response is byte-for-byte
//! the tail of the long response, then run the exact live pipeline twice —
//!   (a) reference: on the full fetched history,
//!   (b) window:    on the truncated tail (`window_bars`, the live default),
//! then compare the last `compare_tail` bars' directions, confidence and gene
//! SL/TP bar-for-bar. PASS requires exact agreement in every compared field.
//!
//! Blocking (broker fetch + feature computation) — call via spawn_blocking.

use anyhow::{Context, Result};
use serde::Serialize;

use crate::app_services::broker_api::{
    CTRADER_RECENT_TRENDBAR_LIMIT, RecentBrokerTrendbarSnapshot,
    fetch_bound_broker_symbol_blocking, fetch_broker_symbols_blocking,
    fetch_recent_broker_trendbar_snapshot_at_blocking,
};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ParityMismatchRow {
    pub bar_ts_ms: i64,
    pub reference: String,
    pub window: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveParityReport {
    pub symbol: String,
    pub base_tf: String,
    pub reference_bars: usize,
    pub window_bars: usize,
    pub compared_bars: usize,
    pub direction_mismatches: usize,
    /// First few mismatching bars (timestamp + both directions) for diagnosis.
    pub mismatch_samples: Vec<ParityMismatchRow>,
    /// Max |Δ| of the gene stop/target (pips) across compared bars where both
    /// runs agree on direction.
    pub max_sl_delta_pips: f64,
    pub max_tp_delta_pips: f64,
    pub max_confidence_delta: f64,
    pub verdict: String, // "PASS" | "FAIL"
    pub note: String,
}

/// Publish exact broker snapshots, run the live pipeline (features → projection
/// → gene combine), and return per-bar (timestamp, direction, sl, tp, confidence).
fn signals_for_snapshots(
    artifact: &neoethos_search::LivePortfolioArtifact,
    snapshots: Vec<RecentBrokerTrendbarSnapshot>,
    pip_size: f64,
) -> Result<Vec<(i64, String, f64, f64, f64)>> {
    let canonical = crate::app_services::live_feature_snapshot::LiveCanonicalFeatureSnapshot::publish_for_artifact(
        artifact,
        snapshots,
    )
    .context("publish canonical live parity inputs")?;
    let base_ohlcv = canonical
        .dataset()
        .frames
        .get(&artifact.base_tf)
        .cloned()
        .context("base timeframe missing from canonical live parity series")?;
    anyhow::ensure!(
        artifact
            .live_trading_policy
            .sealed_evaluation_config()?
            .pip_value
            == pip_size,
        "archived parity pip size differs from the exact broker symbol"
    );
    let raw = artifact
        .prepare_live_features(canonical.dataset())
        .context("prepare exact saved live feature recipe")?;
    let aligned = artifact
        .project_live_features(&raw)
        .context("apply saved Search projection and frozen fit")?;
    let netted = neoethos_trader::combine_gene_signals_with_archived_policy(
        &artifact.genes,
        &aligned,
        &base_ohlcv,
        &artifact.live_trading_policy,
    )
    .context("synthesize parity gene signals from the validated feature frame")?;
    // The same producer already proves exact row/timestamp equality. Never
    // guess a tail offset or synthesize timestamp zero for missing alignment.
    let output = netted
        .directions
        .iter()
        .enumerate()
        .map(|(i, d)| {
            (
                aligned.timestamps[i],
                format!("{d:?}"),
                netted.sl_pips[i],
                netted.tp_pips[i],
                netted.confidences[i],
            )
        })
        .collect();
    drop(aligned);
    drop(raw);
    drop(canonical);
    Ok(output)
}

/// BLOCKING. Fetches broker bars and compares live-window signals against the
/// long-history reference for `portfolio_path`. See the module docs.
pub fn run_live_parity_check(
    portfolio_path: &str,
    window_bars: usize,
    reference_bars: usize,
) -> Result<LiveParityReport> {
    let artifact = neoethos_search::load_live_portfolio_json(portfolio_path)
        .with_context(|| format!("load live portfolio {portfolio_path}"))?;
    if artifact.genes.is_empty() {
        anyhow::bail!("portfolio '{portfolio_path}' has no genes");
    }
    artifact
        .live_trading_policy
        .sealed_evaluation_config()
        .context("parity requires the portfolio's complete archived signal policy")?;
    let broker_symbols = fetch_broker_symbols_blocking()
        .context("resolve active cTrader identity for strict v2 parity check")?;
    let broker_environment = match broker_symbols.environment {
        value if value.eq_ignore_ascii_case("demo") => neoethos_data::CTraderEnvironment::Demo,
        value if value.eq_ignore_ascii_case("live") => neoethos_data::CTraderEnvironment::Live,
        value => anyhow::bail!("active cTrader environment `{value}` is not canonical"),
    };
    let matching_symbols = broker_symbols
        .symbols
        .iter()
        .filter(|candidate| candidate.symbol_name == artifact.symbol)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        matching_symbols.len() == 1,
        "active cTrader account exposes {} exact `{}` symbols; expected one",
        matching_symbols.len(),
        artifact.symbol
    );
    artifact.validate_ctrader_runtime_binding(
        broker_environment,
        broker_symbols.account_id,
        matching_symbols[0].symbol_id,
        &matching_symbols[0].symbol_name,
    )?;
    // Signal diagnostics need exact symbol precision, not permission to trade.
    // Resolve the full broker symbol on the same admitted account/environment;
    // no name-only disk metadata or historical money permit supplies pip size.
    let resolved = fetch_bound_broker_symbol_blocking(
        &artifact.symbol,
        if broker_symbols.environment.eq_ignore_ascii_case("live") {
            crate::app_services::ctrader_live_auth::CTraderEnvironment::Live
        } else {
            crate::app_services::ctrader_live_auth::CTraderEnvironment::Demo
        },
        broker_symbols.account_id,
        matching_symbols[0].symbol_id,
    )?;
    let pip_size = 10.0_f64.powi(-resolved.symbol.pip_position);
    let window = window_bars.clamp(200, CTRADER_RECENT_TRENDBAR_LIMIT - 500);
    let reference = reference_bars.clamp(window + 500, CTRADER_RECENT_TRENDBAR_LIMIT);

    // Both requests use one inclusive broker upper bound. We still verify the
    // short response is exactly the tail of the long response so a correction
    // or inconsistent broker page fails closed instead of becoming a false
    // parity mismatch (or, worse, a false PASS).
    let as_of_ms = chrono::Utc::now().timestamp_millis();
    let mut reference_snapshots = Vec::new();
    let mut window_snapshots = Vec::new();
    for tf in std::iter::once(&artifact.base_tf).chain(artifact.higher_tfs.iter()) {
        let reference_snapshot = fetch_recent_broker_trendbar_snapshot_at_blocking(
            &artifact.symbol,
            tf,
            reference,
            as_of_ms,
        )
        .with_context(|| format!("fetch {reference} recent bars for {tf}"))?;
        let window_snapshot = fetch_recent_broker_trendbar_snapshot_at_blocking(
            &artifact.symbol,
            tf,
            window,
            as_of_ms,
        )
        .with_context(|| format!("fetch {window} recent bars for {tf}"))?;
        anyhow::ensure!(
            reference_snapshot.identity() == window_snapshot.identity(),
            "reference/window cTrader identities disagree for {tf}"
        );
        anyhow::ensure!(
            window_snapshot.bars().len() <= reference_snapshot.bars().len(),
            "window response for {tf} has more rows than its reference"
        );
        let tail_start = reference_snapshot.bars().len() - window_snapshot.bars().len();
        anyhow::ensure!(
            reference_snapshot.bars()[tail_start..] == *window_snapshot.bars(),
            "window response for {tf} is not the exact tail of the fixed-as-of reference response"
        );
        reference_snapshots.push(reference_snapshot);
        window_snapshots.push(window_snapshot);
    }

    let reference_signals = signals_for_snapshots(&artifact, reference_snapshots, pip_size)?;
    let window_signals = signals_for_snapshots(&artifact, window_snapshots, pip_size)?;

    // Compare the freshest bars both runs cover — index by timestamp.
    let compare_tail = (window / 4).clamp(50, 400);
    let ref_by_ts: std::collections::HashMap<i64, &(i64, String, f64, f64, f64)> =
        reference_signals.iter().map(|r| (r.0, r)).collect();

    let mut compared = 0usize;
    let mut mismatches = 0usize;
    let mut samples: Vec<ParityMismatchRow> = Vec::new();
    let mut max_sl_delta = 0.0f64;
    let mut max_tp_delta = 0.0f64;
    let mut max_confidence_delta = 0.0f64;

    for w in window_signals.iter().rev().take(compare_tail) {
        let r = ref_by_ts.get(&w.0).with_context(|| {
            format!(
                "window parity timestamp {} is missing from its reference; refusing a partial PASS",
                w.0
            )
        })?;
        compared += 1;
        if r.1 != w.1 {
            mismatches += 1;
            if samples.len() < 10 {
                samples.push(ParityMismatchRow {
                    bar_ts_ms: w.0,
                    reference: r.1.clone(),
                    window: w.1.clone(),
                });
            }
        } else {
            max_sl_delta = checked_max_delta(max_sl_delta, r.2, w.2, "stop")?;
            max_tp_delta = checked_max_delta(max_tp_delta, r.3, w.3, "target")?;
            max_confidence_delta = checked_max_delta(max_confidence_delta, r.4, w.4, "confidence")?;
        }
    }

    let pass = exact_signal_parity(
        compared,
        mismatches,
        max_sl_delta,
        max_tp_delta,
        max_confidence_delta,
    );
    let note = if compared == 0 {
        "no overlapping bars compared — check data availability".to_string()
    } else if pass {
        format!(
            "live {window}-bar window reproduces the {reference}-bar reference exactly \
             on the last {compared} bars — all tested directions, confidence and brackets agree"
        )
    } else {
        format!(
            "{mismatches}/{compared} directions differ; max SL/TP deltas {max_sl_delta}/{max_tp_delta} pips, confidence delta {max_confidence_delta}. \
             The live window and reference differ — \
             live signals for this portfolio depend on history length (warmup-sensitive \
             features). This window does not establish validated-strategy parity; increase \
             warmup_bars or re-discover with window-stable features."
        )
    };

    Ok(LiveParityReport {
        symbol: artifact.symbol.clone(),
        base_tf: artifact.base_tf.clone(),
        reference_bars: reference,
        window_bars: window,
        compared_bars: compared,
        direction_mismatches: mismatches,
        mismatch_samples: samples,
        max_sl_delta_pips: max_sl_delta,
        max_tp_delta_pips: max_tp_delta,
        max_confidence_delta,
        verdict: if pass { "PASS".into() } else { "FAIL".into() },
        note: format!(
            "{note}. Signal-window diagnostics only: this does not test model inference, broker fills or profitability."
        ),
    })
}

fn checked_max_delta(maximum: f64, reference: f64, window: f64, field: &str) -> Result<f64> {
    let delta = (reference - window).abs();
    anyhow::ensure!(
        maximum.is_finite() && reference.is_finite() && window.is_finite() && delta.is_finite(),
        "non-finite {field} in signal parity; NaN must not disappear through f64::max"
    );
    Ok(maximum.max(delta))
}

fn exact_signal_parity(
    compared: usize,
    direction_mismatches: usize,
    sl_delta: f64,
    tp_delta: f64,
    confidence_delta: f64,
) -> bool {
    compared > 0
        && direction_mismatches == 0
        && [sl_delta, tp_delta, confidence_delta]
            .iter()
            .all(|delta| delta.is_finite() && *delta == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonfinite_signal_differences_cannot_become_zero_parity_error() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(checked_max_delta(0.0, invalid, 1.0, "stop").is_err());
            assert!(checked_max_delta(0.0, 1.0, invalid, "stop").is_err());
            assert!(checked_max_delta(invalid, 1.0, 1.0, "stop").is_err());
        }
        assert!(checked_max_delta(0.0, f64::MAX, -f64::MAX, "stop").is_err());
        assert_eq!(checked_max_delta(0.0, 2.0, 2.0, "stop").unwrap(), 0.0);
        assert_eq!(checked_max_delta(0.5, 2.0, 1.0, "stop").unwrap(), 1.0);
    }

    #[test]
    fn same_direction_does_not_hide_wrong_risk_or_confidence() {
        assert!(exact_signal_parity(100, 0, 0.0, 0.0, 0.0));
        assert!(!exact_signal_parity(0, 0, 0.0, 0.0, 0.0));
        assert!(!exact_signal_parity(100, 1, 0.0, 0.0, 0.0));
        for delta in [f64::EPSILON, 0.01, f64::NAN, f64::INFINITY] {
            assert!(!exact_signal_parity(100, 0, delta, 0.0, 0.0));
            assert!(!exact_signal_parity(100, 0, 0.0, delta, 0.0));
            assert!(!exact_signal_parity(100, 0, 0.0, 0.0, delta));
        }
    }
}
