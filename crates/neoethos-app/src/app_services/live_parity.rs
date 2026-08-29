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
//! Method: fetch ONE consistent set of recent broker bars per timeframe
//! (`reference_bars`, e.g. 3000), run the exact live pipeline twice —
//!   (a) reference: on the full fetched history,
//!   (b) window:    on the truncated tail (`window_bars`, the live default),
//! then compare the last `compare_tail` bars' directions + gene SL/TP
//! bar-for-bar (aligned by timestamp). Zero direction mismatches = PASS.
//!
//! Blocking (broker fetch + feature computation) — call via spawn_blocking.

use anyhow::{Context, Result};
use serde::Serialize;

use crate::app_services::broker_api::{
    fetch_broker_symbols_blocking, fetch_recent_chart_bars_blocking,
};
use crate::app_services::live_trading::bars_to_ohlcv;

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
    pub verdict: String, // "PASS" | "FAIL"
    pub note: String,
}

/// Run the live pipeline (features → projection → gene combine) over `ohlcv`
/// frames and return per-bar (timestamp, direction, sl, tp) for the base TF.
fn signals_for_frames(
    artifact: &neoethos_search::LivePortfolioArtifact,
    frames: std::collections::HashMap<String, neoethos_data::Ohlcv>,
    pip_size: f64,
) -> Result<Vec<(i64, String, f64, f64)>> {
    let base_ohlcv = frames
        .get(&artifact.base_tf)
        .cloned()
        .context("base timeframe missing from frames")?;
    let dataset = neoethos_data::SymbolDataset {
        symbol: artifact.symbol.clone(),
        frames,
        // Broker-fetched comparison bars have no immutable generation receipt.
        // The canonical feature builder must fail closed instead of treating
        // this transient window as historical authority.
        source_artifacts: std::collections::HashMap::new(),
    };
    let higher_refs: Vec<&str> = artifact.higher_tfs.iter().map(|s| s.as_str()).collect();
    let raw =
        neoethos_data::prepare_multitimeframe_features(&dataset, &artifact.base_tf, &higher_refs)
            .context("feature computation")?;
    let aligned =
        neoethos_search::project_features_to_effective(&raw, &artifact.effective_feature_names)
            .context("feature projection (effective_names mismatch?)")?;
    let (dirs, sls, tps) = neoethos_trader::combine_gene_signals_with_brackets(
        &artifact.genes,
        &aligned,
        &base_ohlcv,
        pip_size,
    )
    .context("synthesize parity gene signals from the validated feature frame")?;
    let ts = base_ohlcv.timestamp.clone().unwrap_or_default();
    // Feature rows align to the TAIL of the ohlcv (warmup rows dropped) — pair
    // each signal with its bar timestamp from the tail.
    let offset = ts.len().saturating_sub(dirs.len());
    Ok(dirs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            (
                ts.get(offset + i).copied().unwrap_or(0),
                format!("{d:?}"),
                sls.get(i).copied().unwrap_or(0.0),
                tps.get(i).copied().unwrap_or(0.0),
            )
        })
        .collect())
}

/// Truncate every frame to its last `n` bars (exactly what live fetching
/// `warmup_bars = n` per timeframe produces).
fn tail_frames(
    frames: &std::collections::HashMap<String, neoethos_data::Ohlcv>,
    n: usize,
) -> std::collections::HashMap<String, neoethos_data::Ohlcv> {
    frames
        .iter()
        .map(|(tf, o)| {
            let len = o.close.len();
            let start = len.saturating_sub(n);
            let slice = neoethos_data::Ohlcv {
                timestamp: o.timestamp.as_ref().map(|t| t[start..].to_vec()),
                open: o.open[start..].to_vec(),
                high: o.high[start..].to_vec(),
                low: o.low[start..].to_vec(),
                close: o.close[start..].to_vec(),
                volume: o.volume.as_ref().map(|v| v[start..].to_vec()),
            };
            (tf.clone(), slice)
        })
        .collect()
}

/// BLOCKING. Fetches broker bars and compares live-window signals against the
/// long-history reference for `portfolio_path`. See the module docs.
pub fn run_live_parity_check(
    portfolio_path: &str,
    window_bars: usize,
    reference_bars: usize,
) -> Result<LiveParityReport> {
    let broker_truth = neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::LiveTrading)
        .map_err(anyhow::Error::new)?;

    let artifact = neoethos_search::load_live_portfolio_json(portfolio_path)
        .with_context(|| format!("load live portfolio {portfolio_path}"))?;
    if artifact.genes.is_empty() {
        anyhow::bail!("portfolio '{portfolio_path}' has no genes");
    }
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
    let pip_size = broker_truth
        .exact_pip_size_v1(&artifact.symbol)
        .map_err(anyhow::Error::new)?;
    let window = window_bars.clamp(200, 5000);
    let reference = reference_bars.clamp(window + 500, 10_000);

    // ONE consistent fetch per timeframe (reference size); the window run
    // truncates the same data, so any difference is pipeline-induced.
    let mut frames = std::collections::HashMap::new();
    for tf in std::iter::once(&artifact.base_tf).chain(artifact.higher_tfs.iter()) {
        let bars = fetch_recent_chart_bars_blocking(&artifact.symbol, tf, reference)
            .with_context(|| format!("fetch {reference} recent bars for {tf}"))?;
        frames.insert(tf.clone(), bars_to_ohlcv(&bars));
    }

    let reference_signals = signals_for_frames(&artifact, frames.clone(), pip_size)?;
    let window_signals = signals_for_frames(&artifact, tail_frames(&frames, window), pip_size)?;

    // Compare the freshest bars both runs cover — index by timestamp.
    let compare_tail = (window / 4).clamp(50, 400);
    let ref_by_ts: std::collections::HashMap<i64, &(i64, String, f64, f64)> =
        reference_signals.iter().map(|r| (r.0, r)).collect();

    let mut compared = 0usize;
    let mut mismatches = 0usize;
    let mut samples: Vec<ParityMismatchRow> = Vec::new();
    let mut max_sl_delta = 0.0f64;
    let mut max_tp_delta = 0.0f64;

    for w in window_signals.iter().rev().take(compare_tail) {
        let Some(r) = ref_by_ts.get(&w.0) else {
            continue;
        };
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
            max_sl_delta = max_sl_delta.max((r.2 - w.2).abs());
            max_tp_delta = max_tp_delta.max((r.3 - w.3).abs());
        }
    }

    let pass = mismatches == 0 && compared > 0;
    let note = if compared == 0 {
        "no overlapping bars compared — check data availability".to_string()
    } else if pass {
        format!(
            "live {window}-bar window reproduces the {reference}-bar reference exactly \
             on the last {compared} bars — live signals are window-invariant for this portfolio"
        )
    } else {
        format!(
            "{mismatches}/{compared} bars DIFFER between the live window and the reference — \
             live signals for this portfolio depend on history length (warmup-sensitive \
             features). Live trading will NOT match the validated backtest; increase \
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
        verdict: if pass { "PASS".into() } else { "FAIL".into() },
        note,
    })
}
