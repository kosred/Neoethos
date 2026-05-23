//! GET /indicators?symbol=&timeframe=&indicator=&period=&limit=
//!
//! Compute a single technical indicator on the local OHLCV slice
//! and return the series so the Flutter chart can overlay it on
//! the candlestick canvas. Backed by `vector_ta` via the
//! `neoethos_data::compute_single_indicator` helper — no manual
//! indicator math here. Add a new indicator by extending
//! `ALLOWED_INDICATORS` once the upstream id appears in
//! `crates/neoethos-data/src/core/all_indicators.rs::ALL_INDICATORS`.
//!
//! Wire shape — single-output indicator:
//! ```json
//! { "symbol":"EURUSD","timeframe":"M1","indicator":"sma","period":20,
//!   "candleCount":200,
//!   "lines":[{"name":"sma","values":[1.0823,1.0824,…]}] }
//! ```
//! Multi-output (Bollinger Bands, MACD, Stochastic) decomposes into
//! several entries in `lines`, named `<id>_line0`, `<id>_line1`, etc.

use std::collections::HashMap;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_data::{IndicatorLine, compute_single_indicator, load_symbol_dataset};

use super::state::AppApiState;

/// Top-10 indicators we surface on the Chart screen. Adding a new
/// one here also requires the upstream id to appear in
/// `crates/neoethos-data/src/core/all_indicators.rs::ALL_INDICATORS`.
/// Order matters: it drives the order they show up in the UI
/// dropdown.
pub const ALLOWED_INDICATORS: &[&str] = &[
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "stoch",
    "adx",
    "vwap",
];

#[derive(Debug, serde::Deserialize)]
pub struct IndicatorQuery {
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub indicator: Option<String>,
    /// Optional period for indicators that take one
    /// (sma/ema/rsi/atr/adx). Library default when missing.
    pub period: Option<f64>,
    /// Bollinger Bands standard-deviation multiplier. Library
    /// default when missing.
    pub std_dev: Option<f64>,
    /// MACD specifics — caller can omit any of these to use
    /// library defaults (12/26/9).
    pub fast: Option<f64>,
    pub slow: Option<f64>,
    pub signal: Option<f64>,
    /// Stochastic specifics — library defaults are 14/3/3.
    pub k_period: Option<f64>,
    pub k_slow: Option<f64>,
    pub d_period: Option<f64>,
    /// How many trailing candles to return. Mirrors `/chart`.
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndicatorDto {
    pub symbol: String,
    pub timeframe: String,
    pub indicator: String,
    pub candle_count: usize,
    /// One per output series — multi-output indicators decompose.
    pub lines: Vec<IndicatorLineDto>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndicatorLineDto {
    pub name: String,
    pub values: Vec<f64>,
}

const DEFAULT_LIMIT: usize = 200;
const MAX_LIMIT: usize = 2000;

pub async fn indicators(
    State(_state): State<AppApiState>,
    Query(q): Query<IndicatorQuery>,
) -> Response {
    let symbol = q
        .symbol
        .unwrap_or_else(|| "EURUSD".to_string())
        .trim()
        .to_uppercase();
    let timeframe = q
        .timeframe
        .unwrap_or_else(|| "M1".to_string())
        .trim()
        .to_uppercase();
    let indicator = q
        .indicator
        .unwrap_or_else(|| "sma".to_string())
        .trim()
        .to_ascii_lowercase();
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);

    if !ALLOWED_INDICATORS.contains(&indicator.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "indicator '{indicator}' is not in the allowed list; valid: {}",
                    ALLOWED_INDICATORS.join(", ")
                ),
            })),
        )
            .into_response();
    }

    // Translate the per-query params into a generic key→f64 map.
    // The few keys we honour cover the top-10 indicators the UI
    // surfaces; library defaults fill in the rest.
    let mut params: HashMap<String, f64> = HashMap::new();
    if let Some(p) = q.period {
        params.insert("period".to_string(), p);
    }
    if let Some(s) = q.std_dev {
        params.insert("std_dev".to_string(), s);
    }
    if let Some(f) = q.fast {
        params.insert("fast".to_string(), f);
    }
    if let Some(s) = q.slow {
        params.insert("slow".to_string(), s);
    }
    if let Some(s) = q.signal {
        params.insert("signal".to_string(), s);
    }
    if let Some(k) = q.k_period {
        params.insert("k_period".to_string(), k);
    }
    if let Some(s) = q.k_slow {
        params.insert("k_slow".to_string(), s);
    }
    if let Some(d) = q.d_period {
        params.insert("d_period".to_string(), d);
    }

    let symbol_clone = symbol.clone();
    let timeframe_clone = timeframe.clone();
    let indicator_clone = indicator.clone();
    let result = tokio::task::spawn_blocking(move || {
        load_and_compute(symbol_clone, timeframe_clone, indicator_clone, params, limit)
    })
    .await;

    match result {
        Ok(Ok((candle_count, lines))) => Json(IndicatorDto {
            symbol,
            timeframe,
            indicator,
            candle_count,
            lines: lines
                .into_iter()
                .map(|l| IndicatorLineDto {
                    name: l.name,
                    values: l.values,
                })
                .collect(),
        })
        .into_response(),
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("indicator task panicked: {join_err}"),
            })),
        )
            .into_response(),
    }
}

fn load_and_compute(
    symbol: String,
    timeframe: String,
    indicator: String,
    params: HashMap<String, f64>,
    limit: usize,
) -> anyhow::Result<(usize, Vec<IndicatorLine>)> {
    let settings = Settings::from_yaml("config.yaml")
        .map_err(|e| anyhow::anyhow!("config.yaml not loadable: {e}"))?;
    let dataset = load_symbol_dataset(&settings.system.data_dir, &symbol)
        .map_err(|e| anyhow::anyhow!("dataset load failed for {symbol}: {e}"))?;
    let ohlcv = dataset.frames.get(&timeframe).ok_or_else(|| {
        anyhow::anyhow!(
            "timeframe '{timeframe}' not in dataset for {symbol} (available: {})",
            dataset
                .frames
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    // Compute on the entire dataset, then trim to the trailing
    // `limit` candles to match `/chart` semantics. Trimming after
    // compute avoids edge effects at the start of the window
    // (indicators need warm-up bars).
    let lines_full = compute_single_indicator(ohlcv, &indicator, &params)?;
    let total = ohlcv.len();
    let start = total.saturating_sub(limit);
    let trimmed: Vec<IndicatorLine> = lines_full
        .into_iter()
        .map(|l| IndicatorLine {
            name: l.name,
            values: l.values.into_iter().skip(start).collect(),
        })
        .collect();
    let returned_count = trimmed.first().map(|l| l.values.len()).unwrap_or(0);
    Ok((returned_count, trimmed))
}
