//! GET /chart?symbol=EURUSD&timeframe=M1&limit=200
//!
//! Returns OHLC candles + price range for a given symbol/timeframe,
//! pulled from the local data dir (`data/symbol=<sym>/timeframe=<tf>/
//! data.parquet|data.vortex`). Read-only — no broker session needed,
//! so charts render even when cTrader is disconnected.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_data::{discover_timeframes, load_symbol_timeframe_tail};

use super::state::AppApiState;

const DEFAULT_LIMIT: usize = 200;
const MAX_LIMIT: usize = 2000;

#[derive(Debug, serde::Deserialize)]
pub struct ChartQuery {
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChartDto {
    pub symbol: String,
    pub timeframe: String,
    pub available_timeframes: Vec<String>,
    pub candle_count: usize,
    pub candles: Vec<CandleDto>,
    pub price_min: f64,
    pub price_max: f64,
    pub latest_close: f64,
    /// Percent change from first open in the window to last close.
    pub price_change_pct: f64,
    pub headline: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CandleDto {
    /// Unix timestamp in milliseconds. `None` if the dataset doesn't
    /// carry timestamps (synthetic / older data).
    pub ts_ms: Option<i64>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub async fn chart(State(_state): State<AppApiState>, Query(q): Query<ChartQuery>) -> Response {
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
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT).max(1);

    let result = tokio::task::spawn_blocking(move || {
        load_chart(symbol.clone(), timeframe.clone(), limit).or_else(|err| {
            // Returning 404 made Flutter render a generic
            // "Backend unreachable" banner that hid the *real*
            // remedy ("run Data Bootstrap for this symbol /
            // timeframe"). 200 with an empty candle list plus a
            // human-readable headline lets the Chart screen draw
            // its empty-state UI and the operator knows what to
            // do next.
            Ok::<ChartDto, anyhow::Error>(ChartDto {
                symbol: symbol.clone(),
                timeframe: timeframe.clone(),
                available_timeframes: Vec::new(),
                candle_count: 0,
                candles: Vec::new(),
                price_min: 0.0,
                price_max: 0.0,
                latest_close: 0.0,
                price_change_pct: 0.0,
                headline: format!(
                    "No data on disk for {symbol} {timeframe}. \
                         Go to Data Bootstrap and download a window \
                         from the broker, then come back. ({err})"
                ),
            })
        })
    })
    .await;

    match result {
        Ok(Ok(dto)) => Json(dto).into_response(),
        // load_chart's or_else above always returns Ok, so this arm
        // is only reachable if a future change re-introduces a fatal
        // error path; surface it as 500 so the UI banner makes sense.
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("blocking task panicked: {join_err}"),
            })),
        )
            .into_response(),
    }
}

/// Load OHLC candles for a symbol/timeframe from the local data dir.
pub fn load_chart(symbol: String, timeframe: String, limit: usize) -> anyhow::Result<ChartDto> {
    let settings = Settings::from_yaml("config.yaml")
        .map_err(|e| anyhow::anyhow!("config.yaml not loadable: {e}"))?;

    // #154: previously called `load_symbol_dataset` which eagerly loaded
    // ALL discovered timeframes for the symbol (commonly 19+ Vortex
    // files at ~30 MB each — half a gigabyte of disk reads to render a
    // single timeframe). Now we ask `discover_timeframes` for the
    // dropdown list (cheap directory listing, cached per #79) and load
    // only the requested timeframe's Vortex file.
    let mut available_timeframes = discover_timeframes(&settings.system.data_dir, &symbol)
        .map_err(|e| anyhow::anyhow!("timeframe discovery failed for {symbol}: {e}"))?;
    available_timeframes.sort();
    if !available_timeframes.contains(&timeframe) {
        anyhow::bail!(
            "timeframe '{timeframe}' not in dataset for {symbol} \
             (available: {})",
            available_timeframes.join(", ")
        );
    }

    // #155: ask the data layer for just the trailing `limit` rows so
    // we don't allocate a million-row Ohlcv only to slice the last 200.
    let ohlcv =
        load_symbol_timeframe_tail(&settings.system.data_dir, &symbol, &timeframe, limit)
            .map_err(|e| anyhow::anyhow!("dataset load failed for {symbol} {timeframe}: {e}"))?;

    let total = ohlcv.len();
    let timestamps = ohlcv.timestamp.as_deref();
    let volumes = ohlcv.volume.as_deref();

    let candles: Vec<CandleDto> = (0..total)
        .map(|idx| CandleDto {
            ts_ms: timestamps.and_then(|ts| ts.get(idx)).copied(),
            open: ohlcv.open[idx],
            high: ohlcv.high[idx],
            low: ohlcv.low[idx],
            close: ohlcv.close[idx],
            volume: volumes.and_then(|v| v.get(idx)).copied().unwrap_or(0.0),
        })
        .collect();

    let (price_min, price_max) = if candles.is_empty() {
        (0.0, 0.0)
    } else {
        candles.iter().fold((f64::MAX, f64::MIN), |(mn, mx), c| {
            (mn.min(c.low), mx.max(c.high))
        })
    };
    let latest_close = candles.last().map(|c| c.close).unwrap_or(0.0);
    let first_open = candles.first().map(|c| c.open).unwrap_or(0.0);
    let price_change_pct = if first_open > 0.0 {
        (latest_close - first_open) / first_open * 100.0
    } else {
        0.0
    };
    let headline = if candles.is_empty() {
        format!("No candles loaded for {symbol} {timeframe}")
    } else {
        format!(
            "{} candles · latest close {:.5} · range {:.5}–{:.5} · {:+.2}%",
            candles.len(),
            latest_close,
            price_min,
            price_max,
            price_change_pct
        )
    };

    Ok(ChartDto {
        symbol,
        timeframe,
        available_timeframes,
        candle_count: candles.len(),
        candles,
        price_min,
        price_max,
        latest_close,
        price_change_pct,
        headline,
    })
}
