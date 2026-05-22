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
use neoethos_data::load_symbol_dataset;

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

pub async fn chart(
    State(_state): State<AppApiState>,
    Query(q): Query<ChartQuery>,
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
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT).max(1);

    let result = tokio::task::spawn_blocking(move || {
        load_chart(symbol, timeframe, limit)
    })
    .await;

    match result {
        Ok(Ok(dto)) => Json(dto).into_response(),
        Ok(Err(err)) => (
            StatusCode::NOT_FOUND,
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

fn load_chart(
    symbol: String,
    timeframe: String,
    limit: usize,
) -> anyhow::Result<ChartDto> {
    let settings = Settings::from_yaml("config.yaml")
        .map_err(|e| anyhow::anyhow!("config.yaml not loadable: {e}"))?;
    let dataset = load_symbol_dataset(&settings.system.data_dir, &symbol)
        .map_err(|e| anyhow::anyhow!("dataset load failed for {symbol}: {e}"))?;

    let mut available_timeframes: Vec<String> =
        dataset.frames.keys().cloned().collect();
    available_timeframes.sort();

    let ohlcv = dataset
        .frames
        .get(&timeframe)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "timeframe '{timeframe}' not in dataset for {symbol} \
                 (available: {})",
                available_timeframes.join(", ")
            )
        })?;

    let total = ohlcv.len();
    let start = total.saturating_sub(limit);
    let timestamps = ohlcv.timestamp.as_deref();
    let volumes = ohlcv.volume.as_deref();

    let candles: Vec<CandleDto> = (start..total)
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
        candles
            .iter()
            .fold((f64::MAX, f64::MIN), |(mn, mx), c| {
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
