//! GET /chart?symbol=EURUSD&timeframe=M1&limit=200
//!
//! Returns OHLC candles + price range for a given symbol/timeframe.
//!
//! ## G7 broker-passthrough doctrine (operator-approved 2026-05-25)
//!
//! Operator directive: "δεχόμαστε ότι στέλνει ο broker αυτό είναι
//! αλήθεια το άλλο συνθετικό" — we accept what the broker sends as
//! truth; everything else is synthetic. For chart data this means
//! the routing priority is:
//!
//! 1. **Live broker historical-bars API** — when cTrader session is
//!    connected, fetch via `ProtoOAGetTrendbarsReq` for the exact
//!    `symbol × timeframe × period` requested. This is the ONLY
//!    authoritative source. (Wiring lands in G7 Phase 2 — the
//!    `live_spots_streamer` ring buffer already serves real-time
//!    quotes; the historical bars layer is the next slot.)
//!
//! 2. **Local Vortex cache** — when cTrader is DISCONNECTED. Marked
//!    `source: "disk-cache"` in the response so the UI can render a
//!    "showing cached data, live unavailable" banner. The cache is
//!    LEGITIMATE for offline replay + backtesting reproducibility
//!    (operator preserved that use case in the G7 sign-off).
//!
//! 3. **Empty + headline** — when neither source is available. The
//!    UI renders an empty-state with the bootstrap call-to-action.
//!
//! The current implementation is **G7 Phase 1**: still disk-only, but
//! the response shape carries a `source` annotation so the UI can
//! distinguish disk-cache from broker-passthrough once Phase 2 lands.
//! No behaviour change for disconnected operators; the UI now knows
//! the data is *cached* rather than blindly assuming it's live.

use std::path::PathBuf;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_data::{discover_timeframes, load_symbol_timeframe_tail};

use super::errors::{actionable_error, internal_panic};
use super::state::AppApiState;

const DEFAULT_LIMIT: usize = 500;
const MAX_LIMIT: usize = 2000;

#[derive(Debug, serde::Deserialize)]
pub struct ChartQuery {
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub limit: Option<usize>,
}

/// Provenance tag — distinguishes broker-truth from disk-cached
/// responses. **G7 Phase 1 (2026-05-25)** annotation per the
/// operator-approved broker-passthrough doctrine: chart data sourced
/// from the broker WSS is "live" / "broker"; everything from local
/// Vortex files is "disk-cache" / "empty" so the UI can render the
/// right banner. Phase 2 plumbs the broker source through.
///
/// **Phase 2 (2026-06-01)** — `Broker` reintroduced. `load_chart` now
/// fetches live trendbars straight from cTrader (`ProtoOAGetTrendbarsReq`
/// via `broker_api::fetch_recent_chart_bars_blocking`) and tags the
/// response `broker` when the broker session served the candles; it
/// falls back to `DiskCache` (local Vortex files) only when the broker
/// is unreachable, and `Empty` when neither has data. The Flutter
/// `ChartSnapshot.isBrokerSource` (`source == "broker"`) already
/// consumes this tag to hide the "cached" banner — producer and
/// consumer stay in lockstep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChartDataSource {
    /// Live OHLCV bars fetched from the cTrader broker historical-bars
    /// API. The authoritative, current source — UI shows no "cached"
    /// banner.
    Broker,
    /// Data loaded from the local Vortex cache. Use for offline
    /// replay / backtests, or when the broker is unreachable. UI
    /// should surface a "cached" banner.
    DiskCache,
    /// No data available from any source.
    Empty,
}

// Clone needed by `chart_cache` (in-RAM LRU cache for repeat-click
// timeframe switches — 2026-05-25 operator directive). The cache
// stores DTOs and clones them on get/put so the response path and
// cache state remain independent.
#[derive(Debug, Clone, serde::Serialize)]
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
    /// **G7 Phase 1 (2026-05-25)** — provenance annotation. Tells
    /// the UI whether the response is live broker data or a disk
    /// cache. Default `disk-cache` for current Phase-1 wiring; will
    /// promote to `broker` in Phase 2 when broker historical-bars
    /// integration lands.
    pub source: ChartDataSource,
}

// Clone needed because `CandleDto` is a field of `ChartDto` (which
// derives Clone — see chart_cache rationale above).
#[derive(Debug, Clone, serde::Serialize)]
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

pub async fn chart(State(state): State<AppApiState>, Query(q): Query<ChartQuery>) -> Response {
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

    // **2026-05-25 — chart-cache fast path** (operator directive: TF
    // switch must be immediate, no 100-500ms disk I/O per click).
    // Check the in-memory `ChartDto` LRU cache first; on hit, return
    // without ever touching disk. On miss, fall through to the
    // existing spawn_blocking path which loads from Vortex AND
    // populates the cache for the next click.
    if let Some(cached) = super::chart_cache::get(&symbol, &timeframe, limit) {
        return Json(cached).into_response();
    }

    // F-553/F-576 closure (2026-05-25): config path threaded from the
    // CLI `--config` flag via `AppApiState::config_path()` instead of
    // hardcoded inside `load_chart`.
    let config_path = state.config_path().to_path_buf();
    let symbol_for_cache = symbol.clone();
    let timeframe_for_cache = timeframe.clone();
    let result = tokio::task::spawn_blocking(move || {
        load_chart(&config_path, symbol.clone(), timeframe.clone(), limit).or_else(|err| {
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
                source: ChartDataSource::Empty,
            })
        })
    })
    .await;

    match result {
        Ok(Ok(dto)) => {
            // **2026-05-25 — chart-cache populate**: cache the freshly-
            // loaded DTO so the next click on the same (symbol, TF, limit)
            // is a ~1 µs in-RAM hit instead of another 100-500 ms disk
            // read. TTL inside `chart_cache` keeps the cache from
            // serving stale data once the live bar progresses.
            super::chart_cache::put(&symbol_for_cache, &timeframe_for_cache, limit, dto.clone());
            Json(dto).into_response()
        }
        // load_chart's or_else above always returns Ok, so this arm
        // is only reachable if a future change re-introduces a fatal
        // error path; surface a friendly message so the UI banner helps.
        Ok(Err(err)) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Chart data could not be loaded. If the broker is connected, refresh; \
             otherwise open Data Bootstrap and download a window for this symbol/timeframe.",
            &err,
        ),
        Err(join_err) => internal_panic("Loading chart data", join_err),
    }
}

// ─── GET /chart/history ───────────────────────────────────────────────────
//
// Scroll-back pagination. The Flutter chart calls this from k_chart_plus's
// `onLoadMore` when the operator pans left past the oldest loaded candle:
// it returns the next page of OLDER bars (strictly before `beforeMs`),
// fetched live from the broker and held only in the client's memory. This
// is the TradingView model — panning two years back costs ZERO disk; the
// local Vortex cache is written only by explicit Data Bootstrap / discovery
// auto-fetch, never by viewing a chart.

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChartHistoryQuery {
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    /// Cursor: return bars STRICTLY OLDER than this unix-ms timestamp
    /// (the time of the oldest candle the client currently holds).
    pub before_ms: i64,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChartHistoryDto {
    pub symbol: String,
    pub timeframe: String,
    pub candle_count: usize,
    /// Older candles, oldest→newest, all strictly before the cursor.
    pub candles: Vec<CandleDto>,
    /// `false` once the broker returns an empty page — the client stops
    /// asking for more.
    pub has_more: bool,
    pub source: ChartDataSource,
}

pub async fn chart_history(
    State(_state): State<AppApiState>,
    Query(q): Query<ChartHistoryQuery>,
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
    let before_ms = q.before_ms;
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT).max(1);

    if before_ms <= 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "beforeMs must be a positive unix-millis cursor",
            })),
        )
            .into_response();
    }

    let symbol_for_dto = symbol.clone();
    let timeframe_for_dto = timeframe.clone();
    let result = tokio::task::spawn_blocking(move || {
        crate::app_services::broker_api::fetch_chart_bars_before_blocking(
            &symbol, &timeframe, before_ms, limit,
        )
    })
    .await;

    match result {
        Ok(Ok(bars)) => {
            let candles: Vec<CandleDto> = bars
                .iter()
                .map(|b| CandleDto {
                    ts_ms: Some(b.timestamp_ms),
                    open: b.open,
                    high: b.high,
                    low: b.low,
                    close: b.close,
                    volume: b.volume.unwrap_or(0) as f64,
                })
                .collect();
            let source = if candles.is_empty() {
                ChartDataSource::Empty
            } else {
                ChartDataSource::Broker
            };
            Json(ChartHistoryDto {
                symbol: symbol_for_dto,
                timeframe: timeframe_for_dto,
                candle_count: candles.len(),
                // A non-empty page means there may be more older bars; an
                // empty page means we've reached the broker's earliest
                // coverage, so the client stops paginating.
                has_more: !candles.is_empty(),
                candles,
                source,
            })
            .into_response()
        }
        // Broker unreachable / no session: 200 with an empty page so the
        // chart simply stops scrolling back rather than throwing — older
        // history just isn't available right now.
        Ok(Err(err)) => {
            tracing::debug!(
                target: "neoethos_app::server::chart",
                symbol = %symbol_for_dto,
                timeframe = %timeframe_for_dto,
                error = %err,
                "chart history fetch failed; returning empty page"
            );
            Json(ChartHistoryDto {
                symbol: symbol_for_dto,
                timeframe: timeframe_for_dto,
                candle_count: 0,
                candles: Vec::new(),
                has_more: false,
                source: ChartDataSource::Empty,
            })
            .into_response()
        }
        Err(join_err) => internal_panic("Loading older chart bars", join_err),
    }
}

/// Load OHLC candles for a symbol/timeframe from the local data dir.
pub fn load_chart(
    config_path: &PathBuf,
    symbol: String,
    timeframe: String,
    limit: usize,
) -> anyhow::Result<ChartDto> {
    let settings = Settings::from_yaml(config_path)
        .map_err(|e| anyhow::anyhow!("{} not loadable: {e}", config_path.display()))?;

    // #154: previously called `load_symbol_dataset` which eagerly loaded
    // ALL discovered timeframes for the symbol (commonly 19+ Vortex
    // files at ~30 MB each — half a gigabyte of disk reads to render a
    // single timeframe). Now we ask `discover_timeframes` for the
    // dropdown list (cheap directory listing, cached per #79) and load
    // only the requested timeframe's Vortex file.
    // Local timeframe list for the dropdown. Non-fatal: a symbol with no
    // local cache can still be charted straight from the broker below, and
    // the broker timeframe gets appended on a successful live fetch.
    let mut available_timeframes =
        discover_timeframes(&settings.system.data_dir, &symbol).unwrap_or_default();
    available_timeframes.sort();

    // Phase 2 — broker-passthrough: fetch LIVE trendbars straight from
    // cTrader first (the authoritative, current source). Fall back to the
    // local Vortex cache only when the broker is unreachable / has no
    // session. This is what makes the chart show *current* candles instead
    // of a stale bootstrap snapshot.
    let (candles, source): (Vec<CandleDto>, ChartDataSource) =
        match crate::app_services::broker_api::fetch_recent_chart_bars_blocking(
            &symbol, &timeframe, limit,
        ) {
            Ok(bars) if !bars.is_empty() => {
                // The broker serves this timeframe even if the local cache
                // doesn't — make sure the dropdown lists it.
                if !available_timeframes.contains(&timeframe) {
                    available_timeframes.push(timeframe.clone());
                    available_timeframes.sort();
                }
                let candles = bars
                    .iter()
                    .map(|b| CandleDto {
                        ts_ms: Some(b.timestamp_ms),
                        open: b.open,
                        high: b.high,
                        low: b.low,
                        close: b.close,
                        volume: b.volume.unwrap_or(0) as f64,
                    })
                    .collect();
                (candles, ChartDataSource::Broker)
            }
            broker_result => {
                if let Err(err) = &broker_result {
                    tracing::debug!(
                        target: "neoethos_app::server::chart",
                        symbol = %symbol,
                        timeframe = %timeframe,
                        error = %err,
                        "broker chart fetch failed; falling back to local Vortex cache"
                    );
                }
                // Disk fallback requires the timeframe to exist locally.
                if !available_timeframes.contains(&timeframe) {
                    anyhow::bail!(
                        "timeframe '{timeframe}' not available for {symbol} from \
                         broker or local cache (cached: {})",
                        available_timeframes.join(", ")
                    );
                }
                // #155: trailing `limit` rows only — avoid loading a
                // million-row Ohlcv just to slice the tail.
                let ohlcv = load_symbol_timeframe_tail(
                    &settings.system.data_dir,
                    &symbol,
                    &timeframe,
                    limit,
                )
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
                let source = if candles.is_empty() {
                    ChartDataSource::Empty
                } else {
                    ChartDataSource::DiskCache
                };
                (candles, source)
            }
        };

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
        source,
    })
}
