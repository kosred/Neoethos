//! GET /chart
//!
//! Returns OHLC candles from one of two explicit, mutually exclusive modes:
//!
//! - **broker-live** — omit `datasetIdentity` and `expectedGeneration`, and
//!   provide `symbol` / `timeframe`. The route asks cTrader for that exact
//!   period and returns an error when the live request fails.
//! - **exact-local** — provide the opaque canonical `datasetIdentity` and its
//!   current `expectedGeneration` receipt from `/data/bootstrap`. Symbol and
//!   timeframe are derived from the identity; optional text fields are only
//!   consistency assertions. The route fully verifies and pins that immutable
//!   Vortex generation before reading values.
//!
//! These modes never fall through to one another. Every timeframe is a direct
//! broker/import artifact selected by its own identity.

use std::path::Path;

use anyhow::{Context, ensure};
use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_data::{CanonicalDatasetIdentity, CanonicalOhlcvFrame, load_canonical_timeframe};
use serde::Deserialize;

use super::errors::{actionable_error, internal_panic};
use super::state::AppApiState;

const DEFAULT_LIMIT: usize = 500;
const MAX_LIMIT: usize = 2000;

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChartQuery {
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    /// Exact canonical `d1-*` identity selected from `/data/bootstrap`.
    /// When present, `expected_generation` is mandatory and the route reads
    /// only that local immutable generation; it never contacts the broker.
    #[serde(default, deserialize_with = "deserialize_optional_dataset_identity")]
    pub dataset_identity: Option<CanonicalDatasetIdentity>,
    /// Current content-addressed generation receipt returned by the inventory.
    pub expected_generation: Option<String>,
    pub limit: Option<usize>,
}

pub(super) fn deserialize_optional_dataset_identity<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<CanonicalDatasetIdentity>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let encoded = Option::<String>::deserialize(deserializer)?;
    encoded
        .map(|value| {
            CanonicalDatasetIdentity::from_path_component(&value).map_err(|error| {
                serde::de::Error::custom(format!(
                    "invalid canonical dataset identity {value:?}: {error}"
                ))
            })
        })
        .transpose()
}

/// Opaque selection of one exact current local generation.
///
/// The identity prevents source/account ambiguity. The generation receipt
/// prevents a caller that selected an earlier inventory snapshot from silently
/// reading a newer publication. `load_exact_current_frame` retains the reader
/// lease for the whole computation that consumes the frame.
#[derive(Debug, Clone)]
pub(super) struct ExactDatasetReceipt {
    identity: CanonicalDatasetIdentity,
    expected_generation: String,
}

impl ExactDatasetReceipt {
    pub(super) fn from_optional(
        identity: Option<CanonicalDatasetIdentity>,
        expected_generation: Option<String>,
    ) -> anyhow::Result<Option<Self>> {
        match (identity, expected_generation) {
            (None, None) => Ok(None),
            (Some(identity), Some(expected_generation)) => {
                let expected_generation = expected_generation.trim().to_owned();
                ensure!(
                    !expected_generation.is_empty(),
                    "expectedGeneration cannot be empty"
                );
                Ok(Some(Self {
                    identity,
                    expected_generation,
                }))
            }
            (Some(_), None) => {
                anyhow::bail!("expectedGeneration is required with an exact datasetIdentity")
            }
            (None, Some(_)) => {
                anyhow::bail!("datasetIdentity is required with an expectedGeneration receipt")
            }
        }
    }

    pub(super) const fn identity(&self) -> &CanonicalDatasetIdentity {
        &self.identity
    }

    pub(super) fn expected_generation(&self) -> &str {
        &self.expected_generation
    }
}

pub(super) fn load_exact_current_frame(
    root: impl AsRef<Path>,
    receipt: &ExactDatasetReceipt,
) -> anyhow::Result<CanonicalOhlcvFrame> {
    let frame = load_canonical_timeframe(root, receipt.identity()).with_context(|| {
        format!(
            "failed to fully verify and pin exact dataset {}",
            receipt.identity().to_path_component()
        )
    })?;
    ensure!(
        frame.artifact().generation_id() == receipt.expected_generation(),
        "selected dataset generation changed: identity={}, expected={}, current={}; refresh the data inventory before retrying",
        receipt.identity().to_path_component(),
        receipt.expected_generation(),
        frame.artifact().generation_id()
    );
    Ok(frame)
}

#[derive(Debug, Clone)]
enum ChartRequest {
    BrokerLive { symbol: String, timeframe: String },
    ExactLocal { receipt: ExactDatasetReceipt },
}

impl ChartRequest {
    fn symbol(&self) -> &str {
        match self {
            Self::BrokerLive { symbol, .. } => symbol,
            Self::ExactLocal { receipt } => receipt.identity().symbol_name(),
        }
    }

    fn timeframe(&self) -> &str {
        match self {
            Self::BrokerLive { timeframe, .. } => timeframe,
            Self::ExactLocal { receipt } => receipt.identity().timeframe().as_str(),
        }
    }

    const fn is_broker_live(&self) -> bool {
        matches!(self, Self::BrokerLive { .. })
    }
}

fn resolve_chart_request(query: ChartQuery) -> anyhow::Result<(ChartRequest, usize)> {
    let limit = query.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let receipt =
        ExactDatasetReceipt::from_optional(query.dataset_identity, query.expected_generation)?;
    match receipt {
        Some(receipt) => {
            if let Some(asserted) = query
                .symbol
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                ensure!(
                    asserted.eq_ignore_ascii_case(receipt.identity().symbol_name()),
                    "symbol assertion {asserted:?} disagrees with exact dataset identity symbol {:?}",
                    receipt.identity().symbol_name()
                );
            }
            if let Some(asserted) = query
                .timeframe
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                ensure!(
                    asserted.eq_ignore_ascii_case(receipt.identity().timeframe().as_str()),
                    "timeframe assertion {asserted:?} disagrees with exact dataset identity timeframe {:?}",
                    receipt.identity().timeframe().as_str()
                );
            }
            Ok((ChartRequest::ExactLocal { receipt }, limit))
        }
        None => {
            let symbol = query
                .symbol
                .unwrap_or_else(|| "EURUSD".to_owned())
                .trim()
                .to_uppercase();
            let timeframe = query
                .timeframe
                .unwrap_or_else(|| "M1".to_owned())
                .trim()
                .to_uppercase();
            ensure!(!symbol.is_empty(), "broker-live symbol cannot be empty");
            ensure!(
                !timeframe.is_empty(),
                "broker-live timeframe cannot be empty"
            );
            Ok((ChartRequest::BrokerLive { symbol, timeframe }, limit))
        }
    }
}

/// Provenance tag for the explicitly selected chart mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChartDataSource {
    /// Live OHLCV bars fetched from the cTrader broker historical-bars
    /// API. The authoritative, current source — UI shows no "cached"
    /// banner.
    Broker,
    /// Exact, fully verified local Vortex generation selected by identity and
    /// generation receipt. The historical wire label remains `disk-cache`.
    DiskCache,
    /// A successful broker-history page with no rows.
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
    /// Unix timestamp in milliseconds. Exact local generations always carry
    /// canonical timestamps; broker-live rows carry broker timestamps.
    pub ts_ms: Option<i64>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub async fn chart(State(state): State<AppApiState>, Query(q): Query<ChartQuery>) -> Response {
    let (request, limit) = match resolve_chart_request(q) {
        Ok(selection) => selection,
        Err(error) => {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Chart request is not an exact local receipt or a valid broker-live selection.",
                &error,
            );
        }
    };
    let symbol = request.symbol().to_owned();
    let timeframe = request.timeframe().to_owned();
    let broker_live = request.is_broker_live();

    // The legacy cache key cannot encode a canonical identity or immutable
    // generation. It is therefore safe only for broker-live responses. Exact
    // local reads always re-open, fully verify and pin the requested receipt.
    if broker_live {
        if let Some(cached) = super::chart_cache::get(&symbol, &timeframe, limit)
            .filter(|cached| cached.source == ChartDataSource::Broker)
        {
            return Json(cached).into_response();
        }
    }

    let config_path = state.config_path().to_path_buf();
    let symbol_for_cache = symbol.clone();
    let timeframe_for_cache = timeframe.clone();
    let result = tokio::task::spawn_blocking(move || match request {
        ChartRequest::BrokerLive { symbol, timeframe } => {
            load_broker_chart(symbol, timeframe, limit)
        }
        ChartRequest::ExactLocal { receipt } => {
            load_exact_local_chart(&config_path, receipt, limit)
        }
    })
    .await;

    match result {
        Ok(Ok(dto)) => {
            if broker_live {
                super::chart_cache::put(
                    &symbol_for_cache,
                    &timeframe_for_cache,
                    limit,
                    dto.clone(),
                );
            }
            Json(dto).into_response()
        }
        Ok(Err(err)) => actionable_error(
            if broker_live {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::CONFLICT
            },
            if broker_live {
                "Broker-live chart data could not be loaded; no local fallback was attempted."
            } else {
                "The exact local dataset receipt could not be fully verified and pinned. Refresh the data inventory before retrying."
            },
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
        Ok(Err(err)) => actionable_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Broker-live chart history could not be loaded; no local fallback was attempted.",
            &err,
        ),
        Err(join_err) => internal_panic("Loading older chart bars", join_err),
    }
}

fn load_broker_chart(symbol: String, timeframe: String, limit: usize) -> anyhow::Result<ChartDto> {
    let bars = crate::app_services::broker_api::fetch_recent_chart_bars_blocking(
        &symbol, &timeframe, limit,
    )
    .with_context(|| format!("broker-live chart fetch failed for {symbol} {timeframe}"))?;
    ensure!(
        !bars.is_empty(),
        "broker-live chart returned no bars for {symbol} {timeframe}"
    );
    let candles = bars
        .into_iter()
        .map(|bar| CandleDto {
            ts_ms: Some(bar.timestamp_ms),
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume.unwrap_or(0) as f64,
        })
        .collect();
    Ok(build_chart_dto(
        symbol,
        timeframe.clone(),
        vec![timeframe],
        candles,
        ChartDataSource::Broker,
    ))
}

fn load_exact_local_chart(
    config_path: &Path,
    receipt: ExactDatasetReceipt,
    limit: usize,
) -> anyhow::Result<ChartDto> {
    let settings = Settings::from_yaml(config_path)
        .map_err(|error| anyhow::anyhow!("{} not loadable: {error}", config_path.display()))?;
    let frame = load_exact_current_frame(&settings.system.data_dir, &receipt)?;
    let identity = receipt.identity();
    ensure!(
        frame.artifact().identity() == identity,
        "verified local frame identity changed after exact selection"
    );
    let ohlcv = frame.ohlcv();
    let timestamps = ohlcv
        .timestamp
        .as_deref()
        .context("verified canonical chart generation has no timestamp_ms")?;
    let total = ohlcv.len();
    let start = total.saturating_sub(limit);
    let volumes = ohlcv.volume.as_deref();
    let candles = (start..total)
        .map(|index| CandleDto {
            ts_ms: Some(timestamps[index]),
            open: ohlcv.open[index],
            high: ohlcv.high[index],
            low: ohlcv.low[index],
            close: ohlcv.close[index],
            volume: volumes
                .and_then(|values| values.get(index))
                .copied()
                .unwrap_or(0.0),
        })
        .collect();
    let symbol = identity.symbol_name().to_owned();
    let timeframe = identity.timeframe().as_str().to_owned();
    Ok(build_chart_dto(
        symbol,
        timeframe.clone(),
        vec![timeframe],
        candles,
        ChartDataSource::DiskCache,
    ))
}

fn build_chart_dto(
    symbol: String,
    timeframe: String,
    available_timeframes: Vec<String>,
    candles: Vec<CandleDto>,
    source: ChartDataSource,
) -> ChartDto {
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

    ChartDto {
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
    }
}
