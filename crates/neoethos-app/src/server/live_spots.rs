//! `GET /live/spots` — current live spot ticks per cTrader symbol.
//!
//! Reads the in-memory cache that `app_services::live_spots_streamer`
//! populates from the persistent WebSocket subscription (#137).
//!
//! Response shape:
//! ```json
//! {
//!   "spots": [
//!     {
//!       "symbolId": 1,
//!       "symbolName": "EURUSD",
//!       "bid": 1.0850,
//!       "ask": 1.0852,
//!       "midPrice": 1.0851,
//!       "receivedAtUnixMs": 1700000000000,
//!       "brokerTimestampMs": 1700000000000,
//!       "freshnessSeconds": 0.42
//!     },
//!     ...
//!   ],
//!   "snapshotAtUnixMs": 1700000000000,
//!   "symbolCount": 8
//! }
//! ```
//!
//! When the streamer hasn't connected yet (or no ticks have arrived
//! since the last clear), `spots` is an empty array and `symbolCount`
//! is 0. The UI uses an empty response to render a "waiting for
//! ticks…" placeholder rather than throwing an error.

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::app_services::live_spots::{SpotTick, snapshot_all};

use super::state::AppApiState;

/// Wire shape — camelCase here matches the rest of the HTTP layer.
/// The internal `SpotTick` struct is snake_case (Rust default); we
/// remap with this DTO so Flutter doesn't have to know the field
/// names diverge.
#[derive(Debug, Serialize)]
struct SpotTickDto {
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    #[serde(rename = "symbolName")]
    symbol_name: String,
    bid: Option<f64>,
    ask: Option<f64>,
    /// Convenience field — `(bid + ask) / 2` when both present.
    #[serde(rename = "midPrice")]
    mid_price: Option<f64>,
    #[serde(rename = "receivedAtUnixMs")]
    received_at_unix_ms: i64,
    #[serde(rename = "brokerTimestampMs")]
    broker_timestamp_ms: Option<i64>,
    /// Seconds since this tick was received. Lets the UI show a
    /// "stale tick" warning without doing clock math itself.
    #[serde(rename = "freshnessSeconds")]
    freshness_seconds: f64,
}

impl SpotTickDto {
    fn from_tick(tick: SpotTick, now_ms: i64) -> Self {
        let mid = tick.mid_price();
        let freshness = ((now_ms - tick.received_at_unix_ms) as f64 / 1000.0).max(0.0);
        Self {
            symbol_id: tick.symbol_id,
            symbol_name: tick.symbol_name,
            bid: tick.bid,
            ask: tick.ask,
            mid_price: mid,
            received_at_unix_ms: tick.received_at_unix_ms,
            broker_timestamp_ms: tick.broker_timestamp_ms,
            freshness_seconds: freshness,
        }
    }
}

#[derive(Debug, Serialize)]
struct SpotsResponse {
    spots: Vec<SpotTickDto>,
    #[serde(rename = "snapshotAtUnixMs")]
    snapshot_at_unix_ms: i64,
    #[serde(rename = "symbolCount")]
    symbol_count: usize,
}

pub async fn list(State(_state): State<AppApiState>) -> Response {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let mut ticks = snapshot_all();
    // Stable order so the UI's polling diff is easier on the eye.
    ticks.sort_by(|a, b| a.symbol_name.cmp(&b.symbol_name));
    let count = ticks.len();
    let spots: Vec<SpotTickDto> = ticks
        .into_iter()
        .map(|t| SpotTickDto::from_tick(t, now_ms))
        .collect();
    Json(SpotsResponse {
        spots,
        snapshot_at_unix_ms: now_ms,
        symbol_count: count,
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dto_computes_mid_and_freshness() {
        let tick = SpotTick {
            symbol_id: 1,
            symbol_name: "EURUSD".to_string(),
            bid: Some(1.085),
            ask: Some(1.0852),
            received_at_unix_ms: 1_000_000,
            broker_timestamp_ms: Some(999_999),
        };
        let dto = SpotTickDto::from_tick(tick, 1_500_000);
        assert_eq!(dto.symbol_id, 1);
        assert_eq!(dto.bid, Some(1.085));
        assert_eq!(dto.ask, Some(1.0852));
        assert!(dto.mid_price.is_some());
        assert!((dto.mid_price.unwrap() - 1.0851).abs() < 1e-6);
        // Δt is 500_000 ms = 500 s
        assert!((dto.freshness_seconds - 500.0).abs() < 1e-3);
    }
}
