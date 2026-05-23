//! `/account/snapshot` — current account balance + open positions.
//!
//! Wire shape mirrors the `AccountSnapshot` class in
//! `experiments/forex-flutter-ui/lib/api/backend_client.dart`. Field
//! names use serde-rename-style camelCase so the Flutter side can
//! deserialize without a custom mapper — see
//! `serde(rename_all = "camelCase")` on each struct.
//!
//! ## Behaviour when broker is offline
//!
//! Phase 1 server fills the cache with a deterministic seed at boot
//! (see `state::AppApiState::with_seed_account`). Once the live
//! broker session lands, the seed gets overwritten the moment the
//! first cTrader account-info message arrives. Either way the route
//! returns 200 — Flutter doesn't need to special-case "no data yet".
//!
//! If the cache is truly empty (no seed AND no live data — only
//! happens if the bootstrap code is wrong) we return `503 Service
//! Unavailable` so the Flutter side can render a meaningful error
//! state instead of an empty json blob.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[cfg(test)]
use super::state::PositionPayload;
use super::state::{AccountSnapshotPayload, AppApiState};

/// Wire DTO. `serde(rename_all = "camelCase")` keeps the JSON keys
/// matching the Dart field names without us having to maintain two
/// independent naming conventions.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountSnapshotDto {
    pub balance: f64,
    pub equity: f64,
    pub free_margin: f64,
    pub used_margin: f64,
    pub currency: String,
    /// Server-side wall-clock (Unix milliseconds, UTC) for when
    /// this snapshot was assembled. The Flutter Dashboard renders
    /// "as of HH:MM:SS" in the user's local timezone next to the
    /// balance number so the operator can tell at a glance whether
    /// the displayed equity is fresh or carried over from a stale
    /// poll. Optional only because the DTO predates this field — a
    /// missing value renders as "—" and triggers the staleness
    /// banner.
    pub fetched_at_unix_ms: Option<i64>,
    pub positions: Vec<PositionDto>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PositionDto {
    /// cTrader position id — needed by the Close button to call
    /// `POST /positions/close`.
    pub position_id: i64,
    /// Broker volume in centi-lot units (what the close endpoint
    /// wants). The `volume` field below is the human-readable lot
    /// count.
    pub volume_units: i64,
    pub symbol: String,
    pub side: String,
    pub volume: f64,
    /// Unix-ms timestamp of the position open fill (UTC). Flutter
    /// converts to local time for the "since HH:MM" badge in the
    /// position row. None when cTrader didn't include a stamp in
    /// the reconcile payload (rare race window).
    pub open_timestamp_ms: Option<i64>,
    pub pnl_pips: f64,
    pub pnl_usd: f64,
}

impl From<crate::server::state::PositionPayload> for PositionDto {
    fn from(p: crate::server::state::PositionPayload) -> Self {
        PositionDto {
            position_id: p.position_id,
            volume_units: p.volume_units,
            symbol: p.symbol,
            side: p.side,
            volume: p.volume,
            open_timestamp_ms: p.open_timestamp_ms,
            pnl_pips: p.pnl_pips,
            pnl_usd: p.pnl_usd,
        }
    }
}

impl From<AccountSnapshotPayload> for AccountSnapshotDto {
    fn from(p: AccountSnapshotPayload) -> Self {
        Self {
            balance: p.balance,
            equity: p.equity,
            free_margin: p.free_margin,
            used_margin: p.used_margin,
            currency: p.currency,
            fetched_at_unix_ms: Some(p.fetched_at_unix_ms),
            positions: p.positions.into_iter().map(Into::into).collect(),
        }
    }
}

pub async fn snapshot(State(state): State<AppApiState>) -> Response {
    match state.account().await {
        Some(payload) => Json(AccountSnapshotDto::from(payload)).into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "broker session not ready",
                "code": "broker_not_ready",
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    fn seeded_state() -> AppApiState {
        AppApiState::new().with_seed_account(AccountSnapshotPayload {
            balance: 10_000.0,
            equity: 10_125.5,
            free_margin: 9_750.0,
            used_margin: 250.0,
            currency: "EUR".to_string(),
            fetched_at_unix_ms: 0,
            positions: vec![PositionPayload {
                position_id: 0,
                volume_units: 0,
                symbol: "EURUSD".to_string(),
                side: "LONG".to_string(),
                volume: 0.10,
                open_timestamp_ms: None,
                pnl_pips: 12.5,
                pnl_usd: 11.30,
            }],
        })
    }

    #[tokio::test]
    async fn snapshot_returns_seeded_account_as_camel_case_json() {
        let app = super::super::router(seeded_state());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/account/snapshot")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body collects");
        let text = std::str::from_utf8(&body).expect("utf-8 body");
        // CamelCase keys — important for Flutter side to deserialize.
        assert!(
            text.contains("\"freeMargin\""),
            "expected camelCase, got: {text}"
        );
        assert!(text.contains("\"usedMargin\""));
        assert!(text.contains("\"pnlPips\""));
        assert!(text.contains("\"pnlUsd\""));
        assert!(text.contains("EURUSD"));
    }

    #[tokio::test]
    async fn snapshot_returns_503_when_no_account_seeded() {
        let app = super::super::router(AppApiState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/account/snapshot")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
