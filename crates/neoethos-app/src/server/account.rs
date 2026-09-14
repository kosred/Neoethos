//! `/account/snapshot` — current account balance + open positions.
//!
//! Production starts without an account snapshot. A successful broker
//! refresh publishes camelCase data; a missing or failed refresh returns 503
//! with the recorded cause. Test-only seeded accounts are never startup data.

use std::convert::Infallible;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream::{Stream, StreamExt};
use tokio_stream::wrappers::BroadcastStream;

#[cfg(test)]
use super::state::PositionPayload;
use super::state::{AccountRefreshFailure, AccountSnapshotPayload, AppApiState};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;

/// Wire DTO. `serde(rename_all = "camelCase")` keeps the JSON keys
/// matching the Dart field names without us having to maintain two
/// independent naming conventions.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountSnapshotDto {
    /// Identity from the completed broker response/request scope, never reloaded settings.
    /// A string preserves all i64 account IDs when consumed by JavaScript.
    pub source_account_id: String,
    pub source_environment: &'static str,
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
    pub pnl_pips: Option<f64>,
    pub pnl_usd: f64,
    /// Entry price / SL / TP straight from the broker — no client merge.
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    /// Volume in lots (cTrader parity); `None` if symbol not in metadata.
    pub volume_lots: Option<f64>,
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
            entry_price: p.entry_price,
            stop_loss: p.stop_loss,
            take_profit: p.take_profit,
            volume_lots: p.volume_lots,
        }
    }
}

impl From<AccountSnapshotPayload> for AccountSnapshotDto {
    fn from(p: AccountSnapshotPayload) -> Self {
        Self {
            source_account_id: p.source_account_id.to_string(),
            source_environment: match p.source_environment {
                crate::app_services::ctrader_live_auth::CTraderEnvironment::Demo => "Demo",
                crate::app_services::ctrader_live_auth::CTraderEnvironment::Live => "Live",
            },
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

/// **2026-05-25 — operator directive "uniform push everywhere"**:
/// `GET /account/snapshot/stream` — Server-Sent Events that pushes
/// every account update (balance, equity, free margin, positions,
/// PnL) the moment the bridge writes a fresh snapshot to the cache.
///
/// Replaces the Flutter 1Hz polling of `/account/snapshot` with a
/// real-time push channel. Latency = network RTT (~1-5 ms) instead
/// of poll interval (~1000 ms).
///
/// Mirror of `live_spots::stream` for ticks. Same SSE wire shape:
/// `event: account` + JSON payload per snapshot, plus a 15 s
/// keep-alive so HTTP proxies don't idle-close the connection.
///
/// The polling `/account/snapshot` route is kept as a fallback for
/// cold-start (Flutter calls it once on mount before switching to
/// the SSE stream) and for HTTP clients without SSE support.
///
/// **Architectural note**: positions are part of the
/// `AccountSnapshotPayload` shape so the same stream covers
/// balance + equity + open positions. A separate `/positions/stream`
/// would be redundant; one channel serves the Dashboard's full view.
pub async fn stream(
    State(state): State<AppApiState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let receiver = state.subscribe_account();
    let stream = BroadcastStream::new(receiver).filter_map(move |res| {
        let state = state.clone();
        async move {
            let payload = res.ok()?;
            // A queued event can outlive an account switch or a newer refresh.
            let (current, failure) =
                observation_with_scope(&state, super::bridge::current_execution_account_scope)
                    .await;
            let current = current?;
            if failure.is_some()
                || payload.source_account_id != current.source_account_id
                || payload.source_environment != current.source_environment
                || payload.fetched_at_unix_ms != current.fetched_at_unix_ms
            {
                return None;
            }
            let json = serde_json::to_string(&AccountSnapshotDto::from(payload)).ok()?;
            Some(Ok(Event::default().event("account").data(json)))
        }
    });
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

/// `POST /account/snapshot/refresh` — operator-triggered immediate
/// account refresh. Pings the bridge's `account_refresh_rx` channel
/// so the next polling iteration fires NOW instead of waiting up to
/// 5 s for the timer. Returns the freshly-cached snapshot (or 503
/// if the cache is still empty after the refresh, e.g. broker
/// session not yet established).
///
/// **2026-05-25 — operator directive "uniform push everywhere"**:
/// same trigger channel that the future `OAExecutionEvent` handler
/// will use; exposing it as an HTTP endpoint gives the operator a
/// "force refresh" button in the UI without any extra plumbing.
pub async fn refresh(State(state): State<AppApiState>) -> Response {
    state.trigger_account_refresh();
    // Allow the triggered request to begin. This delay does not certify a new
    // response: the returned snapshot retains its actual fetch timestamp.
    tokio::time::sleep(std::time::Duration::from_millis(750)).await;
    snapshot(State(state)).await
}

pub async fn snapshot(State(state): State<AppApiState>) -> Response {
    snapshot_with_scope(state, super::bridge::current_execution_account_scope).await
}

/// Shared monitoring read for HTTP, SSE and Supervisor. The resolver is run on
/// the blocking pool under the cache lock, not against a test-specific profile.
pub(crate) async fn observation_with_scope(
    state: &AppApiState,
    resolve_scope: impl FnOnce() -> anyhow::Result<(i64, CTraderEnvironment)> + Send + 'static,
) -> (
    Option<AccountSnapshotPayload>,
    Option<AccountRefreshFailure>,
) {
    let state = state.clone();
    match tokio::task::spawn_blocking(move || state.current_account_observation(resolve_scope))
        .await
    {
        Ok(Ok(observation)) => observation,
        result => {
            let detail = match result {
                Ok(Err(error)) => error.to_string(),
                Err(_) => "Account observation worker failed".to_owned(),
                Ok(Ok(_)) => unreachable!(),
            };
            (
                None,
                Some(AccountRefreshFailure {
                    code: "account_scope_unavailable".to_owned(),
                    detail,
                    observed_at_unix_ms: chrono::Utc::now().timestamp_millis(),
                }),
            )
        }
    }
}

async fn snapshot_with_scope(
    state: AppApiState,
    resolve_scope: impl FnOnce() -> anyhow::Result<(i64, CTraderEnvironment)> + Send + 'static,
) -> Response {
    let (account, failure) = observation_with_scope(&state, resolve_scope).await;
    if let Some(failure) = failure {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Account snapshot refresh failed; account values and positions are not confirmed current.",
                "code": failure.code,
                "detail": failure.detail,
                "observedAtUnixMs": failure.observed_at_unix_ms,
                "lastSnapshotAtUnixMs": account.as_ref().map(|value| value.fetched_at_unix_ms),
            })),
        ).into_response();
    }
    match account {
        Some(payload) => Json(AccountSnapshotDto::from(payload)).into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Waiting for the first verified account snapshot; positions are unknown.",
                "code": "account_snapshot_pending",
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
            source_account_id: 42,
            source_environment: crate::app_services::ctrader_live_auth::CTraderEnvironment::Demo,
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
                pnl_pips: Some(12.5),
                pnl_usd: 11.30,
                entry_price: Some(1.0850),
                stop_loss: None,
                take_profit: None,
                volume_lots: Some(0.10),
            }],
        })
    }

    #[tokio::test]
    async fn snapshot_returns_seeded_account_as_camel_case_json() {
        let app = axum::Router::new()
            .route(
                "/account/snapshot",
                axum::routing::get(|State(state): State<AppApiState>| async move {
                    snapshot_with_scope(state, || Ok((42, CTraderEnvironment::Demo))).await
                }),
            )
            .with_state(seeded_state());
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
        let wire: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(wire["sourceAccountId"], "42");
        assert_eq!(wire["sourceEnvironment"], "Demo");
    }

    #[tokio::test]
    async fn snapshot_dto_retains_exact_source_identity_for_late_updates_without_relabelling() {
        use crate::app_services::ctrader_live_auth::CTraderEnvironment;
        let state = seeded_state();
        let mut old = state.account().await.unwrap();
        old.source_account_id = 9_007_199_254_740_993;
        old.source_environment = CTraderEnvironment::Live;
        old.balance = 125.25;
        let mut current = old.clone();
        current.source_account_id = 99;
        current.source_environment = CTraderEnvironment::Demo;
        current.balance = 750.50;
        for (payload, account_id, environment, balance) in [
            (current, "99", "Demo", 750.50),
            // The late old request must remain labelled old, even after a newer snapshot.
            (old, "9007199254740993", "Live", 125.25),
        ] {
            let wire = serde_json::to_value(AccountSnapshotDto::from(payload)).unwrap();
            assert_eq!(wire["sourceAccountId"], account_id);
            assert_eq!(wire["sourceEnvironment"], environment);
            assert_eq!(wire["balance"], balance);
            assert_eq!(wire["positions"][0]["symbol"], "EURUSD");
        }
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

    #[tokio::test]
    async fn a_financial_truth_failure_is_not_misreported_as_broker_auth_failure() {
        let state = seeded_state();
        let saved = state.account().await.expect("seeded account");
        state
            .set_account_failure(&anyhow::Error::new(
                neoethos_core::BrokerFinancialTruthErrorV1::unavailable_for(
                    neoethos_core::BrokerFinancialOperationV1::LiveRiskAndPnl,
                ),
            ))
            .await;
        let response =
            snapshot_with_scope(state.clone(), || Ok((42, CTraderEnvironment::Demo))).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            payload["code"],
            neoethos_core::BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1
        );
        assert!(
            payload["detail"]
                .as_str()
                .unwrap()
                .contains("broker_position_unrealized_pnl")
        );
        assert!(
            payload.get("balance").is_none(),
            "stale values are not a successful refresh"
        );
        state.set_account(saved).await;
        assert_eq!(
            snapshot_with_scope(state.clone(), || Ok((42, CTraderEnvironment::Demo)))
                .await
                .status(),
            StatusCode::OK
        );
        assert!(state.account_observation().await.1.is_none());
    }

    #[tokio::test]
    async fn snapshot_never_returns_a_previous_account_or_environment() {
        for scope in [
            (99, CTraderEnvironment::Demo),
            (42, CTraderEnvironment::Live),
        ] {
            let response = snapshot_with_scope(seeded_state(), move || Ok(scope)).await;
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert!(value.get("balance").is_none());
        }
    }
}
