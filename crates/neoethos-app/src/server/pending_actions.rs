//! HTTP surface for the LLM trade-management confirmation flow (#136).
//!
//! - `GET  /actions/pending`        → list all known actions (Pending +
//!                                     recent history).
//! - `POST /actions/{id}/confirm`   → user clicked Confirm; mark
//!                                     confirmed + dispatch the actual
//!                                     broker call. Returns the
//!                                     result.
//! - `POST /actions/{id}/reject`    → user clicked Reject; mark
//!                                     rejected with optional reason.
//!
//! The LLM never hits these endpoints. The Flutter UI does. Proposals
//! land in the queue via `pending_actions::propose`, called by whichever
//! AI helper has a Confirm/Reject-required tool (none today; the queue
//! still works as a manual confirm-before-execute surface).

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::app_services::broker_api::close_position_blocking;
use crate::app_services::pending_actions::{
    ActionKind, ActionStatus, list_all, mark_completed, mark_confirmed, mark_rejected,
};

use super::state::AppApiState;

/// `GET /actions/pending` — returns the full queue (live + recent
/// history). Newest first. Operator UI polls this every couple of
/// seconds to surface freshly-proposed actions.
pub async fn list(State(_state): State<AppApiState>) -> Response {
    Json(serde_json::json!({"actions": list_all()})).into_response()
}

#[derive(Debug, Deserialize)]
pub struct ConfirmBody {
    /// Optional override of the action's volume_units. Lets the UI
    /// support partial-close even when the LLM proposed full-close
    /// — operator clicks "Close half" instead of "Confirm".
    #[serde(rename = "volumeUnitsOverride", default)]
    pub volume_units_override: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct RejectBody {
    /// Optional free-form reason the operator typed when rejecting.
    /// Persisted in the audit row so a future AI helper that reads
    /// the journal can adjust its next proposal ("got it, the user
    /// said X").
    #[serde(default)]
    pub reason: Option<String>,
}

/// `POST /actions/{id}/confirm` — flip Pending → Confirmed, then
/// execute the underlying broker call. The broker outcome (Executed
/// vs Failed) is stamped back onto the action so the UI / next
/// /actions/pending response shows the final state.
pub async fn confirm(
    State(_state): State<AppApiState>,
    Path(id): Path<String>,
    body: Option<Json<ConfirmBody>>,
) -> Response {
    let override_volume = body.and_then(|b| b.volume_units_override);

    let snapshot = match mark_confirmed(&id) {
        Ok(s) => s,
        Err(err) => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": err.to_string(),
                    "code": "confirm_failed",
                })),
            )
                .into_response();
        }
    };

    // Dispatch to the broker. Match on ActionKind so the whitelist is
    // explicit at the call site — no `dyn Action::execute` polymorphism
    // that could be sneaked into accepting an unaudited action kind.
    match &snapshot.kind {
        ActionKind::ClosePosition {
            position_id,
            volume_units,
            ..
        } => {
            let pos_id = *position_id;
            let vol = override_volume.unwrap_or(*volume_units);
            // `volume_units == 0` is the LLM-side convention for "close
            // the entire position". The broker requires a real number,
            // so we'd ideally look up the current volume here. For
            // now: when 0, reject the confirm with a hint so the UI
            // can prompt the operator to pick a volume.
            if vol <= 0 {
                let note =
                    "volume_units is 0 — UI must pass volumeUnitsOverride with the broker volume to close".to_string();
                mark_completed(&id, ActionStatus::Failed, note.clone());
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": note,
                        "code": "missing_volume",
                    })),
                )
                    .into_response();
            }
            let result =
                tokio::task::spawn_blocking(move || close_position_blocking(pos_id, vol)).await;
            match result {
                Ok(Ok(outcome)) => {
                    let note = format!(
                        "Broker executed: status={:?} deal_id={:?} net_profit={:?}",
                        outcome.status, outcome.deal_id, outcome.net_profit
                    );
                    mark_completed(&id, ActionStatus::Executed, note);
                    Json(serde_json::json!({
                        "ok": true,
                        "action_id": id,
                        "status": "executed",
                        "broker_outcome": {
                            "deal_id": outcome.deal_id,
                            "execution_price": outcome.execution_price,
                            "net_profit": outcome.net_profit,
                        },
                    }))
                    .into_response()
                }
                Ok(Err(err)) => {
                    let note = format!("broker rejected close: {err}");
                    mark_completed(&id, ActionStatus::Failed, note.clone());
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(serde_json::json!({
                            "error": note,
                            "code": "broker_failed",
                            "action_id": id,
                        })),
                    )
                        .into_response()
                }
                Err(join_err) => {
                    let note = format!("close_position blocking task panicked: {join_err}");
                    mark_completed(&id, ActionStatus::Failed, note.clone());
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": note,
                            "code": "broker_panic",
                            "action_id": id,
                        })),
                    )
                        .into_response()
                }
            }
        }
    }
}

/// `POST /actions/{id}/reject` — flip Pending → Rejected. No broker
/// side effects.
pub async fn reject(
    State(_state): State<AppApiState>,
    Path(id): Path<String>,
    body: Option<Json<RejectBody>>,
) -> Response {
    let reason = body.and_then(|b| b.0.reason);
    match mark_rejected(&id, reason.as_deref()) {
        Ok(snap) => Json(serde_json::json!({
            "ok": true,
            "action": snap,
        }))
        .into_response(),
        Err(err) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": err.to_string(),
                "code": "reject_failed",
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    //! #146: replaces the prior `_shape_check` stub with a real
    //! end-to-end smoke test against `GET /actions/pending`. The
    //! stub was a no-op `fn(_a: PendingAction)` whose only purpose
    //! was to keep the type in scope; it asserted nothing.
    //!
    //! The wire DTO is generated by `serde_json::json!({"actions":
    //! list_all()})` which inlines `Serialize` on `PendingAction`.
    //! Flutter's `LlmActionsBanner` parses by exact field names, so
    //! a silent serde rename or field removal would break the UI.
    //! These assertions pin the contract.

    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::app_services::pending_actions::{ActionKind, clear_for_tests, propose};

    use super::super::state::AppApiState;

    #[tokio::test]
    async fn list_returns_wire_shape_with_pending_close_position() {
        clear_for_tests();
        let _id = propose(
            ActionKind::ClosePosition {
                position_id: 42,
                volume_units: 10_000,
                symbol_hint: Some("EURUSD".to_string()),
            },
            "stop-loss approaching".to_string(),
        )
        .expect("propose ok");

        let app = super::super::router(AppApiState::new());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/actions/pending")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body collects");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");

        let actions = json["actions"].as_array().expect("actions is an array");
        assert!(!actions.is_empty(), "proposed action must appear");
        let a = &actions[0];

        // Flat-shape fields the Flutter banner reads.
        assert!(a["id"].is_string(), "id must be a string");
        assert!(a["reason"].is_string());
        assert!(a["proposed_at_unix_ms"].is_i64());
        assert!(a["expires_at_unix_ms"].is_i64());
        assert!(a["status"].is_string(), "status is enum-as-string");

        // ActionKind uses `#[serde(tag = "kind", rename_all = "snake_case")]`
        // so the variant fields flatten into the parent object with a
        // `kind` discriminator. Verify both the discriminator and the
        // embedded fields survive serialization.
        let kind = &a["kind"];
        assert_eq!(kind["kind"], "close_position");
        assert_eq!(kind["position_id"], 42);
        assert_eq!(kind["volume_units"], 10_000);
        assert_eq!(kind["symbol_hint"], "EURUSD");

        clear_for_tests();
    }
}
