//! Broker control endpoints.
//!
//! POST /broker/reauth — kick off the full cTrader OAuth flow. Opens
//! a browser window, captures the loopback callback, exchanges the
//! auth code for a token bundle, persists it to the keyring. Blocks
//! the HTTP response until the flow either completes or fails
//! (typical wall-clock time: 10–30 s depending on how fast the
//! operator clicks "Continue" in the consent screen).
//!
//! The bridge picks up the new token automatically on its next 5 s
//! refresh — no server restart needed.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::app_services::reauth::run_reauth_flow_blocking;

use super::state::AppApiState;

pub async fn reauth(State(_state): State<AppApiState>) -> Response {
    // run_reauth_flow_blocking() does sync filesystem + reqwest::blocking
    // + std::net listener I/O. We MUST hop to spawn_blocking — calling
    // it directly on the tokio runtime would either panic on drop
    // ("Cannot drop a runtime in a context where blocking is not
    // allowed") or block the reactor for the full duration of the OAuth
    // flow, stalling every other route.
    match tokio::task::spawn_blocking(run_reauth_flow_blocking).await {
        Ok(Ok(outcome)) => Json(outcome).into_response(),
        Ok(Err(err)) => {
            tracing::warn!(
                target: "neoethos_app::server::broker_control",
                error = %err,
                "POST /broker/reauth: OAuth flow failed"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": err.to_string(),
                })),
            )
                .into_response()
        }
        Err(join_err) => {
            tracing::error!(
                target: "neoethos_app::server::broker_control",
                error = %join_err,
                "POST /broker/reauth: blocking task panicked"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("reauth task panicked: {join_err}"),
                })),
            )
                .into_response()
        }
    }
}

