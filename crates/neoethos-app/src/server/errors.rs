//! Shared, user-facing error responses for the HTTP API (F-347).
//!
//! Most route handlers used to answer failures with
//! `(StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":
//! err.to_string()})))` — dumping a raw anyhow chain ("No such file or
//! directory (os error 2)", "missing column 'close'", "settings load
//! panicked") that means nothing to an operator. This module gives every
//! handler one helper that:
//!   1. leads with a friendly, actionable message (what to do + where),
//!   2. keeps the raw error in a `detail` field for logs / copy-to-support,
//!   3. attaches the cTrader `translation` block when the underlying error
//!      carries a recognised broker code, so the UI renders the right CTA
//!      (Re-authenticate, Open Settings, …).
//!
//! The Flutter side already prefers `translation.message`, then the
//! `error` field, via `describeError()` — so producer and consumer stay
//! in lockstep, and these messages flow straight into the UI banners.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Friendly error response for a failed `anyhow` operation.
///
/// `friendly` is the actionable, jargon-free message the UI shows;
/// `status` is used only when the error is NOT a recognised broker error
/// — broker-coded errors answer `502` with their translation block so the
/// UI can render the matching call-to-action.
pub fn actionable_error(
    status: StatusCode,
    friendly: impl Into<String>,
    err: &anyhow::Error,
) -> Response {
    let raw = err.to_string();
    if let Some(t) = crate::app_services::ctrader_errors::translate_anyhow(err) {
        return (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "error": t.message,
                "detail": raw,
                "translation": t,
            })),
        )
            .into_response();
    }
    (
        status,
        Json(serde_json::json!({
            "error": friendly.into(),
            "detail": raw,
        })),
    )
        .into_response()
}

/// Friendly response for an internal panic (`JoinError` — a blocking task
/// crashed). Users can't fix these, so point them at restart + the report
/// flow and keep the raw cause in `detail` for the diagnostic bundle.
pub fn internal_panic(context: &str, cause: impl std::fmt::Display) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": format!(
                "{context} crashed unexpectedly. Please restart the app; if it \
                 keeps happening, use Help → Report Issue to send us the logs."
            ),
            "detail": cause.to_string(),
        })),
    )
        .into_response()
}
