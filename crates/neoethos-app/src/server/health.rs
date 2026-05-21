//! `/healthz` — single-byte liveness check.
//!
//! The Flutter client polls this on startup (200 ms timeout) to decide
//! whether the bundled Rust process is ready before showing the main
//! window. If the server can't even answer this, something is very wrong.

use axum::Json;
use axum::extract::State;

use super::state::AppApiState;

#[derive(serde::Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    /// Crate version baked in at compile time. Useful for the Flutter
    /// client to detect mismatched bundles (UI says 0.4.21 but server
    /// is still 0.4.20).
    pub version: &'static str,
}

pub async fn healthz(State(_state): State<AppApiState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        version: env!("CARGO_PKG_VERSION"),
    })
}
