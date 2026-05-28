//! `/healthz` — single-byte liveness check.
//!
//! The Flutter client polls this on startup (200 ms timeout) to decide
//! whether the bundled Rust process is ready before showing the main
//! window. If the server can't even answer this, something is very wrong.

use axum::Json;
use axum::extract::State;

use super::state::{AppApiState, launched_by_flutter};

#[derive(serde::Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    /// Crate version baked in at compile time. Useful for the Flutter
    /// client to detect mismatched bundles (UI says 0.4.21 but server
    /// is still 0.4.20).
    pub version: &'static str,
    /// F-270 (2026-05-28): true iff this backend was spawned by a
    /// Flutter supervisor (`--launched-by-flutter` CLI flag set).
    /// The Flutter `BackendSupervisor.ensureRunning()` reads this to
    /// distinguish "another NeoEthos UI owns this backend" (refuse
    /// second launch) from "a stale backend is holding port 7423 with
    /// no UI" (attach to it instead of exiting).
    pub launched_by_flutter: bool,
}

pub async fn healthz(State(_state): State<AppApiState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        version: env!("CARGO_PKG_VERSION"),
        launched_by_flutter: launched_by_flutter(),
    })
}
