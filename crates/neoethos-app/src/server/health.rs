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
    // REMOVED 2026-08-09 (dead-code purge, batch D2): `launched_by_flutter`.
    // Its sole intended consumer was Flutter's `BackendSupervisor`, deleted in
    // the 2026-06-22 Tauri migration. The desktop shell now runs the backend
    // in-process on an ephemeral port (`desktop/src-tauri/src/lib.rs:31`
    // binds 127.0.0.1:0), so the port-7423-ownership problem this field
    // encoded no longer exists. No client ever read it: grep for
    // `launchedByFlutter` across desktop/src, crates/neoethos-mcp,
    // crates/neoethos-cli, mesh/ and mcp/ returned zero.
}

pub async fn healthz(State(_state): State<AppApiState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        version: env!("CARGO_PKG_VERSION"),
    })
}
