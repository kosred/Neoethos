//! HTTP API surface that the Flutter front-end talks to.
//!
//! Phase 1 of the egui → Flutter migration (task #87). The goal of this
//! module is **not** to replicate every egui panel — it is to expose the
//! same TradingSession state and broker actions over a stable JSON +
//! Server-Sent-Events surface so a thin Flutter client can render the UI.
//!
//! ## Layering
//!
//! - `mod.rs` — router + `serve()` entry point. Owns the axum app.
//! - `state.rs` — `AppApiState`, the `Arc<Mutex<...>>`-wrapped handle
//!   to the long-lived `TradingSession`. All routes pull through this.
//! - `account.rs` — `/account/snapshot` route + DTO.
//! - `health.rs` — `/healthz` route, no app state needed.
//!
//! ## Port
//!
//! The Flutter client (`experiments/forex-flutter-ui/lib/api/
//! backend_client.dart`) hard-codes `http://127.0.0.1:7423`. We bind there
//! by default. Override with the `NEOETHOS_SERVER_BIND` env var
//! (`host:port` form) when running multiple instances on the same machine.
//!
//! ## CORS
//!
//! Flutter desktop binaries open a native window and don't need CORS, but
//! `flutter run -d chrome` (used for hot-reload dev) does. We allow any
//! origin for now — the server is loopback-only so the surface is small.
//! Tighten before exposing on non-loopback interfaces.

pub mod account;
pub mod bridge;
pub mod broker_control;
pub mod chart;
pub mod engines_control;
pub mod hardware;
pub mod intelligence;
pub mod health;
pub mod risk;
pub mod settings;
pub mod state;
pub mod system_status;

use anyhow::Context;
use axum::Router;
use axum::routing::{get, post};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use self::state::AppApiState;

/// Build the axum router. Kept as a free function so tests can mount
/// individual routes against a mock `AppApiState` without going through
/// the actual TCP bind.
pub fn router(state: AppApiState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/healthz", get(health::healthz))
        .route("/account/snapshot", get(account::snapshot))
        .route("/hardware", get(hardware::hardware))
        .route("/risk", get(risk::risk))
        .route("/settings", get(settings::settings))
        .route("/engines/status", get(system_status::engines))
        .route(
            "/engines/discovery/start",
            post(engines_control::discovery_start),
        )
        .route(
            "/engines/discovery/stop",
            post(engines_control::discovery_stop),
        )
        .route(
            "/engines/training/start",
            post(engines_control::training_start),
        )
        .route(
            "/engines/training/stop",
            post(engines_control::training_stop),
        )
        .route("/broker/status", get(system_status::broker_status))
        .route("/broker/reauth", post(broker_control::reauth))
        .route("/data/bootstrap", get(system_status::data_bootstrap))
        .route("/intelligence", get(intelligence::intelligence))
        .route("/chart", get(chart::chart))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}

/// Resolve the bind address: env-var override or 127.0.0.1:7423.
fn default_bind_addr() -> SocketAddr {
    if let Ok(raw) = std::env::var("NEOETHOS_SERVER_BIND") {
        if let Ok(parsed) = raw.parse::<SocketAddr>() {
            return parsed;
        }
        tracing::warn!(
            target: "neoethos_app::server",
            raw = %raw,
            "NEOETHOS_SERVER_BIND set but unparseable; falling back to 127.0.0.1:7423"
        );
    }
    "127.0.0.1:7423".parse().expect("hard-coded default must parse")
}

/// Bind the HTTP listener and serve until the process is killed. The
/// returned future runs forever in the happy path; an error means the
/// bind failed (port already in use, EACCES, etc.).
pub async fn serve(state: AppApiState) -> anyhow::Result<()> {
    let addr = default_bind_addr();
    let app = router(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind axum HTTP server on {addr}"))?;

    tracing::info!(
        target: "neoethos_app::server",
        bind_addr = %addr,
        "NeoEthos HTTP server listening — Flutter client should connect here"
    );

    axum::serve(listener, app)
        .await
        .context("axum::serve returned with an unrecoverable error")
}
