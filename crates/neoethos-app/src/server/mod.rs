//! HTTP API surface that the Flutter front-end talks to.
//!
//! Backend HTTP surface for the Flutter migration. The goal of this module
//! is to expose TradingSession state and broker actions over stable JSON so
//! a thin Flutter client can render the UI.
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
pub mod codex;
pub mod data_control;
pub mod diagnostics;
pub mod engines_control;
pub mod hardware;
pub mod health;
pub mod indicators;
pub mod intelligence;
pub mod live_spots;
pub mod orders;
pub mod pending_actions;
pub mod risk;
pub mod settings;
pub mod state;
pub mod system_status;

use anyhow::Context;
use axum::Router;
use axum::routing::{get, post};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
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
        .route("/risk/preset", post(risk::update_preset))
        .route(
            "/settings",
            get(settings::settings).post(settings::update_settings),
        )
        // #193: raw config.yaml for the Flutter Settings "Advanced"
        // panel that surfaces knobs the typed /settings DTO can't list.
        .route("/settings/raw", get(settings::settings_raw_yaml))
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
        .route(
            "/broker/credentials",
            get(broker_control::credentials_get).post(broker_control::credentials_post),
        )
        .route("/broker/symbols", get(data_control::symbols))
        .route("/broker/timeframes", get(data_control::timeframes))
        .route("/broker/accounts", get(data_control::accounts))
        .route("/data/bootstrap", get(system_status::data_bootstrap))
        .route("/data/fetch", post(data_control::fetch))
        // #192: import user-provided CSV/Parquet/Arrow/JSON/JSONL/TSV
        // files into the canonical Vortex layout. Routes through
        // `neoethos_data::convert_to_vortex` for the actual conversion.
        .route("/data/import", post(data_control::import_file))
        .route("/orders", post(orders::place))
        .route("/orders/cancel", post(orders::cancel_order))
        .route("/positions/close", post(orders::close_position))
        // #204 ChatGPT subscription via Codex CLI OAuth flow.
        // Replaces the previous local-Gemma path. Status renders the
        // UI badge, start kicks off PKCE, logout wipes the token,
        // chat proxies requests through the subscription bearer.
        .route("/auth/codex/status", get(codex::status))
        .route("/auth/codex/start", post(codex::start))
        .route("/auth/codex/logout", post(codex::logout))
        .route("/codex/chat", post(codex::chat))
        .route("/intelligence", get(intelligence::intelligence))
        // Live tick stream (#137). Reads from the cache that the
        // long-running spot streamer populates; sub-2s freshness
        // for active majors.
        .route("/live/spots", get(live_spots::list))
        .route("/chart", get(chart::chart))
        .route("/indicators", get(indicators::indicators))
        .route("/diagnostics/report", post(diagnostics::report))
        // ── Trade-management confirmation flow (#136) ──────────
        .route("/actions/pending", get(pending_actions::list))
        .route("/actions/{id}/confirm", post(pending_actions::confirm))
        .route("/actions/{id}/reject", post(pending_actions::reject))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}

/// Hard-coded fallback bind address. Constructed from primitives so the
/// compiler can verify it at build time — no runtime `.parse()`, no
/// `.unwrap()`/`.expect()` that could surprise us on a future refactor.
/// The port is mirrored in `lib/api/backend_client.dart`; if either side
/// changes, both must change in the same commit.
const DEFAULT_BIND_ADDR: SocketAddr =
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 7423);

/// Resolve the bind address: env-var override or [`DEFAULT_BIND_ADDR`].
fn default_bind_addr() -> SocketAddr {
    if let Ok(raw) = std::env::var("NEOETHOS_SERVER_BIND") {
        if let Ok(parsed) = raw.parse::<SocketAddr>() {
            return parsed;
        }
        tracing::warn!(
            target: "neoethos_app::server",
            raw = %raw,
            fallback = %DEFAULT_BIND_ADDR,
            "NEOETHOS_SERVER_BIND set but unparseable; falling back to default"
        );
    }
    DEFAULT_BIND_ADDR
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
