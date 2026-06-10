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
pub mod autonomous;
pub mod bridge;
pub mod broker_control;
pub mod chart;
// **2026-05-25 — operator directive "live tick + chart switch
// must be immediate"**: in-memory LRU cache for `ChartDto` so
// repeat timeframe switches don't re-read the 30 MB Vortex file.
// Mirrors the in-RAM series cache that TradingView / cTrader /
// MT5 all use server-side or client-side.
pub mod chart_cache;
pub mod codex;
pub mod data_control;
pub mod errors;
pub mod diagnostics;
pub mod engines_control;
pub mod hardware;
pub mod health;
pub mod indicators;
pub mod intelligence;
pub mod journal;
pub mod knob_catalog;
pub mod live_spots;
pub mod news;
pub mod orders;
pub mod pending_actions;
pub mod risk;
pub mod risky;
pub mod settings;
pub mod state;
pub mod strategy_lab;
pub mod system_status;
pub mod watchlist;

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
        // **2026-05-25 — operator directive "uniform push everywhere"**:
        // SSE push channel mirror of `/live/spots/stream` for account
        // updates. Bridge writes a fresh snapshot every 5 s (and in
        // future on every `OAExecutionEvent` push); subscribers receive
        // each update with ~5 ms latency vs. the previous 1000 ms poll
        // of `/account/snapshot`.
        .route("/account/snapshot/stream", get(account::stream))
        // **2026-05-25 — uniform-push doctrine**: force-refresh
        // trigger that bridges the 5 s safety poll. Operator clicks
        // a "refresh" button → POST here → bridge skips the timer
        // and runs `refresh_once` immediately → fresh snapshot
        // broadcast within ~750 ms. Same channel the future
        // `OAExecutionEvent` handler will use.
        .route("/account/snapshot/refresh", post(account::refresh))
        .route("/hardware", get(hardware::hardware))
        // The AI news desk: public no-API-key RSS headlines fetched
        // server-side + a Codex (ChatGPT subscription) market briefing.
        // `?force=true` bypasses the fetch-coalescing cache.
        .route("/news/feed", get(news::feed))
        .route("/risk", get(risk::risk))
        .route("/risk/preset", post(risk::update_preset))
        // Risky/Growth Mode time-to-target projection, computed by the
        // live engine's own math (no hardcoded growth rates in the UI).
        .route("/risky/scenarios", get(risky::scenarios))
        .route(
            "/settings",
            get(settings::settings).post(settings::update_settings),
        )
        // #193: raw config.yaml for the Flutter Settings "Advanced"
        // panel that surfaces knobs the typed /settings DTO can't list.
        //
        // F-312 (2026-05-29): POST handler on the same route writes the
        // whole YAML verbatim, closing the silent-drop hole where the
        // typed `POST /settings` DTO dropped edits to any of the 200+
        // fields outside its 5-field allowlist.
        .route(
            "/settings/raw",
            get(settings::settings_raw_yaml).post(settings::update_settings_raw_yaml),
        )
        // **2026-05-25 — operator-approved**: the Flutter "Advanced
        // Settings" screen consumes this catalog to render every
        // runtime knob with help text + presets. Catalog is the
        // machine-readable counterpart of `docs/CONFIG-KNOBS-REFERENCE.md`.
        .route(
            "/settings/knob-catalog",
            get(knob_catalog::get_knob_catalog),
        )
        .route("/settings/presets", get(knob_catalog::get_presets))
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
        // F-330 Strategy Lab — Promotion Gate status + promote-to-live.
        .route(
            "/strategy_lab/promotion",
            get(strategy_lab::promotion_status),
        )
        .route("/strategy_lab/promote", post(strategy_lab::promote))
        // Autonomous trader (Phase 1.5): offline dry-run over on-disk history,
        // the SAME engine + helper the CLI `trader-replay` drives → identical
        // EngineStats from both front-ends.
        .route("/autonomous/replay", post(autonomous::replay))
        .route("/broker/status", get(system_status::broker_status))
        .route("/broker/reauth", post(broker_control::reauth))
        .route(
            "/broker/credentials",
            get(broker_control::credentials_get).post(broker_control::credentials_post),
        )
        .route("/broker/symbols", get(data_control::symbols))
        .route("/broker/timeframes", get(data_control::timeframes))
        .route("/broker/accounts", get(data_control::accounts))
        // F-333: set the *active* cTrader account by promoting it to the
        // front of broker_credentials.toml's accounts list (resolve_creds
        // reads accounts.first()). MVP — takes effect on next start.
        .route(
            "/broker/account/select",
            post(broker_control::account_select),
        )
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
        // **2026-05-25 — operator directive "push, not poll"**:
        // SSE endpoint that streams ticks to Flutter as they arrive
        // (target latency ~5 ms vs. 1000 ms for the polling route
        // above). The polling route is kept as a fallback for HTTP
        // clients without SSE support + for the cold-start snapshot.
        .route("/live/spots/stream", get(live_spots::stream))
        // F-338 (Feature #12): editable Market Watch watchlist. GET
        // returns the saved `system.watchlist`; POST replaces it and
        // re-subscribes the live spot stream within ~5 s (no restart).
        .route("/watchlist", get(watchlist::get).post(watchlist::post))
        .route("/chart", get(chart::chart))
        // Scroll-back pagination: older bars on demand, broker-only, never
        // persisted (TradingView model — panning back costs zero disk).
        .route("/chart/history", get(chart::chart_history))
        .route("/indicators", get(indicators::indicators))
        // Trade journal (#flagship): closed-trade log + computed stats
        // (myfxbook-style). Reads the JSONL store under <data_dir>/journal/.
        .route("/journal/trades", get(journal::trades))
        .route("/journal/stats", get(journal::stats))
        .route("/diagnostics/report", post(diagnostics::report))
        // ── Trade-management confirmation flow (#136) ──────────
        .route("/actions/pending", get(pending_actions::list))
        .route("/actions/{id}/confirm", post(pending_actions::confirm))
        .route("/actions/{id}/reject", post(pending_actions::reject))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}

/// **F-527/F-CORE3 closure (2026-05-25)**: the hard-coded fallback bind
/// address + the env-var resolution now live in the canonical
/// `app_services::env_overrides` registry so the operator can grep one
/// file to find every NeoEthos env knob. This shim keeps the
/// `default_bind_addr()` callsite stable.
///
/// The port is mirrored in `lib/api/backend_client.dart`; if either
/// side changes, both must change in the same commit.
fn default_bind_addr() -> SocketAddr {
    crate::app_services::env_overrides::server_bind_addr()
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

    // 2026-06-10: the trading endpoints (/orders, /positions/close,
    // /broker/credentials, …) carry NO authentication — the security model
    // is "loopback-only", enforced solely by the bind address. Make that
    // assumption explicit in the logs, and SHOUT if the operator ever points
    // the bind at a non-loopback interface, where those endpoints would become
    // reachable (and trade-capable) from the network with no auth in front.
    if addr.ip().is_loopback() {
        tracing::info!(
            target: "neoethos_app::server",
            bind_addr = %addr,
            "server is loopback-only (no endpoint authentication by design — \
             do not expose this port on a public interface)"
        );
    } else {
        tracing::warn!(
            target: "neoethos_app::server",
            bind_addr = %addr,
            "SECURITY: HTTP server bound to a NON-loopback address — the trading \
             endpoints have NO authentication and are now reachable from the \
             network. Anyone who can reach this port can place/close live trades \
             and change broker credentials. Bind to 127.0.0.1 unless you have put \
             your own auth/firewall in front."
        );
    }

    axum::serve(listener, app)
        .await
        .context("axum::serve returned with an unrecoverable error")
}
