//! `/watchlist` — editable Market Watch symbol list (F-338, Feature #12).
//!
//! The operator curates which symbols the live spot stream subscribes to
//! from the Flutter Market Watch screen. That set is persisted as
//! `system.watchlist` in `config.yaml`.
//!
//! - GET  → `{ "symbols": [String] }` — the current saved watchlist
//!         (empty vec when unset).
//! - POST → body `{ "symbols": [String] }`. The entries are normalised
//!         (uppercased, trimmed, blanks dropped, de-duplicated while
//!         preserving first-seen order), written back to
//!         `system.watchlist`, and the live spot streamer is restarted so
//!         the change takes effect within ~5 s WITHOUT an app restart
//!         (see [`crate::app_services::live_spots_streamer::restart_streamer`]).
//!         Responds `{ "saved": N, "symbols": [...], "restarted": bool }`.
//!
//! Persistence reuses the same load-mutate-save path the typed
//! `POST /settings` handler uses (`Settings::from_yaml` → mutate →
//! `Settings::save`), so the other 200+ config knobs are preserved.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;

use super::errors::actionable_error;
use super::state::AppApiState;

/// Request body for `POST /watchlist`.
#[derive(Debug, serde::Deserialize)]
pub struct WatchlistUpdate {
    pub symbols: Vec<String>,
}

/// `GET /watchlist` — return the current saved Market Watch set.
///
/// Reads `Settings::from_yaml(state.config_path()).system.watchlist`,
/// answering `{ "symbols": [] }` when the key is unset. A config-load
/// failure surfaces via [`actionable_error`] so the UI shows an
/// actionable banner instead of a raw anyhow chain.
pub async fn get(State(state): State<AppApiState>) -> Response {
    let config_path = state.config_path().to_path_buf();
    let loaded = tokio::task::spawn_blocking(move || Settings::from_yaml(&config_path)).await;

    match loaded {
        Ok(Ok(settings)) => Json(serde_json::json!({
            "symbols": settings.system.watchlist,
        }))
        .into_response(),
        Ok(Err(err)) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not read the Market Watch list. Check that config.yaml is present \
             and valid in Settings, then reload.",
            &err,
        ),
        Err(join_err) => super::errors::internal_panic("Loading the Market Watch list", join_err),
    }
}

/// `POST /watchlist` — replace the Market Watch set + re-subscribe live.
///
/// Pipeline:
///   1. Normalise the incoming symbols — uppercase, trim, drop blanks,
///      de-duplicate (first occurrence wins, order preserved).
///   2. Load the current `Settings`, set `system.watchlist`, save back to
///      `config.yaml` (preserving every other knob).
///   3. Restart the live spot streamer so the new set takes effect
///      within ~5 s without an app restart.
///
/// Responds `{ "saved": N, "symbols": [...], "restarted": bool }`, where
/// `restarted` reflects whether a fresh streamer could be spawned (it is
/// `false` when broker creds/token are missing or the broker is
/// unreachable — the watchlist is still saved and the old streamer still
/// stops in that case).
pub async fn post(
    State(state): State<AppApiState>,
    Json(payload): Json<WatchlistUpdate>,
) -> Response {
    // (1) Normalise: uppercase + trim, drop blanks, dedupe preserving
    // first-seen order. A plain `HashSet` would lose the operator's
    // ordering, so we track seen entries explicitly.
    let mut symbols: Vec<String> = Vec::with_capacity(payload.symbols.len());
    let mut seen = std::collections::HashSet::new();
    for raw in payload.symbols {
        let normalised = raw.trim().to_uppercase();
        if normalised.is_empty() {
            continue;
        }
        if seen.insert(normalised.clone()) {
            symbols.push(normalised);
        }
    }

    // (2) Persist via the same load-mutate-save path POST /settings uses.
    let config_path = state.config_path().to_path_buf();
    let symbols_to_save = symbols.clone();
    let save_result = tokio::task::spawn_blocking(move || {
        let mut settings = Settings::from_yaml(&config_path)?;
        settings.system.watchlist = symbols_to_save;
        settings.save(&config_path)?;
        anyhow::Ok(())
    })
    .await;

    match save_result {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            tracing::error!(
                target: "neoethos_app::server::watchlist",
                error = %err,
                "failed to persist watchlist to config.yaml"
            );
            return actionable_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "The Market Watch list could not be saved. Close any editor that may have \
                 config.yaml open and make sure the folder is writable, then try again.",
                &err,
            );
        }
        Err(join_err) => {
            return super::errors::internal_panic("Saving the Market Watch list", join_err);
        }
    }

    // (3) Re-subscribe the live stream. `restart_streamer` does blocking
    // broker I/O (lists symbols to resolve ids), so it runs on the
    // blocking pool. It bumps the stream generation unconditionally — so
    // even if the new streamer can't spawn, the old one stops rather than
    // keep streaming the stale symbol set.
    let restarted = tokio::task::spawn_blocking(
        crate::app_services::live_spots_streamer::restart_streamer,
    )
    .await
    .unwrap_or(false);

    tracing::info!(
        target: "neoethos_app::server::watchlist",
        saved = symbols.len(),
        restarted,
        "watchlist updated via POST /watchlist"
    );

    Json(serde_json::json!({
        "saved": symbols.len(),
        "symbols": symbols,
        "restarted": restarted,
    }))
    .into_response()
}
