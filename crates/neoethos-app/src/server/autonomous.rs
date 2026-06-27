//! `POST /autonomous/replay` — offline dry-run of the autonomous-trader engine
//! over on-disk history.
//!
//! Returns the SAME `EngineStats` the CLI `trader-replay` command prints, from
//! the SAME `neoethos_trader::replay_symbol_from_dir` helper — so the two
//! front-ends are byte-identical (the UI↔CLI parity mandate, applied to the
//! trader from day one). ZERO broker calls: the engine runs the mock execution
//! adapter over replayed bars.
//!
//! Symbol/base resolve through the shared `SystemConfig` resolvers — exactly as
//! `/engines/discovery/start` does — so an omitted field defaults from
//! `config.yaml` identically to the CLI.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use neoethos_core::Settings;

use crate::app_services::live_trading::{LiveTradingStatus, StartRequest};
use super::errors::actionable_error;
use super::state::AppApiState;

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct ReplayBody {
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
}

pub async fn replay(State(state): State<AppApiState>, body: Option<Json<ReplayBody>>) -> Response {
    let body = body.map(|Json(b)| b).unwrap_or_default();

    let settings = Settings::from_yaml(state.config_path()).ok();
    let symbol = body
        .symbol
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.as_ref().map(|s| s.system.resolve_symbol()))
        .unwrap_or_default();
    let base_tf = body
        .base_tf
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .or_else(|| settings.as_ref().map(|s| s.system.resolve_base_timeframe()))
        .unwrap_or_default();

    if symbol.is_empty() || base_tf.is_empty() {
        return actionable_error(
            StatusCode::BAD_REQUEST,
            "Replay can't run — no symbol / base timeframe was supplied and config.yaml \
             couldn't provide a default. Set them in Settings or include them in the request.",
            &anyhow::anyhow!("symbol='{symbol}' base_tf='{base_tf}'"),
        );
    }

    let data_dir = match settings.as_ref() {
        Some(s) => s.system.data_dir.clone(),
        None => {
            return actionable_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Replay can't run — config.yaml couldn't be read to locate the data folder.",
                &anyhow::anyhow!("settings unavailable"),
            );
        }
    };

    // The replay reads + crunches a whole history synchronously — keep it off the
    // async runtime's worker threads.
    let sym = symbol.clone();
    let base = base_tf.clone();
    let result = tokio::task::spawn_blocking(move || {
        neoethos_trader::replay_symbol_from_dir(
            &data_dir,
            &sym,
            &base,
            neoethos_trader::EngineConfig::default(),
        )
    })
    .await;

    match result {
        Ok(Ok(stats)) => Json(stats).into_response(),
        Ok(Err(err)) => actionable_error(
            StatusCode::BAD_REQUEST,
            "Replay failed — make sure the data folder has this symbol + base timeframe \
             (run Data Bootstrap or import a file first).",
            &err,
        ),
        Err(join_err) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "The replay task panicked.",
            &anyhow::anyhow!("{join_err}"),
        ),
    }
}

// ── Live autonomous trading ───────────────────────────────────────────────────

/// `POST /autonomous/start` — begin live trading from a discovered portfolio.
///
/// Body: `StartRequest` JSON (portfolio_path required; lot_size, sl/tp optional).
/// Returns 409 if already running, 200 with the initial status on success.
pub async fn start_live(State(state): State<AppApiState>, Json(req): Json<StartRequest>) -> Response {
    let mut slot = match state.live_trading.lock() {
        Ok(g) => g,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "state lock poisoned").into_response(),
    };

    if slot.as_ref().map(|h| h.is_running()).unwrap_or(false) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "live trading is already running — POST /autonomous/stop first"
            })),
        )
            .into_response();
    }

    match crate::app_services::live_trading::start(req) {
        Ok(handle) => {
            let status = handle.snapshot();
            *slot = Some(handle);
            (StatusCode::OK, Json(status)).into_response()
        }
        Err(e) => actionable_error(
            StatusCode::BAD_REQUEST,
            "Failed to start live trading. Check the portfolio_path and broker credentials.",
            &e,
        ),
    }
}

/// `POST /autonomous/stop` — gracefully stop the live trading loop.
pub async fn stop_live(State(state): State<AppApiState>) -> Response {
    let slot = match state.live_trading.lock() {
        Ok(g) => g,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "state lock poisoned").into_response(),
    };
    match slot.as_ref() {
        Some(handle) => {
            handle.stop();
            (StatusCode::OK, Json(serde_json::json!({"stopped": true}))).into_response()
        }
        None => (
            StatusCode::OK,
            Json(serde_json::json!({"stopped": false, "reason": "was not running"})),
        )
            .into_response(),
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct GateQuery {
    pub portfolio: String,
}

/// `GET /autonomous/gate?portfolio=...` — demo forward-test eligibility for a
/// portfolio, so the UI can show WHY live is (not) yet allowed BEFORE the
/// operator clicks Start. `enforced` is true only on a Live (real-money) env;
/// on Demo the gate is informational (eligibility still tracked, never blocks).
pub async fn gate(Query(q): Query<GateQuery>) -> Response {
    let portfolio = q.portfolio;
    let env_is_live = crate::app_services::live_gate::active_env_is_live();
    let result = tokio::task::spawn_blocking(move || {
        crate::app_services::live_gate::evaluate_for_portfolio(&portfolio)
    })
    .await;
    match result {
        Ok(Ok(decision)) => Json(serde_json::json!({
            "envIsLive": env_is_live,
            "enforced": env_is_live,
            "eligible": decision.eligible,
            "summary": decision.summary,
            "criteria": decision.criteria,
        }))
        .into_response(),
        Ok(Err(e)) => actionable_error(
            StatusCode::BAD_REQUEST,
            "Couldn't evaluate the demo forward-test gate — make sure the portfolio path \
             and its sibling *.quality.json exist.",
            &e,
        ),
        Err(join_err) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "The gate evaluation task panicked.",
            &anyhow::anyhow!("{join_err}"),
        ),
    }
}

/// `GET /autonomous/status` — poll live trading state.
pub async fn live_status(State(state): State<AppApiState>) -> Response {
    let slot = match state.live_trading.lock() {
        Ok(g) => g,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "state lock poisoned").into_response(),
    };
    let status: LiveTradingStatus = slot
        .as_ref()
        .map(|h| h.snapshot())
        .unwrap_or_default();
    Json(status).into_response()
}
