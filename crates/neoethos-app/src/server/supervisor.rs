//! `/supervisor/*` — control surface for the autonomous LLM supervisor.
//!
//! GET  /supervisor/status  — config + recent action log (the UI panel)
//! POST /supervisor/config  — enable/disable + interval (persisted)
//! POST /supervisor/tick    — run one cycle NOW (manual trigger)

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::errors::actionable_error;
use super::state::AppApiState;
use crate::app_services::supervisor;

pub async fn status(State(_state): State<AppApiState>) -> Response {
    let cfg = supervisor::load_config();
    let log = supervisor::recent_log(50);
    Json(serde_json::json!({ "config": cfg, "log": log })).into_response()
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigBody {
    pub enabled: Option<bool>,
    pub interval_minutes: Option<u64>,
    pub max_actions_per_tick: Option<usize>,
}

pub async fn update_config(
    State(_state): State<AppApiState>,
    Json(body): Json<ConfigBody>,
) -> Response {
    let mut cfg = supervisor::load_config();
    if let Some(e) = body.enabled {
        cfg.enabled = e;
    }
    if let Some(m) = body.interval_minutes {
        cfg.interval_minutes = m.clamp(5, 240);
    }
    if let Some(n) = body.max_actions_per_tick {
        cfg.max_actions_per_tick = n.clamp(1, 5);
    }
    match supervisor::save_config(&cfg) {
        Ok(()) => Json(serde_json::json!({ "config": cfg })).into_response(),
        Err(e) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not save supervisor.json — check the data folder is writable.",
            &e,
        ),
    }
}

pub async fn tick(State(state): State<AppApiState>) -> Response {
    match supervisor::tick(state).await {
        Ok(summary) => Json(serde_json::json!({ "summary": summary })).into_response(),
        Err(e) => actionable_error(
            StatusCode::BAD_GATEWAY,
            "Supervisor tick failed — make sure the AI Desk is signed in (ChatGPT) \
             and try again.",
            &e,
        ),
    }
}
