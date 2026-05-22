//! Control endpoints for the Discovery and Training engines.
//!
//! POST /engines/discovery/start  — kick off a discovery job
//! POST /engines/discovery/stop   — request cancellation
//! POST /engines/training/start   — kick off a training job
//! POST /engines/training/stop    — request cancellation
//!
//! Each engine has at most one in-flight job at a time. Starting while
//! one is already running returns 409 Conflict. Stopping when nothing is
//! running returns 200 with `{"running": false}` — idempotent.
//!
//! Engine state ("Idle" / "Running" / "Failed: …" / "Succeeded") is
//! tracked through a `EngineSlot` held inside `AppApiState`. The
//! background task that drives each job drains the `ServiceEvent`
//! channel and writes the latest `JobState` back into the slot, which
//! `/engines/status` then reads.

use anyhow::Result;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use std::path::PathBuf;
use tokio::sync::mpsc;

use crate::app_services::ServiceEvent;
use crate::app_services::discovery::{DiscoveryRequest, start_discovery_job};
use crate::app_services::jobs::{JobKind, JobState};
use crate::app_services::training::{TrainingRequest, start_training_job};

use super::state::AppApiState;

/// Shared request body for `start` endpoints — picks the symbol +
/// timeframe to operate on. Empty fields fall back to "EURUSD" / "M1"
/// so the dashboard "Start" button can fire without any params.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct StartJobBody {
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct StartResponse {
    pub started: bool,
    pub kind: &'static str,
    pub symbol: String,
    pub base_tf: String,
}

#[derive(Debug, serde::Serialize)]
pub struct StopResponse {
    pub running: bool,
    pub kind: &'static str,
}

// ─── Discovery ────────────────────────────────────────────────────────────

pub async fn discovery_start(
    State(state): State<AppApiState>,
    body: Option<Json<StartJobBody>>,
) -> Response {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let symbol = body
        .symbol
        .unwrap_or_else(|| "EURUSD".to_string())
        .trim()
        .to_uppercase();
    let base_tf = body
        .base_tf
        .unwrap_or_else(|| "M1".to_string())
        .trim()
        .to_uppercase();

    if state.engine_state(JobKind::Discovery).await == EngineRunState::Running {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "discovery already running — stop the current job first",
            })),
        )
            .into_response();
    }

    let data_root = match resolve_data_root().await {
        Ok(p) => p,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };

    let request = DiscoveryRequest {
        data_root,
        symbol: symbol.clone(),
        base_tf: base_tf.clone(),
        higher_tfs: vec![
            "M5".to_string(),
            "M15".to_string(),
            "H1".to_string(),
        ],
        config: neoethos_search::DiscoveryConfig::default(),
        prop_firm_rules: neoethos_search::PropFirmRiskRules::default(),
    };

    let (tx, rx) = mpsc::channel::<ServiceEvent>(1000);
    let handle = match start_discovery_job(request, tx) {
        Ok(h) => h,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };

    state
        .install_engine(JobKind::Discovery, handle.cancel.clone())
        .await;
    spawn_state_drainer(state.clone(), JobKind::Discovery, rx);

    Json(StartResponse {
        started: true,
        kind: "discovery",
        symbol,
        base_tf,
    })
    .into_response()
}

pub async fn discovery_stop(State(state): State<AppApiState>) -> Json<StopResponse> {
    let running = state.cancel_engine(JobKind::Discovery).await;
    Json(StopResponse {
        running,
        kind: "discovery",
    })
}

// ─── Training ─────────────────────────────────────────────────────────────

pub async fn training_start(
    State(state): State<AppApiState>,
    body: Option<Json<StartJobBody>>,
) -> Response {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let symbol = body
        .symbol
        .unwrap_or_else(|| "EURUSD".to_string())
        .trim()
        .to_uppercase();
    let base_tf = body
        .base_tf
        .unwrap_or_else(|| "M1".to_string())
        .trim()
        .to_uppercase();

    if state.engine_state(JobKind::Training).await == EngineRunState::Running {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "training already running — stop the current job first",
            })),
        )
            .into_response();
    }

    let request = TrainingRequest {
        config_path: "config.yaml".to_string(),
        models_dir: PathBuf::from("models"),
        symbol: symbol.clone(),
        base_tf: base_tf.clone(),
    };

    let (tx, rx) = mpsc::channel::<ServiceEvent>(1000);
    let handle = match start_training_job(request, tx) {
        Ok(h) => h,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": err.to_string()})),
            )
                .into_response();
        }
    };

    state
        .install_engine(JobKind::Training, handle.cancel.clone())
        .await;
    spawn_state_drainer(state.clone(), JobKind::Training, rx);

    Json(StartResponse {
        started: true,
        kind: "training",
        symbol,
        base_tf,
    })
    .into_response()
}

pub async fn training_stop(State(state): State<AppApiState>) -> Json<StopResponse> {
    let running = state.cancel_engine(JobKind::Training).await;
    Json(StopResponse {
        running,
        kind: "training",
    })
}

// ─── shared helpers ───────────────────────────────────────────────────────

/// Where the engines pull their input data from. Mirrors the wiring
/// the egui main loop used: load `config.yaml` from CWD, take
/// `system.data_dir` from it.
async fn resolve_data_root() -> Result<PathBuf> {
    tokio::task::spawn_blocking(|| {
        let settings = Settings::from_yaml("config.yaml")
            .map_err(|e| anyhow::anyhow!("config.yaml not loadable: {e}"))?;
        Ok(settings.system.data_dir)
    })
    .await
    .map_err(|e| anyhow::anyhow!("blocking task panicked: {e}"))?
}

/// Spawn a background task that drains the ServiceEvent rx channel
/// emitted by the job and reflects the latest `JobState` into the
/// `AppApiState` engine slot. The task exits when the channel closes
/// (job's send end dropped after terminal event).
fn spawn_state_drainer(
    state: AppApiState,
    kind: JobKind,
    mut rx: mpsc::Receiver<ServiceEvent>,
) {
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let snapshot = match (&event, kind) {
                (ServiceEvent::DiscoveryUpdated(s), JobKind::Discovery) => Some(s),
                (ServiceEvent::TrainingUpdated(s), JobKind::Training) => Some(s),
                _ => None,
            };
            let Some(snap) = snapshot else { continue };
            state
                .update_engine(
                    kind,
                    EngineRunState::from(snap.state),
                    snap.report.summary.clone(),
                )
                .await;
        }
        // Channel closed — make sure we don't leave a dangling
        // "Running" state if the producer side dropped without a
        // terminal event (shouldn't happen, defensive guard).
        state.finalize_engine_if_running(kind).await;
    });
}

// ─── EngineRunState (wire-friendly subset of JobState) ────────────────────

/// Compact engine state for `/engines/status`. We collapse Queued and
/// Running into the same "Running" label (the UI only cares whether
/// it should show a green dot + a "Stop" button), and Degraded into
/// Succeeded (still a terminal-OK outcome).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineRunState {
    Idle,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl EngineRunState {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineRunState::Idle => "Idle",
            EngineRunState::Running => "Running",
            EngineRunState::Succeeded => "Succeeded",
            EngineRunState::Failed => "Failed",
            EngineRunState::Cancelled => "Cancelled",
        }
    }
}

impl From<JobState> for EngineRunState {
    fn from(value: JobState) -> Self {
        match value {
            JobState::Queued | JobState::Running => EngineRunState::Running,
            JobState::Succeeded | JobState::Degraded => EngineRunState::Succeeded,
            JobState::Failed => EngineRunState::Failed,
            JobState::Cancelled => EngineRunState::Cancelled,
        }
    }
}
