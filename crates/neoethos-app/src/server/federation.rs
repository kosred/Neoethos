//! HTTP surface for Federation Phase 0 (`app_services::federation`).
//!
//! Coordinator endpoints (`/federation/{jobs,job,submit,status}`) are meant
//! to be exposed to trusted peers via a tunnel (Tailscale serve, port
//! forward, ngrok) — the optional shared token gates the worker-facing pair.
//! Worker endpoints (`/federation/worker/*`) are local operator controls.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::app_services::federation as fed;

use super::state::AppApiState;

fn provided_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-fed-token")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

pub async fn fed_status() -> Response {
    Json(fed::status()).into_response()
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetJobsBody {
    pub combos: Vec<fed::FedJob>,
    /// Optional shared secret; when set, workers must send `x-fed-token`.
    pub token: Option<String>,
}

/// Operator (coordinator role): replace the federated work plan.
pub async fn fed_set_jobs(Json(body): Json<SetJobsBody>) -> Response {
    let n = fed::set_jobs(body.combos, body.token);
    Json(serde_json::json!({ "queued": n })).into_response()
}

#[derive(Debug, serde::Deserialize)]
pub struct NextJobQuery {
    pub worker: Option<String>,
}

/// Worker-facing: lease the next combo. 404 when the queue is empty.
pub async fn fed_next_job(headers: HeaderMap, Query(q): Query<NextJobQuery>) -> Response {
    if !fed::token_ok(provided_token(&headers).as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "bad or missing x-fed-token" })),
        )
            .into_response();
    }
    let worker = q.worker.unwrap_or_else(|| "anonymous".into());
    match fed::next_job(&worker) {
        Some(job) => Json(job).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "no federated work queued" })),
        )
            .into_response(),
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmitBody {
    pub worker: String,
    pub symbol: String,
    pub base_tf: String,
    pub portfolio_json: String,
    pub trades_json: Option<String>,
}

/// Worker-facing: deliver a discovered portfolio into the coordinator inbox.
pub async fn fed_submit(headers: HeaderMap, Json(b): Json<SubmitBody>) -> Response {
    if !fed::token_ok(provided_token(&headers).as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "bad or missing x-fed-token" })),
        )
            .into_response();
    }
    match fed::submit(
        &b.worker,
        &b.symbol,
        &b.base_tf,
        &b.portfolio_json,
        b.trades_json.as_deref(),
    ) {
        Ok(saved) => Json(serde_json::json!({ "saved": saved })).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerStartBody {
    pub coordinator_url: String,
    pub worker_id: Option<String>,
    pub token: Option<String>,
}

/// Local operator: start contributing this machine's cores to a coordinator.
pub async fn fed_worker_start(
    State(state): State<AppApiState>,
    Json(b): Json<WorkerStartBody>,
) -> Response {
    match fed::worker_start(
        state,
        b.coordinator_url,
        b.worker_id.unwrap_or_default(),
        b.token.filter(|t| !t.trim().is_empty()),
    ) {
        Ok(()) => Json(serde_json::json!({ "started": true })).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub async fn fed_worker_stop() -> Response {
    fed::worker_stop();
    Json(serde_json::json!({ "stopped": true })).into_response()
}

/// `GET /mesh/swarm` — the P2P mesh sidecar writes the aggregated swarm
/// capacity (nodes / total cores / RAM / GPUs — the "network as one machine")
/// to a well-known temp file every 30s. This surfaces it to the UI/supervisor
/// so the app can SEE the total resources it could scale the search across.
/// `running: false` when no mesh sidecar is publishing.
pub async fn swarm_capacity() -> Response {
    let path = std::env::temp_dir().join("neoethos_mesh_swarm.json");
    match std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()) {
        Some(mut v) => {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("running".into(), serde_json::json!(true));
            }
            Json(v).into_response()
        }
        None => Json(serde_json::json!({ "running": false, "nodes": 0, "totalCores": 0 })).into_response(),
    }
}

// ── Distributed island-model migration (Federation Phase 1) ──────────────────
// The mesh sidecar drives these; the GA hook is OFF until `enable` is called,
// so a node not in the swarm behaves byte-identically to a single machine.

/// `POST /mesh/migration/enable` — turn on island migration for this process.
/// The mesh calls it once on startup; every local discovery run then publishes
/// its elites and accepts peer migrants.
pub async fn migration_enable() -> Response {
    neoethos_search::set_migration_enabled(true);
    Json(serde_json::json!({ "enabled": true })).into_response()
}

/// `GET /mesh/elites` — drain the genes the local GA published this interval,
/// for the mesh to gossip to peer islands.
pub async fn migration_elites() -> Response {
    Json(neoethos_search::take_elites()).into_response()
}

/// `POST /mesh/migrants` — inject a peer island's elite genes into the local
/// GA's next generation (they are re-scored on our data before they can
/// survive, so a bad migrant just fails selection).
pub async fn migration_migrants(Json(genes): Json<Vec<neoethos_search::Gene>>) -> Response {
    let n = genes.len();
    neoethos_search::push_migrants(genes);
    Json(serde_json::json!({ "accepted": n })).into_response()
}
