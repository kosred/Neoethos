//! `/journal/trades` + `/journal/stats` — the trade-journal read API.
//!
//! Serves the JSONL closed-trade store ([`crate::app_services::journal_store`])
//! plus the computed professional stats
//! ([`crate::app_services::journal_stats`]). The data dir is resolved from
//! the live `config.yaml` — the same source the rest of the server uses,
//! so it honours the operator's `--config` / user-data path.
//!
//! Defensive: a config-load failure returns `500` with a clear,
//! actionable message; an empty/missing journal returns an empty list /
//! all-zero stats (never a panic).

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use serde::Deserialize;

use super::state::AppApiState;
use crate::app_services::{journal_stats, journal_store};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JournalQuery {
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    /// Max closed trades to return (most-recent first). Default 500.
    pub limit: Option<usize>,
}

/// Resolve the data dir from the live `config.yaml`, or a 500 Response
/// with a clear message.
fn resolve_data_dir() -> Result<std::path::PathBuf, Response> {
    let path = super::state::current_config_path();
    Settings::from_yaml(&path)
        .map(|s| s.system.data_dir)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("config.yaml not loadable ({}): {e}", path.display()),
                    "code": "config_load_failed",
                })),
            )
                .into_response()
        })
}

/// `GET /journal/trades?fromMs&toMs&limit` — closed trades, most-recent first.
pub async fn trades(State(_state): State<AppApiState>, Query(q): Query<JournalQuery>) -> Response {
    let data_dir = match resolve_data_dir() {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let mut rows = journal_store::query_closed_trades(&data_dir, q.from_ms, q.to_ms);
    rows.reverse(); // most-recent first
    rows.truncate(q.limit.unwrap_or(500));
    Json(rows).into_response()
}

/// `GET /journal/stats?fromMs&toMs` — computed performance stats.
pub async fn stats(State(_state): State<AppApiState>, Query(q): Query<JournalQuery>) -> Response {
    let data_dir = match resolve_data_dir() {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let trades = journal_store::query_closed_trades(&data_dir, q.from_ms, q.to_ms);
    let equity = journal_store::query_equity(&data_dir, q.from_ms, q.to_ms);
    Json(journal_stats::compute_stats(&trades, &equity)).into_response()
}
