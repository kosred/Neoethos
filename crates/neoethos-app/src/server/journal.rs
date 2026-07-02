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

/// Heal a served row IN PLACE (display only — the JSONL on disk is never
/// rewritten):
///   - `#<id>` / `sym#<id>` placeholder symbols resolve through the live
///     symbol catalog (records written before the catalog populated).
///   - v1 rows stored the CLOSING deal's side (opposite of the position) and
///     raw base-unit volume; flip the side and convert units → lots.
fn heal_row(t: &mut journal_store::ClosedTrade, names: &std::collections::HashMap<i64, String>) {
    if let Some(id) = t
        .symbol
        .strip_prefix("sym#")
        .or_else(|| t.symbol.strip_prefix('#'))
        .and_then(|s| s.parse::<i64>().ok())
    {
        if let Some(name) = names.get(&id) {
            t.symbol = name.clone();
        }
    }
    if t.schema_version < 2 {
        t.side = match t.side.trim().to_ascii_uppercase().as_str() {
            "BUY" => "SELL".to_string(),
            "SELL" => "BUY".to_string(),
            _ => t.side.clone(),
        };
        if let Some(m) = neoethos_core::symbol_metadata::resolve(&t.symbol)
            .filter(|m| m.contract_size.is_finite() && m.contract_size > 0.0)
        {
            t.lots /= m.contract_size;
        }
        // Serve with v2 semantics so the UI treats every row uniformly.
        t.schema_version = 2;
    }
}

/// `GET /journal/trades?fromMs&toMs&limit` — closed trades of the ACTIVE
/// account only (automatic — same account selection as the execution path),
/// most-recent first. Legacy rows written before per-account scoping carry no
/// account id (unattributable mixed history) and are hidden once an active
/// account is known; the JSONL on disk keeps everything.
pub async fn trades(State(state): State<AppApiState>, Query(q): Query<JournalQuery>) -> Response {
    let data_dir = match resolve_data_dir() {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let names = state.symbol_catalog_snapshot().await;
    let mut rows = journal_store::query_closed_trades(&data_dir, q.from_ms, q.to_ms);
    if let Some(active) = journal_store::active_account_id() {
        rows.retain(|r| r.account_id.as_deref() == Some(active.as_str()));
    }
    for row in &mut rows {
        heal_row(row, &names);
    }
    rows.reverse(); // most-recent first
    rows.truncate(q.limit.unwrap_or(500));
    Json(rows).into_response()
}

/// `GET /journal/stats?fromMs&toMs` — computed performance stats, scoped to
/// the ACTIVE account (automatic — mirrors `/journal/trades`).
pub async fn stats(State(_state): State<AppApiState>, Query(q): Query<JournalQuery>) -> Response {
    let data_dir = match resolve_data_dir() {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let mut trades = journal_store::query_closed_trades(&data_dir, q.from_ms, q.to_ms);
    let mut equity = journal_store::query_equity(&data_dir, q.from_ms, q.to_ms);
    if let Some(active) = journal_store::active_account_id() {
        trades.retain(|r| r.account_id.as_deref() == Some(active.as_str()));
        equity.retain(|e| e.account_id.as_deref() == Some(active.as_str()));
    }
    Json(journal_stats::compute_stats(&trades, &equity)).into_response()
}
