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
use crate::app_services::{journal_analytics, journal_stats, journal_store};

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


/// Bars for a trade's own window, read from the price store the engine already
/// keeps.
///
/// The broker reports where a trade opened and closed and nothing in between,
/// so how far it ran in favour before giving the move back is invisible in the
/// account history — which is exactly the question being asked. The stored
/// series has it. The finest timeframe present wins, because excursion measured
/// on daily bars would flatten intraday trades to nothing.
struct StorePrices {
    root: std::path::PathBuf,
}

impl StorePrices {
    /// Finest first: a trade held for an hour needs minute bars to show its
    /// path, and reading a coarse series would silently understate excursion.
    const TIMEFRAMES: [&'static str; 8] = ["M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"];
}

impl journal_analytics::PriceWindow for StorePrices {
    fn window(&self, symbol: &str, from_ms: i64, to_ms: i64) -> Option<(Vec<f64>, Vec<f64>)> {
        let available = neoethos_data::discover_timeframes(&self.root, symbol).ok()?;
        let timeframe = Self::TIMEFRAMES
            .iter()
            .find(|tf| available.iter().any(|have| have.eq_ignore_ascii_case(tf)))?;
        let ohlcv = neoethos_data::load_symbol_timeframe(&self.root, symbol, timeframe).ok()?;
        let stamps = ohlcv.timestamp.as_ref()?;
        let mut highs = Vec::new();
        let mut lows = Vec::new();
        for (index, ts) in stamps.iter().enumerate() {
            if *ts >= from_ms && *ts <= to_ms {
                highs.push(ohlcv.high[index]);
                lows.push(ohlcv.low[index]);
            }
        }
        (!highs.is_empty()).then_some((highs, lows))
    }

    fn pip_size(&self, symbol: &str) -> Option<f64> {
        neoethos_core::symbol_metadata::resolve(symbol)
            .map(|meta| meta.pip_size)
            .filter(|pip| pip.is_finite() && *pip > 0.0)
    }
}

/// `GET /journal/analytics?fromMs&toMs` — the same closed trades, in units that
/// compare and sliced the ways that locate a problem.
///
/// `/journal/stats` reports the totals. This reports where they came from: per
/// trade in pips and R with the excursion recovered from the price series, and
/// grouped by symbol, hour, weekday and direction. A total says the account
/// lost money; these say which hour, which symbol, and whether the profit was
/// ever there to begin with.
pub async fn analytics(
    State(state): State<AppApiState>,
    Query(q): Query<JournalQuery>,
) -> Response {
    let data_dir = match resolve_data_dir() {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let names = state.symbol_catalog_snapshot().await;
    let mut trades = journal_store::query_closed_trades(&data_dir, q.from_ms, q.to_ms);
    if let Some(active) = journal_store::active_account_id() {
        trades.retain(|r| r.account_id.as_deref() == Some(active.as_str()));
    }
    // Same healing `/journal/trades` applies. Without it the per-symbol
    // breakdown splits one instrument across a name and a placeholder id — 97
    // of the operator's 238 trades are stored as `#1`, `#5`, `#41` from before
    // the catalog populated, so the grouping this endpoint exists for would be
    // wrong in exactly the rows that matter most.
    for row in &mut trades {
        heal_row(row, &names);
    }
    // Excursion needs the price store. When it is absent the analytics still
    // return — with the excursion fields empty rather than zeroed, so a missing
    // series never reads as "the trades never went anywhere".
    let prices = StorePrices { root: data_dir };
    Json(journal_analytics::analyse(&trades, Some(&prices))).into_response()
}
