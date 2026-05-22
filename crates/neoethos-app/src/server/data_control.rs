//! Endpoints that talk to the broker symbol catalog + historical
//! bars feed:
//!
//!   GET  /broker/symbols           — what this account can trade
//!   POST /data/fetch               — download bars + persist to disk
//!
//! Both share the `broker_api` helper module so the route bodies
//! are thin wrappers around `spawn_blocking`.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;

use crate::app_services::broker_api::{
    download_history_blocking, fetch_broker_symbols_blocking,
};

use super::state::AppApiState;

// ─── GET /broker/timeframes ───────────────────────────────────────────────

/// Returns the canonical 11 timeframes that the cTrader Open API
/// trendbar period mapper accepts — sourced from
/// `neoethos_core::CANONICAL_TIMEFRAMES` so a workspace-wide change
/// to that contract is picked up by the UI automatically. The Flutter
/// chart + bootstrap screens read this instead of hardcoding chip
/// lists locally.
///
/// Why this is **not** per-symbol: cTrader's ProtoOATrendbarPeriod is
/// a global enum (M1..MN1) — every symbol the broker offers supports
/// the same set. If we ever flip to a broker that varies timeframes
/// per symbol, this endpoint grows a `?symbol=` query and the wire
/// shape stays compatible.
pub async fn timeframes(State(_state): State<AppApiState>) -> Response {
    let list: Vec<String> = neoethos_core::CANONICAL_TIMEFRAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    Json(serde_json::json!({
        "timeframes": list,
        "count": list.len(),
    }))
    .into_response()
}

// ─── GET /broker/symbols ──────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerSymbolsDto {
    pub account_id: i64,
    pub environment: String,
    pub symbol_count: usize,
    pub symbols: Vec<BrokerSymbolDto>,
    pub archived_symbols: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerSymbolDto {
    pub symbol_id: i64,
    pub symbol_name: String,
    pub enabled: bool,
    pub description: Option<String>,
}

pub async fn symbols(State(_state): State<AppApiState>) -> Response {
    match tokio::task::spawn_blocking(fetch_broker_symbols_blocking).await {
        Ok(Ok(bundle)) => {
            let dto = BrokerSymbolsDto {
                account_id: bundle.account_id,
                environment: bundle.environment.to_string(),
                symbol_count: bundle.symbols.len(),
                symbols: bundle
                    .symbols
                    .into_iter()
                    .map(|s| BrokerSymbolDto {
                        symbol_id: s.symbol_id,
                        symbol_name: s.symbol_name,
                        enabled: s.enabled,
                        description: s.description,
                    })
                    .collect(),
                archived_symbols: bundle.archived_symbols,
            };
            Json(dto).into_response()
        }
        Ok(Err(err)) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("symbols task panicked: {join_err}"),
            })),
        )
            .into_response(),
    }
}

// ─── POST /data/fetch ─────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct FetchBody {
    pub symbol: String,
    pub timeframe: String,
    /// Unix-millis inclusive lower bound.
    #[serde(rename = "fromMs")]
    pub from_ms: i64,
    /// Unix-millis exclusive upper bound. `None` → now.
    #[serde(rename = "toMs")]
    pub to_ms: Option<i64>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FetchOutcomeDto {
    pub symbol: String,
    pub timeframe: String,
    pub bar_count: usize,
    pub has_more: bool,
    pub written_path: String,
}

pub async fn fetch(
    State(_state): State<AppApiState>,
    Json(body): Json<FetchBody>,
) -> Response {
    let now_ms = chrono::Utc::now().timestamp_millis();
    let to_ms = body.to_ms.unwrap_or(now_ms);

    let symbol = body.symbol.trim().to_uppercase();
    let timeframe = body.timeframe.trim().to_uppercase();
    if symbol.is_empty() || timeframe.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "symbol and timeframe must be non-empty",
            })),
        )
            .into_response();
    }

    let from_ms = body.from_ms;
    let result = tokio::task::spawn_blocking(move || {
        let settings = Settings::from_yaml("config.yaml")
            .map_err(|e| anyhow::anyhow!("config.yaml not loadable: {e}"))?;
        download_history_blocking(
            &symbol,
            &timeframe,
            from_ms,
            to_ms,
            &settings.system.data_dir,
        )
    })
    .await;

    match result {
        Ok(Ok(outcome)) => Json(FetchOutcomeDto {
            symbol: outcome.symbol,
            timeframe: outcome.timeframe,
            bar_count: outcome.bar_count,
            has_more: outcome.has_more,
            written_path: outcome.written_path.display().to_string(),
        })
        .into_response(),
        Ok(Err(err)) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("fetch task panicked: {join_err}"),
            })),
        )
            .into_response(),
    }
}
