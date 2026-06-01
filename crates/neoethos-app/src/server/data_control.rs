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
    download_history_blocking, fetch_broker_accounts_blocking, fetch_broker_symbols_blocking,
};
use crate::app_services::ctrader_errors::translate_anyhow;

use super::errors::internal_panic;
use super::state::AppApiState;

/// Build a 502 BAD_GATEWAY response that includes the cTrader error
/// translation (when one can be extracted) so the Flutter side can
/// render a friendly banner + action button instead of the raw
/// "errorCode=CH_ACCESS_TOKEN_INVALID" gibberish.
fn broker_gateway_error(err: anyhow::Error) -> Response {
    let raw = err.to_string();
    if let Some(t) = translate_anyhow(&err) {
        let body = serde_json::json!({
            "error": t.message,
            "detail": raw,
            "translation": t,
        });
        return (StatusCode::BAD_GATEWAY, Json(body)).into_response();
    }
    let body = serde_json::json!({
        "error": "Broker request failed — could not reach cTrader. Make sure you're \
                  authenticated (Broker Setup → Re-authenticate) and connected.",
        "detail": raw,
    });
    (StatusCode::BAD_GATEWAY, Json(body)).into_response()
}

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
    /// F-341: canonical asset bucket from the broker's classification —
    /// "forex" | "metals" | "indices" | "commodities". `None` when the
    /// broker's class tables were unavailable (the list is then
    /// unfiltered and the UI falls back to name heuristics).
    pub asset_class: Option<String>,
}

pub async fn symbols(State(state): State<AppApiState>) -> Response {
    match tokio::task::spawn_blocking(fetch_broker_symbols_blocking).await {
        Ok(Ok(bundle)) => {
            // Mirror the (id → name) lookup into AppApiState so the
            // bridge can label positions with real tickers (e.g.
            // `EURUSD`) instead of the previous `sym#1` placeholder.
            // Every successful Markets-tab fetch refreshes this cache —
            // no staleness even after a broker maintenance window
            // that re-issues IDs.
            let catalog: std::collections::HashMap<i64, String> = bundle
                .symbols
                .iter()
                .map(|s| (s.symbol_id, s.symbol_name.clone()))
                .collect();
            state.set_symbol_catalog(catalog).await;

            let asset_class_by_id = bundle.asset_class_by_id;
            let dto = BrokerSymbolsDto {
                account_id: bundle.account_id,
                environment: bundle.environment.to_string(),
                symbol_count: bundle.symbols.len(),
                symbols: bundle
                    .symbols
                    .into_iter()
                    .map(|s| BrokerSymbolDto {
                        asset_class: asset_class_by_id.get(&s.symbol_id).cloned(),
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
        Ok(Err(err)) => broker_gateway_error(err),
        Err(join_err) => internal_panic("Loading broker symbols", join_err),
    }
}

// ─── GET /broker/accounts ─────────────────────────────────────────────────

/// Wire shape for the Settings-screen account picker.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerAccountsDto {
    pub environment: String,
    pub permission_scope: String,
    pub account_count: usize,
    pub accounts: Vec<BrokerAccountDto>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerAccountDto {
    /// Numeric cTID as a string — cTrader's account_id can exceed
    /// i32 range so we serialize as text to keep the wire safe.
    pub account_id: String,
    pub broker_title: String,
    pub account_name: String,
    pub trader_login: Option<i64>,
    pub is_live: Option<bool>,
    /// Whether this account had the "execution" scope checked during
    /// OAuth. The trader-scope flow we use grants execution by
    /// default, but if a user pinned a more restrictive scope here we
    /// surface it so the UI can grey out trade buttons accordingly.
    pub enabled_for_execution: bool,
}

/// Pulls the full list of accounts the user granted access to during
/// OAuth (`ProtoOAGetAccountListByAccessTokenReq` → payload 2150). The
/// Settings dropdown reads this so the operator picks from a real
/// list instead of typing a numeric cTID by hand — which was the
/// root cause of the `CH_ACCESS_TOKEN_INVALID` loop in v0.4.20 where
/// the on-disk config still held a deleted sandbox account_id.
pub async fn accounts(State(_state): State<AppApiState>) -> Response {
    match tokio::task::spawn_blocking(fetch_broker_accounts_blocking).await {
        Ok(Ok(bundle)) => {
            let dto = BrokerAccountsDto {
                environment: bundle.environment.to_string(),
                permission_scope: bundle.permission_scope,
                account_count: bundle.accounts.len(),
                accounts: bundle
                    .accounts
                    .into_iter()
                    .map(|a| BrokerAccountDto {
                        account_id: a.account_id,
                        broker_title: a.broker_title,
                        account_name: a.account_name,
                        trader_login: a.trader_login,
                        is_live: a.is_live,
                        enabled_for_execution: a.enabled_for_execution,
                    })
                    .collect(),
            };
            Json(dto).into_response()
        }
        Ok(Err(err)) => broker_gateway_error(err),
        Err(join_err) => internal_panic("Loading broker accounts", join_err),
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

pub async fn fetch(State(state): State<AppApiState>, Json(body): Json<FetchBody>) -> Response {
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
    // F-553/F-576 closure (2026-05-25): config path threaded from CLI.
    let config_path = state.config_path().to_path_buf();
    let result = tokio::task::spawn_blocking(move || {
        let settings = Settings::from_yaml(&config_path)
            .map_err(|e| anyhow::anyhow!("{} not loadable: {e}", config_path.display()))?;
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
        Ok(Ok(outcome)) => {
            // **2026-05-25 — chart-cache invalidation**: the Vortex
            // file for this (symbol, *) was just rewritten by the
            // `download_history_blocking` path. Drop any cached
            // `ChartDto` for that symbol so the next chart click
            // re-reads the fresh bars from disk instead of serving
            // a 15s-old snapshot of the previous file.
            super::chart_cache::clear_symbol(&outcome.symbol);
            Json(FetchOutcomeDto {
                symbol: outcome.symbol,
                timeframe: outcome.timeframe,
                bar_count: outcome.bar_count,
                has_more: outcome.has_more,
                written_path: outcome.written_path.display().to_string(),
            })
            .into_response()
        }
        Ok(Err(err)) => broker_gateway_error(err),
        Err(join_err) => internal_panic("Downloading market data", join_err),
    }
}

// ─── POST /data/import ────────────────────────────────────────────────────

/// Request body for `POST /data/import` (#192).
///
/// `source_path` is the absolute path to the file the user wants to
/// ingest. `symbol`/`timeframe` decide where the converted Vortex file
/// lands on disk (`data/symbol=<sym>/timeframe=<tf>/data.vortex`). The
/// source format is auto-detected from the file extension by the data
/// layer's `DataFormat::from_extension`, so we don't ask the user to
/// pick "CSV vs Parquet" — they just give us a file.
#[derive(Debug, serde::Deserialize)]
pub struct ImportBody {
    #[serde(rename = "sourcePath")]
    pub source_path: String,
    pub symbol: String,
    pub timeframe: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportOutcomeDto {
    pub symbol: String,
    pub timeframe: String,
    pub source_format: String,
    pub written_path: String,
}

/// `POST /data/import` — convert a user-provided CSV/Parquet/Arrow/
/// JSON/JSONL/TSV file into the canonical Vortex layout under
/// `data_dir/symbol=<S>/timeframe=<T>/data.vortex`.
///
/// This is the "I have my own data, don't make me re-download from
/// the broker" workflow. The data layer's `convert_to_vortex` does
/// the actual schema validation + write; we just route requests at
/// it and return a tidy DTO.
pub async fn import_file(
    State(state): State<AppApiState>,
    Json(body): Json<ImportBody>,
) -> Response {
    let symbol = body.symbol.trim().to_uppercase();
    let timeframe = body.timeframe.trim().to_uppercase();
    let source_path = body.source_path.trim().to_string();

    if symbol.is_empty() || timeframe.is_empty() || source_path.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "sourcePath, symbol, and timeframe must all be non-empty",
            })),
        )
            .into_response();
    }

    // F-553/F-576 closure (2026-05-25): config path threaded from CLI.
    let config_path = state.config_path().to_path_buf();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<ImportOutcomeDto> {
        let settings = Settings::from_yaml(&config_path)
            .map_err(|e| anyhow::anyhow!("{} not loadable: {e}", config_path.display()))?;
        let source = std::path::Path::new(&source_path);
        if !source.exists() {
            anyhow::bail!("source file not found: {}", source.display());
        }
        // Auto-detect format from extension (CSV/TSV/Parquet/JSON/
        // JSONL/Arrow/IPC/Feather). Anything else returns an `Err`.
        let ext = source.extension().and_then(|e| e.to_str()).unwrap_or("");
        let format =
            neoethos_data::core::discover::DataFormat::from_extension(ext).ok_or_else(|| {
                anyhow::anyhow!(
                    "unsupported extension on {} — \
                     supported: csv, tsv, parquet, json, jsonl, arrow, ipc, feather",
                    source.display()
                )
            })?;
        let destination = neoethos_data::symbol_timeframe_vortex_path(
            &settings.system.data_dir,
            &symbol,
            &timeframe,
        );
        let hint = neoethos_data::core::to_vortex::IngestionSchema {
            optional: vec!["volume".to_string()],
            timeframe_hint: Some(timeframe.clone()),
        };
        let written = neoethos_data::core::to_vortex::convert_to_vortex(
            source,
            format,
            &destination,
            Some(&hint),
        )?;
        Ok(ImportOutcomeDto {
            symbol: symbol.clone(),
            timeframe: timeframe.clone(),
            source_format: format!("{format:?}"),
            written_path: written.display().to_string(),
        })
    })
    .await;

    match result {
        Ok(Ok(dto)) => {
            // **2026-05-25 — chart-cache invalidation**: import_file
            // rewrites the Vortex for (symbol, *). Same reasoning as
            // the `fetch` handler — drop the now-stale chart cache.
            super::chart_cache::clear_symbol(&dto.symbol);
            Json(dto).into_response()
        }
        Ok(Err(err)) => {
            let friendly_err = anyhow::anyhow!("{err}");
            super::errors::actionable_error(
                StatusCode::BAD_REQUEST,
                "File import failed. Supported formats: CSV, TSV, Parquet, JSON, JSONL, Arrow. \
                 Check the path is correct and the file isn't open elsewhere.",
                &friendly_err,
            )
        }
        Err(join_err) => internal_panic("Importing the file", join_err),
    }
}
