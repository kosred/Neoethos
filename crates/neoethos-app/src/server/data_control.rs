//! Endpoints that talk to the broker symbol catalog + historical
//! bars feed:
//!
//!   GET  /broker/symbols           — what this account can trade
//!   POST /data/fetch               — download bars + persist to disk
//!   GET  /data/fetch/status        — exact active run id + phase
//!   POST /data/fetch/stop          — cancel one exact capturing run id
//!
//! Both share the `broker_api` helper module. CPU-bound route work first
//! enters the process admission coordinator, then transfers that exact lease
//! into the blocking lifetime; `spawn_blocking` is only the async/runtime
//! boundary and never grants capacity by itself.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_broker_history::{
    BrokerHistoryConflict, HistoricalFetchCancelResult, HistoricalFetchStartFailure,
    begin_process_historical_capture, cancel_process_historical_capture,
    is_historical_capture_cancelled, process_historical_capture_status,
};
use neoethos_core::Settings;
use neoethos_data::ExactDatasetGenerationConflict;
use neoethos_data::core::dataset_manifest::PublicationConflict;

use crate::app_services::broker_api::{
    download_history_blocking, fetch_broker_accounts_blocking,
    fetch_broker_cash_flow_history_blocking, fetch_broker_ctid_profile_blocking,
    fetch_broker_expected_margin_blocking, fetch_broker_order_history_blocking,
    fetch_broker_symbols_blocking, fetch_broker_version_blocking,
};
use crate::app_services::ctrader_errors::translate_anyhow;
use crate::app_services::ctrader_messages::CTraderBlockedPayloadError;

use super::errors::internal_panic;
use super::state::AppApiState;

/// Build a 502 BAD_GATEWAY response that includes the cTrader error
/// translation (when one can be extracted) so the Flutter side can
/// render a friendly banner + action button instead of the raw
/// "errorCode=CH_ACCESS_TOKEN_INVALID" gibberish.
fn broker_gateway_error(err: anyhow::Error) -> Response {
    let raw = err.to_string();
    let retry_after_seconds = err
        .downcast_ref::<CTraderBlockedPayloadError>()
        .map(CTraderBlockedPayloadError::retry_after_seconds)
        .or_else(|| {
            err.downcast_ref::<
                neoethos_broker_history::ctrader_messages::CTraderBlockedPayloadError,
            >()
            .map(
                neoethos_broker_history::ctrader_messages::CTraderBlockedPayloadError::retry_after_seconds,
            )
        });
    if let Some(retry_after_seconds) = retry_after_seconds {
        let mut response = (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": "cTrader temporarily blocked this historical payload type; the batch was stopped without retry.",
                "detail": raw,
                "code": "BLOCKED_PAYLOAD_TYPE",
                "retryAfterSeconds": retry_after_seconds,
            })),
        )
            .into_response();
        if let Some(seconds) = retry_after_seconds
            && let Ok(value) = axum::http::HeaderValue::from_str(&seconds.to_string())
        {
            response
                .headers_mut()
                .insert(axum::http::header::RETRY_AFTER, value);
        }
        return response;
    }
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

fn cancelled_fetch_response(run_id: u64) -> Response {
    (
        StatusCode::CONFLICT,
        Json(serde_json::json!({
            "error": "broker historical fetch was cancelled",
            "code": "FETCH_CANCELLED",
            "outcome": "cancelled",
            "runId": run_id,
        })),
    )
        .into_response()
}

fn broker_fetch_error(err: anyhow::Error, run_id: u64) -> Response {
    if is_historical_capture_cancelled(err.as_ref()) {
        return cancelled_fetch_response(run_id);
    }

    let conflict_code = if err
        .downcast_ref::<ExactDatasetGenerationConflict>()
        .is_some()
    {
        Some("STALE_DATASET_RECEIPT")
    } else if err.downcast_ref::<PublicationConflict>().is_some() {
        Some("DATASET_PUBLICATION_CONFLICT")
    } else {
        err.downcast_ref::<BrokerHistoryConflict>()
            .map(BrokerHistoryConflict::response_code)
    };
    if let Some(conflict_code) = conflict_code {
        let detail = err.to_string();
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "broker dataset selection conflicts with current canonical state",
                "detail": detail,
                "code": conflict_code,
                "runId": run_id,
            })),
        )
            .into_response();
    }

    broker_gateway_error(err)
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
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct FetchBody {
    pub symbol: String,
    pub timeframe: String,
    /// Unix-millis inclusive lower bound.
    #[serde(rename = "fromMs")]
    pub from_ms: i64,
    /// Unix-millis exclusive upper bound. `None` → now.
    #[serde(rename = "toMs")]
    pub to_ms: Option<i64>,
    /// Exact refresh receipt from `/data/bootstrap`; `None` is CREATE-only.
    pub dataset_selection: Option<neoethos_data::SelectedDatasetGenerationV1>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FetchOutcomeDto {
    pub symbol: String,
    pub timeframe: String,
    pub bar_count: usize,
    pub has_more: bool,
    pub written_path: String,
    /// Unix-millis of the oldest bar returned (serialized `oldestMs`); null when
    /// 0 bars. UI uses it to show actual history depth + warn on shallow data.
    pub oldest_ms: Option<i64>,
    pub dataset_identity: String,
    pub generation: String,
    pub durable_commit_id: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FetchStatusDto {
    pub active: bool,
    pub run_id: Option<u64>,
    pub phase: Option<&'static str>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct StopFetchBody {
    pub run_id: u64,
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "outcome")]
enum StopFetchOutcomeDto {
    #[serde(rename = "cancelled")]
    Cancelled {
        #[serde(rename = "runId")]
        run_id: u64,
    },
    #[serde(rename = "publication_in_progress")]
    PublicationInProgress {
        #[serde(rename = "runId")]
        run_id: u64,
    },
    #[serde(rename = "stale_run")]
    StaleRun {
        #[serde(rename = "requestedRunId")]
        requested_run_id: u64,
        #[serde(rename = "activeRunId")]
        active_run_id: u64,
    },
    #[serde(rename = "no_active_fetch")]
    NoActiveFetch,
}

fn cancelled_before_broker_execution(run_id: u64) -> Response {
    cancelled_fetch_response(run_id)
}

/// Current CPU demand of the broker-history pipeline. The network fetch,
/// validation and canonical publication are serial today, so reserving the
/// process-wide N-2 limit would strand capacity without creating parallel
/// work. Keep this typed hook at the route boundary so a future proven bounded
/// parallel publisher can raise its declared demand without bypassing shared
/// admission.
fn broker_fetch_cpu_demand() -> neoethos_core::execution_budget::CpuPermitRequest {
    let width = neoethos_core::execution_budget::WorkerLimit::new(1)
        .expect("one broker-fetch worker is a valid positive CPU demand");
    neoethos_core::execution_budget::CpuPermitRequest::local(width)
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
    let dataset_selection = body.dataset_selection;
    let active_fetch = match begin_process_historical_capture() {
        Ok(active_fetch) => active_fetch,
        Err(HistoricalFetchStartFailure::AlreadyActive { active_run_id }) => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": "a broker historical fetch is already active",
                    "activeRunId": active_run_id,
                })),
            )
                .into_response();
        }
        Err(HistoricalFetchStartFailure::RunIdOverflow) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "broker historical fetch id space is exhausted",
                })),
            )
                .into_response();
        }
        Err(HistoricalFetchStartFailure::Cancelled { run_id }) => {
            return cancelled_before_broker_execution(run_id);
        }
    };
    let run_id = active_fetch.run_id();
    let Some(execution) = state.execution_state() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "broker fetch unavailable: process CPU admission was not installed before AppApiState",
            })),
        )
            .into_response();
    };
    let pending_admission = match execution
        .admission_client()
        .submit(broker_fetch_cpu_demand())
    {
        Ok(pending) => pending,
        Err(error) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "broker fetch admission failed",
                    "detail": error.to_string(),
                })),
            )
                .into_response();
        }
    };
    let mut pending_wait = Box::pin(pending_admission.wait());
    let admitted = loop {
        tokio::select! {
            result = &mut pending_wait => match result {
                Ok(admitted) => break admitted,
                Err(error) => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({
                            "error": "broker fetch admission failed",
                            "detail": error.to_string(),
                        })),
                    )
                        .into_response();
                }
            },
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                if active_fetch.is_cancelled() {
                    return cancelled_before_broker_execution(run_id);
                }
            }
        }
    };
    drop(pending_wait);
    if active_fetch.is_cancelled() {
        return cancelled_before_broker_execution(run_id);
    }

    // F-553/F-576 closure (2026-05-25): config path threaded from CLI.
    let config_path = state.config_path().to_path_buf();
    let executor = execution.executor().clone();
    let result = tokio::task::spawn_blocking(move || {
        admitted.execute(&executor, move || {
            let settings = Settings::from_yaml(&config_path)
                .map_err(|e| anyhow::anyhow!("{} not loadable: {e}", config_path.display()))?;
            download_history_blocking(
                &symbol,
                &timeframe,
                from_ms,
                to_ms,
                &settings.system.data_dir,
                dataset_selection.as_ref(),
                &active_fetch,
            )
        })
    })
    .await;

    match result {
        Ok(Ok(Ok(outcome))) => {
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
                oldest_ms: outcome.oldest_ms,
                dataset_identity: outcome.dataset_identity,
                generation: outcome.generation,
                durable_commit_id: outcome.durable_commit_id,
            })
            .into_response()
        }
        Ok(Ok(Err(err))) => broker_fetch_error(err, run_id),
        Ok(Err(error)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "broker fetch CPU execution failed",
                "detail": error.to_string(),
            })),
        )
            .into_response(),
        Err(join_err) => internal_panic("Downloading market data", join_err),
    }
}

pub async fn fetch_status() -> Json<FetchStatusDto> {
    let status = process_historical_capture_status();
    Json(match status {
        Some(status) => FetchStatusDto {
            active: true,
            run_id: Some(status.run_id),
            phase: Some(status.phase),
        },
        None => FetchStatusDto {
            active: false,
            run_id: None,
            phase: None,
        },
    })
}

pub async fn stop_fetch(Json(body): Json<StopFetchBody>) -> Response {
    match cancel_process_historical_capture(body.run_id) {
        HistoricalFetchCancelResult::Cancelled { run_id } => (
            StatusCode::ACCEPTED,
            Json(StopFetchOutcomeDto::Cancelled { run_id }),
        )
            .into_response(),
        HistoricalFetchCancelResult::PublicationInProgress { run_id } => (
            StatusCode::CONFLICT,
            Json(StopFetchOutcomeDto::PublicationInProgress { run_id }),
        )
            .into_response(),
        HistoricalFetchCancelResult::StaleRun {
            requested_run_id,
            active_run_id,
        } => (
            StatusCode::CONFLICT,
            Json(StopFetchOutcomeDto::StaleRun {
                requested_run_id,
                active_run_id,
            }),
        )
            .into_response(),
        HistoricalFetchCancelResult::NoActiveFetch => (
            StatusCode::CONFLICT,
            Json(StopFetchOutcomeDto::NoActiveFetch),
        )
            .into_response(),
    }
}

// ─── POST /data/import ────────────────────────────────────────────────────

/// Request body for `POST /data/import` (#192).
///
/// This adapter carries every provenance choice explicitly. The source
/// extension may help the desktop pre-fill `source_format`, but the server
/// never guesses the route or timestamp meaning from the filename/bytes.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ImportBody {
    #[serde(rename = "sourcePath")]
    pub source_path: String,
    pub source_format: neoethos_data::core::import_provenance::ImportSourceFormat,
    pub source_namespace: String,
    pub symbol: String,
    pub timeframe: String,
    pub bar_timestamp_convention: String,
    pub expected_generation: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportOutcomeDto {
    pub symbol: String,
    pub timeframe: String,
    pub source_format: String,
    pub dataset_identity: String,
    pub written_path: String,
    pub row_count: u64,
    pub generation: String,
    pub durable_commit_id: String,
    pub source_sha256: String,
}

/// `POST /data/import` — explicitly import one user-provided source into an
/// immutable, verified canonical Vortex generation. Admission atomically
/// reserves the full import CPU plan and one SourceSeal slot before any source
/// byte is opened. Runtime consumers only reopen the published Vortex path.
pub async fn import_file(
    State(state): State<AppApiState>,
    Json(body): Json<ImportBody>,
) -> Response {
    let parsed = match ParsedImportBody::try_from(body) {
        Ok(parsed) => parsed,
        Err(error) => {
            return super::errors::actionable_error(
                StatusCode::BAD_REQUEST,
                "Import request is invalid. Declare the exact source format, external source namespace, canonical cTrader timeframe, and timestamp meaning. Only explicitly evidenced bar-open timestamps can become canonical data.",
                &error,
            );
        }
    };
    let Some(execution) = state.execution_state() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "import unavailable: process CPU admission was not installed before AppApiState",
            })),
        )
            .into_response();
    };
    let snapshot = execution.admission_snapshot();
    let admitted = match execution
        .admission_client()
        .admit_import(neoethos_core::execution_budget::CpuPermitRequest::local(
            snapshot.cpu.installed_limit,
        ))
        .await
    {
        Ok(admitted) => admitted,
        Err(error) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "import admission failed",
                    "detail": error.to_string(),
                })),
            )
                .into_response();
        }
    };

    let config_path = state.config_path().to_path_buf();
    let result = tokio::task::spawn_blocking(move || {
        admitted.execute(move |_cpu_lease, source_seal_slot| {
            run_admitted_import(&config_path, parsed, source_seal_slot)
        })
    })
    .await;

    match result {
        Ok(Ok(dto)) => {
            super::chart_cache::clear_symbol(&dto.symbol);
            Json(dto).into_response()
        }
        Ok(Err(err)) => {
            let friendly_err = anyhow::anyhow!("{err}");
            super::errors::actionable_error(
                StatusCode::BAD_REQUEST,
                "File import failed before a canonical generation could be acknowledged. Check the declared format/schema, bar-open timestamp contract, source stability, disk limits, and expectedGeneration.",
                &friendly_err,
            )
        }
        Err(join_err) => internal_panic("Importing the file", join_err),
    }
}

struct ParsedImportBody {
    source_path: std::path::PathBuf,
    source_format: neoethos_data::core::import_provenance::ImportSourceFormat,
    identity: neoethos_data::CanonicalDatasetIdentity,
    expected_generation: Option<String>,
}

impl TryFrom<ImportBody> for ParsedImportBody {
    type Error = anyhow::Error;

    fn try_from(body: ImportBody) -> anyhow::Result<Self> {
        let source_path = std::path::PathBuf::from(body.source_path.trim());
        if source_path.as_os_str().is_empty() || !source_path.is_absolute() {
            anyhow::bail!("sourcePath must be a non-empty absolute path");
        }
        let source_namespace = body.source_namespace.trim();
        let symbol = body.symbol.trim();
        if source_namespace.is_empty() || symbol.is_empty() {
            anyhow::bail!("sourceNamespace and symbol must be non-empty");
        }
        let timeframe = body
            .timeframe
            .trim()
            .parse::<neoethos_data::CanonicalTimeframe>()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let convention = body
            .bar_timestamp_convention
            .trim()
            .parse::<neoethos_data::BarTimestampConvention>()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if !convention.is_canonical_bar_open() {
            anyhow::bail!(
                "barTimestampConvention={} cannot become canonical; only bar_open is accepted",
                convention
            );
        }
        let expected_generation = match body.expected_generation {
            Some(generation) => {
                let generation = generation.trim().to_owned();
                if generation.is_empty() {
                    anyhow::bail!("expectedGeneration cannot be an empty string");
                }
                Some(generation)
            }
            None => None,
        };
        let identity = neoethos_data::CanonicalDatasetIdentity::external(
            source_namespace,
            symbol,
            timeframe,
            convention,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(Self {
            source_path,
            source_format: body.source_format,
            identity,
            expected_generation,
        })
    }
}

fn run_admitted_import(
    config_path: &std::path::Path,
    parsed: ParsedImportBody,
    source_seal_slot: &neoethos_core::execution_budget::AuxiliarySlotLease,
) -> anyhow::Result<ImportOutcomeDto> {
    let settings = Settings::from_yaml(config_path)
        .map_err(|error| anyhow::anyhow!("{} not loadable: {error}", config_path.display()))?;
    let limits = neoethos_data::core::import_limits::ImportLimits::default();
    let imported = neoethos_data::core::import_service::import_path_to_vortex(
        neoethos_data::core::import_service::ImportRequest {
            source_path: &parsed.source_path,
            configured_root: &settings.system.data_dir,
            identity: &parsed.identity,
            declared_format: parsed.source_format,
            expected_generation: parsed.expected_generation.as_deref(),
            limits: &limits,
            auxiliary_slot: source_seal_slot,
        },
    )?;
    let manifest = imported.manifest();
    let provenance = imported.provenance();
    if provenance.dataset_identity() != &parsed.identity
        || provenance.selected_format() != parsed.source_format
        || provenance.detected_format() != parsed.source_format
    {
        anyhow::bail!("reopened canonical import provenance disagrees with the request");
    }
    Ok(ImportOutcomeDto {
        symbol: parsed.identity.symbol_name().to_owned(),
        timeframe: parsed.identity.timeframe().to_string(),
        source_format: parsed.source_format.as_str().to_owned(),
        dataset_identity: parsed.identity.to_path_component(),
        written_path: manifest.generation_path().display().to_string(),
        row_count: imported.row_count(),
        generation: imported.generation().to_owned(),
        durable_commit_id: imported.durable_commit_id().to_owned(),
        source_sha256: hex::encode(provenance.source_sha256()),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 2026-06-10 — cTrader Open API history / margin / profile endpoints.
// Thin three-arm `spawn_blocking` wrappers over the broker_api fetch fns,
// exactly like `symbols` above. The bundle structs are Serialize + camelCase,
// so the handlers return them directly.
// ═══════════════════════════════════════════════════════════════════════════

const DEFAULT_HISTORY_WINDOW_MS: i64 = 604_800_000; // 7 days (the broker cap)

/// `?from=<ms>&to=<ms>`; both optional. Default = the last 7 days (the broker's
/// maximum window), which is the most useful default for a journal view.
#[derive(Debug, serde::Deserialize)]
pub struct HistoryWindowQuery {
    pub from: Option<i64>,
    pub to: Option<i64>,
}

fn resolve_history_window(q: &HistoryWindowQuery) -> (i64, i64) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let to = q.to.unwrap_or(now_ms);
    let from = q.from.unwrap_or(to - DEFAULT_HISTORY_WINDOW_MS);
    (from, to)
}

// ─── GET /broker/orders/history ───────────────────────────────────────────

pub async fn order_history(
    State(_state): State<AppApiState>,
    Query(q): Query<HistoryWindowQuery>,
) -> Response {
    let (from, to) = resolve_history_window(&q);
    match tokio::task::spawn_blocking(move || fetch_broker_order_history_blocking(from, to)).await {
        Ok(Ok(bundle)) => Json(bundle).into_response(),
        Ok(Err(err)) => broker_gateway_error(err),
        Err(join_err) => internal_panic("Loading broker order history", join_err),
    }
}

// ─── GET /broker/cashflow ─────────────────────────────────────────────────

pub async fn cash_flow_history(
    State(_state): State<AppApiState>,
    Query(q): Query<HistoryWindowQuery>,
) -> Response {
    let (from, to) = resolve_history_window(&q);
    match tokio::task::spawn_blocking(move || fetch_broker_cash_flow_history_blocking(from, to))
        .await
    {
        Ok(Ok(bundle)) => Json(bundle).into_response(),
        Ok(Err(err)) => broker_gateway_error(err),
        Err(join_err) => internal_panic("Loading broker cash-flow history", join_err),
    }
}

// ─── GET /broker/margin/expected?symbolId=..&volume=.. ────────────────────

/// `symbolId` is required; `volume` is the wire volume (0.01-unit cents) to
/// price the margin for — defaults to one standard lot (10_000_000 cents).
#[derive(Debug, serde::Deserialize)]
pub struct ExpectedMarginQuery {
    #[serde(rename = "symbolId")]
    pub symbol_id: i64,
    pub volume: Option<i64>,
}

pub async fn expected_margin(
    State(_state): State<AppApiState>,
    Query(q): Query<ExpectedMarginQuery>,
) -> Response {
    let symbol_id = q.symbol_id;
    let volume = q.volume.unwrap_or(10_000_000); // 1.0 lot default
    if symbol_id <= 0 || volume <= 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "symbolId and volume must both be positive"})),
        )
            .into_response();
    }
    match tokio::task::spawn_blocking(move || {
        fetch_broker_expected_margin_blocking(symbol_id, vec![volume])
    })
    .await
    {
        Ok(Ok(bundle)) => Json(bundle).into_response(),
        Ok(Err(err)) => broker_gateway_error(err),
        Err(join_err) => internal_panic("Computing expected margin", join_err),
    }
}

// ─── GET /broker/profile ──────────────────────────────────────────────────

pub async fn ctid_profile(State(_state): State<AppApiState>) -> Response {
    match tokio::task::spawn_blocking(fetch_broker_ctid_profile_blocking).await {
        Ok(Ok(snapshot)) => Json(snapshot).into_response(),
        Ok(Err(err)) => broker_gateway_error(err),
        Err(join_err) => internal_panic("Loading cTID profile", join_err),
    }
}

// ─── GET /broker/version ──────────────────────────────────────────────────

pub async fn server_version(State(_state): State<AppApiState>) -> Response {
    match tokio::task::spawn_blocking(fetch_broker_version_blocking).await {
        Ok(Ok(snapshot)) => Json(snapshot).into_response(),
        Ok(Err(err)) => broker_gateway_error(err),
        Err(join_err) => internal_panic("Loading broker version", join_err),
    }
}
