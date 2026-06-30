//! Three small status endpoints that the Flutter shell consumes to
//! render the "what's actually running" surfaces (Engine Health card,
//! Broker Setup tab, Data Bootstrap tab).
//!
//! Each is read-only and dirt cheap — they read on-disk artifacts and
//! return a small struct. Control endpoints (start/stop discovery,
//! re-OAuth broker, kick off bootstrap) land in a follow-up because
//! they involve writes to running state.

use std::path::PathBuf;
use std::time::SystemTime;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;

use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::jobs::JobKind;

use super::errors::{actionable_error, internal_panic};
use super::state::AppApiState;

// ─── /engines/status ──────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnginesDto {
    pub discovery: String,
    pub training: String,
    pub auto_trader: String,
    /// Human-readable progress / status line for whichever engine is
    /// currently active. Empty when all three are Idle.
    pub discovery_summary: String,
    pub training_summary: String,
    /// F-340 (Feature #14): live discovery progress mirrored from the
    /// running job's `JobSnapshot`. `discoveryStage` is the coarse phase
    /// label (e.g. `"search_generations"`), `""` when idle.
    pub discovery_stage: String,
    /// 0.0..=1.0 completion fraction for the active discovery run;
    /// 0.0 when idle.
    pub discovery_percent: f64,
    /// The live `(name, value)` counters the discovery job accumulates
    /// (candidates evaluated, generations done, …). Empty when idle.
    pub discovery_counters: Vec<EngineCounterDto>,
    /// Live machine-resource readout so the UI can show what discovery is
    /// consuming (operator visibility — the run used to be a black box).
    /// Total / currently-available physical RAM, in GB.
    pub ram_total_gb: f64,
    pub ram_available_gb: f64,
    /// On-disk size of the feature-store temp dir (MB) — the multi-TF cubes
    /// discovery streams to disk; 0 when everything fits in RAM. Reclaimed
    /// per-TF as each unit finishes.
    pub feature_store_mb: u64,
}

/// Sum the on-disk `.fstore` cubes discovery is currently holding (MB).
fn feature_store_disk_mb() -> u64 {
    let dir = std::env::temp_dir().join("neoethos_feature_store");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return 0;
    };
    let bytes: u64 = entries
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("fstore"))
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum();
    bytes / (1 << 20)
}

/// F-340 (Feature #14): one live counter from a running engine's
/// `JobReport`. Serialized as `{ "name": String, "value": u64 }`.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineCounterDto {
    pub name: String,
    pub value: u64,
}

/// Engine-state endpoint. Reads the latest `EngineRunState` written
/// by the background ServiceEvent drainer that the `engines_control`
/// start handlers spawn. Auto-Trader still reports `"Idle"` — it ships
/// in a follow-up wiring along with the order-ticket endpoints.
pub async fn engines(State(state): State<AppApiState>) -> Json<EnginesDto> {
    // F-340 (Feature #14): pull the live discovery progress triple
    // (stage, percent, counters) alongside the existing state/summary.
    let (discovery_stage, discovery_percent, discovery_counters) =
        state.engine_progress(JobKind::Discovery).await;
    Json(EnginesDto {
        discovery: state
            .engine_state(JobKind::Discovery)
            .await
            .as_str()
            .to_string(),
        training: state
            .engine_state(JobKind::Training)
            .await
            .as_str()
            .to_string(),
        auto_trader: "Idle".to_string(),
        discovery_summary: state.engine_summary(JobKind::Discovery).await,
        training_summary: state.engine_summary(JobKind::Training).await,
        discovery_stage,
        discovery_percent,
        discovery_counters: discovery_counters
            .into_iter()
            .map(|(name, value)| EngineCounterDto { name, value })
            .collect(),
        ram_total_gb: neoethos_core::total_memory_bytes() as f64 / 1e9,
        ram_available_gb: neoethos_core::available_memory_bytes() as f64 / 1e9,
        feature_store_mb: feature_store_disk_mb(),
    })
}

// ─── /broker/status ───────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerStatusDto {
    /// Active broker adapter ("cTrader" / "DXtrade"). Picked from the
    /// runtime broker_credentials.toml.
    pub adapter: String,
    /// "Live" or "Demo".
    pub environment: String,
    /// First (and currently only) account configured for execution.
    pub account_id: String,
    /// Whether the bridge's last cTrader refresh succeeded. The
    /// Flutter side uses this to render a green/red dot next to the
    /// adapter name.
    pub connected: bool,
    /// `client_id` of the OAuth app baked into this binary. We mask
    /// everything after the underscore prefix so the full secret
    /// never escapes the server logs / wire.
    pub client_id_prefix: String,
}

pub async fn broker_status(State(state): State<AppApiState>) -> Response {
    let settings = match tokio::task::spawn_blocking(load_broker_settings).await {
        Ok(s) => s,
        Err(join_err) => {
            tracing::warn!(
                target: "neoethos_app::server::system_status",
                error = %join_err,
                "load_broker_settings panicked"
            );
            return internal_panic("Loading broker status", join_err);
        }
    };

    let ct = &settings.ctrader;
    let account_id = ct
        .accounts
        .first()
        .map(|a| a.account_id.clone())
        .unwrap_or_else(|| "(none)".to_string());
    let environment = match ct.environment {
        crate::app_services::broker_config::CTraderBrokerEnvironment::Demo => "Demo",
        crate::app_services::broker_config::CTraderBrokerEnvironment::Live => "Live",
    };
    // `connected` derives from whether the bridge has filled
    // `AppApiState.account`. That field only gets set on a successful
    // full 5-message handshake — so it's the strongest "yes, we
    // actually have a working session" signal we have without adding
    // dedicated heartbeat tracking.
    let connected = state.account().await.is_some();

    let client_id_prefix = ct
        .client_id
        .split_once('_')
        .map(|(prefix, _)| format!("{prefix}_…"))
        .unwrap_or_else(|| "(unset)".to_string());

    Json(BrokerStatusDto {
        adapter: "cTrader".to_string(),
        environment: environment.to_string(),
        account_id,
        connected,
        client_id_prefix,
    })
    .into_response()
}

// ─── /data/bootstrap ──────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DataBootstrapDto {
    pub data_dir: String,
    /// Whether the configured data dir actually exists on disk.
    pub data_dir_exists: bool,
    /// First-level symbol directories discovered, sorted alphabetically.
    pub symbols: Vec<String>,
    /// Total file count under data_dir (1-level walk). Gives the
    /// operator a one-glance read on "do I have any history at all".
    pub file_count: usize,
    /// mtime of the most-recently-touched file in data_dir, as a
    /// Unix-millis stamp. `None` if the dir is empty or doesn't exist.
    pub last_touched_unix_ms: Option<u64>,
}

pub async fn data_bootstrap(State(state): State<AppApiState>) -> Response {
    // F-553/F-576 closure (2026-05-25): config path threaded from CLI.
    let config_path = state.config_path().to_path_buf();
    let result = tokio::task::spawn_blocking(move || {
        let settings = Settings::from_yaml(&config_path)
            .map_err(|e| anyhow::anyhow!("{} not loadable: {e}", config_path.display()))?;
        let dir = settings.system.data_dir.clone();
        scan_data_dir(dir)
    })
    .await;

    match result {
        Ok(Ok(dto)) => Json(dto).into_response(),
        Ok(Err(err)) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not read the data inventory. Check the data directory in Settings → Data.",
            &err,
        ),
        Err(join_err) => internal_panic("Loading the data inventory", join_err),
    }
}

fn scan_data_dir(dir: PathBuf) -> anyhow::Result<DataBootstrapDto> {
    let data_dir_str = dir.display().to_string();
    if !dir.exists() {
        return Ok(DataBootstrapDto {
            data_dir: data_dir_str,
            data_dir_exists: false,
            symbols: Vec::new(),
            file_count: 0,
            last_touched_unix_ms: None,
        });
    }

    let mut symbols = Vec::new();
    let mut file_count = 0usize;
    let mut latest_mtime: Option<SystemTime> = None;

    if let Ok(read_dir) = std::fs::read_dir(&dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    // Only the `symbol=XXX` directories are actual
                    // market-data symbols. Co-located dirs (`forex-ai`,
                    // `neoethos`, `news`, `symbol_metadata`, ...) are
                    // produced by other modules and must not be
                    // surfaced as tradeable symbols.
                    if let Some(symbol) = name.strip_prefix("symbol=") {
                        symbols.push(symbol.to_string());
                    }
                }
                // 1-level deep file count + mtime sweep so we don't
                // walk the whole tree on every request.
                if let Ok(sub) = std::fs::read_dir(&path) {
                    for inner in sub.flatten() {
                        if let Ok(meta) = inner.metadata() {
                            if meta.is_file() {
                                file_count += 1;
                            }
                            if let Ok(mtime) = meta.modified() {
                                latest_mtime = Some(match latest_mtime {
                                    Some(prev) if prev > mtime => prev,
                                    _ => mtime,
                                });
                            }
                        }
                    }
                }
            } else if path.is_file() {
                file_count += 1;
                if let Ok(meta) = entry.metadata() {
                    if let Ok(mtime) = meta.modified() {
                        latest_mtime = Some(match latest_mtime {
                            Some(prev) if prev > mtime => prev,
                            _ => mtime,
                        });
                    }
                }
            }
        }
    }
    symbols.sort();

    let last_touched_unix_ms = latest_mtime
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64);

    Ok(DataBootstrapDto {
        data_dir: data_dir_str,
        data_dir_exists: true,
        symbols,
        file_count,
        last_touched_unix_ms,
    })
}
