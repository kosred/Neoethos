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
use neoethos_data::{CanonicalDatasetIdentity, DatasetDiscovery};

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
    /// 0..=100 completion PERCENT for the active discovery run; 0 when
    /// idle. The internal `JobProgress::percent` is a 0.0..=1.0 fraction;
    /// this DTO multiplies by 100 because the UI renders the value
    /// directly with a `%` suffix. (Bug fix 2026-07-11: the raw fraction
    /// used to be forwarded, so a search at 0.78 displayed as "1%" — the
    /// operator's "stuck at 1%" report.)
    pub discovery_percent: f64,
    /// The live `(name, value)` counters the discovery job accumulates
    /// (candidates evaluated, generations done, …). Empty when idle.
    pub discovery_counters: Vec<EngineCounterDto>,
    /// Live machine-resource readout so the UI can show what discovery is
    /// consuming (operator visibility — the run used to be a black box).
    /// Total / currently-available physical RAM, in GB.
    pub ram_total_gb: f64,
    pub ram_available_gb: f64,
    /// On-disk size of active Vortex feature-run scratch data (MB). This is 0
    /// when every feature block fits in RAM. Each run owns a lease-backed
    /// directory that is reclaimed after the final consumer releases it.
    pub feature_store_mb: u64,
}

/// Sum regular files under the only production feature scratch root without
/// following symlinks. Vortex is the sole shared feature format, so the status
/// endpoint reads the same lease-backed root as the production writer.
fn feature_store_disk_mb() -> u64 {
    super::feature_store_disk::vortex_feature_store_disk_mb(
        &neoethos_data::vortex_feature_run_root(),
    )
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
        // Fraction → percent (see DTO field doc — fixes the "stuck at 1%" display).
        discovery_percent: discovery_percent * 100.0,
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
    /// Active broker adapter ("cTrader"). Picked from the runtime
    /// broker_credentials.toml.
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
    /// Symbols represented by canonical manifest-backed datasets.
    pub symbols: Vec<String>,
    /// Number of canonical dataset identities, not a filesystem file count.
    pub dataset_count: usize,
    /// mtime of the most-recently-touched file in data_dir, as a
    /// Unix-millis stamp. `None` if the dir is empty or doesn't exist.
    pub last_touched_unix_ms: Option<u64>,
    /// Authoritative, reversible canonical identities. The desktop must send
    /// one of these exact values back; symbol/timeframe text is not an identity.
    pub datasets: Vec<CanonicalDatasetInventoryDto>,
    /// Raw/import-required/retired/corrupt entries are visible rather than
    /// disappearing into an empty-success inventory.
    pub skipped: Vec<SkippedDatasetInventoryDto>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalDatasetInventoryDto {
    pub dataset_identity: String,
    pub generation: String,
    pub manifest_binding_sha256: String,
    /// Authoritative source classification so the desktop never decodes the
    /// opaque identity merely to decide whether broker refresh is valid.
    pub source_kind: &'static str,
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub verification: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkippedDatasetInventoryDto {
    pub path: String,
    pub category: String,
    pub detail: String,
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
            dataset_count: 0,
            last_touched_unix_ms: None,
            datasets: Vec::new(),
            skipped: Vec::new(),
        });
    }

    let discovery = DatasetDiscovery::scan_metadata(&dir)?;
    let mut symbols = discovery
        .entries
        .iter()
        .filter_map(|entry| entry.symbol.clone())
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();

    let mut latest_mtime: Option<SystemTime> = None;
    for entry in &discovery.entries {
        if let Ok(mtime) = entry
            .path
            .metadata()
            .and_then(|metadata| metadata.modified())
        {
            latest_mtime = Some(match latest_mtime {
                Some(previous) if previous > mtime => previous,
                _ => mtime,
            });
        }
    }
    let datasets = discovery
        .entries
        .into_iter()
        .map(|entry| {
            let identity = CanonicalDatasetIdentity::from_path_component(&entry.dataset_identity)
                .map_err(|error| {
                anyhow::anyhow!(
                    "validated inventory identity {} no longer decodes: {error}",
                    entry.dataset_identity
                )
            })?;
            Ok(CanonicalDatasetInventoryDto {
                dataset_identity: entry.dataset_identity,
                generation: entry.generation,
                manifest_binding_sha256: entry.manifest_binding_sha256,
                source_kind: if identity.is_broker_real() {
                    "ctrader"
                } else {
                    "external"
                },
                symbol: entry.symbol,
                timeframe: entry.timeframe,
                verification: entry.verification.as_str().to_owned(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let skipped = discovery
        .skipped
        .into_iter()
        .map(|entry| SkippedDatasetInventoryDto {
            path: entry.path.display().to_string(),
            category: entry.reason.category().to_owned(),
            detail: entry.reason.detail().to_owned(),
        })
        .collect::<Vec<_>>();

    let last_touched_unix_ms = latest_mtime
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64);

    Ok(DataBootstrapDto {
        data_dir: data_dir_str,
        data_dir_exists: true,
        symbols,
        dataset_count: datasets.len(),
        last_touched_unix_ms,
        datasets,
        skipped,
    })
}
