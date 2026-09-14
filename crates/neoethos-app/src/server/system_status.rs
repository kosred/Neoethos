//! Read-only desktop engine, broker and exact dataset inventory status.
//! Research admission is distinct from historical financial authority;
//! this module never starts work or acquires an execution lease.

use std::path::PathBuf;
use std::time::SystemTime;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_data::{CanonicalDatasetIdentity, DatasetDiscovery};

use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::canonical_native_discovery::CanonicalNativeResearchTerminalSnapshotV1;
use crate::app_services::jobs::JobKind;

use super::errors::{actionable_error, internal_panic};
use super::state::AppApiState;

// ─── /engines/status ──────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnginesDto {
    pub discovery: String,
    /// Advisory availability of the ResearchOnly start endpoint. Exact dataset,
    /// settings and cost-source checks still run during request admission.
    pub discovery_start_available: bool,
    pub discovery_start_unavailable_reason: Option<String>,
    pub discovery_start_mode: &'static str,
    /// Separate from research admission: never turn a research request into
    /// historical financial authority, training permission, or live trading.
    pub historical_evaluation_available: bool,
    pub historical_evaluation_unavailable_reason: Option<String>,
    pub training: String,
    pub canonical_native_research: CanonicalNativeResearchStatusDto,
    pub auto_trader: String,
    /// Human-readable progress / status line for whichever engine is
    /// currently active. Empty when all three are Idle.
    pub discovery_summary: String,
    pub training_summary: String,
    /// F-340 (Feature #14): live discovery progress mirrored from the
    /// running job's `JobSnapshot`. `discoveryStage` is the coarse phase
    /// label (e.g. `"search_generations"`), `""` when idle.
    pub discovery_stage: String,
    /// 0..=100 completion PERCENT when the engine reports one; null when
    /// indeterminate or idle. Internal `JobProgress::percent` is a 0.0..=1.0 fraction;
    /// this DTO multiplies by 100 because the UI renders the value
    /// directly with a `%` suffix. (Bug fix 2026-07-11: the raw fraction
    /// used to be forwarded, so a search at 0.78 displayed as "1%" — the
    /// operator's "stuck at 1%" report.)
    pub discovery_percent: Option<f64>,
    /// The latest run's observed counters, retained after completion/failure
    /// and cleared when the next Discovery starts. Empty before the first run.
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

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalNativeResearchStatusDto {
    /// True only when this process installed the sealed native startup
    /// authority. On Windows and non-CUDA builds the lane can be Idle but is
    /// intentionally unavailable.
    pub available: bool,
    pub availability_detail: String,
    pub state: String,
    pub stage: String,
    pub percent: f64,
    /// Opaque decimal token for exact cancellation. A string avoids loss of
    /// `u64` precision in JavaScript clients and disappears at terminal.
    pub lease_token: Option<String>,
    pub cancellation_requested: bool,
    pub failure_stage: Option<String>,
    pub failure_code: Option<String>,
    pub failure_detail: Option<String>,
    pub published: Option<CanonicalNativeResearchPublishedStatusDto>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalNativeResearchPublishedStatusDto {
    pub relative_path: String,
    pub byte_count: u64,
    pub file_sha256: String,
    pub evidence_identity_sha256: String,
    pub configured_population: usize,
    pub resolved_population: usize,
    pub population_cap: usize,
    pub hard_growth_cap: usize,
    pub term_cap: usize,
    pub selected_device_ordinal: u32,
    pub engine: String,
    pub parent_h2d_bytes: u64,
    pub adaptive_h2d_bytes: u64,
    pub metric_rows: u64,
    pub metric_bytes: u64,
    pub consumer_completion_confirmed: bool,
    pub replay_identity_sealed: bool,
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

fn summarize_live_engine_states(states: impl IntoIterator<Item = Option<bool>>) -> &'static str {
    let mut unknown = false;
    for running in states {
        match running {
            Some(true) => return "Running",
            Some(false) => {}
            None => unknown = true,
        }
    }
    if unknown { "Unknown" } else { "Idle" }
}

fn auto_trader_status(state: &AppApiState) -> &'static str {
    // Read the actual autonomous registry without waiting behind a start/stop
    // operation on the async reactor. Missing lock access is not proof of Idle.
    let Ok(handles) = state.live_trading.try_lock() else {
        return "Unknown";
    };
    summarize_live_engine_states(handles.iter().map(|handle| {
        handle
            .status
            .try_lock()
            .ok()
            .map(|snapshot| snapshot.running)
    }))
}

/// Engine-state endpoint. Research/training read the ServiceEvent drainer;
/// Auto-Trader reads the same live handles as the autonomous status endpoint.
pub async fn engines(State(state): State<AppApiState>) -> Result<Json<EnginesDto>, Response> {
    engines_with_disk_scan(state, feature_store_disk_mb).await
}

async fn engines_with_disk_scan(
    state: AppApiState,
    scan: impl FnOnce() -> u64 + Send + 'static,
) -> Result<Json<EnginesDto>, Response> {
    // Recursive filesystem IO must not occupy the async status/Stop executor.
    let feature_store_mb = tokio::task::spawn_blocking(scan)
        .await
        .map_err(|error| internal_panic("Reading feature-store disk usage", error))?;
    // Do not combine a new run's state with an older run's counters across awaits.
    let discovery = state.discovery_observation().await;
    let native_slot = state.canonical_native_research_slot_v1().await;
    let native_available = state.canonical_native_startup_authority_v1().is_some();
    let native_availability_detail = if native_available {
        "Sealed canonical CUDA runtime authority is installed for this process."
    } else if !cfg!(target_os = "linux") {
        "Canonical CUDA research is currently supported only by the Linux gpu-nvidia build."
    } else if !cfg!(feature = "gpu-nvidia") {
        "This executable was built without the gpu-nvidia feature."
    } else {
        "The canonical native startup authority was not installed."
    };
    let financial_gate = neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation);
    let discovery_start_unavailable_reason = neoethos_search::active_process_execution_kind_v1()
        .map(|active| format!("{active} is already active. Wait for it to finish or stop it before starting another research run."));
    let discovery_start_available = discovery_start_unavailable_reason.is_none();
    let native_snapshot = native_slot.snapshot();
    let native_terminal = native_slot.terminal();
    let native_failure = native_terminal.and_then(|terminal| terminal.failure());
    let native_published = native_terminal
        .and_then(CanonicalNativeResearchTerminalSnapshotV1::published)
        .map(|published| CanonicalNativeResearchPublishedStatusDto {
            relative_path: published.relative_path().to_owned(),
            byte_count: published.byte_count(),
            file_sha256: published.file_sha256().to_owned(),
            evidence_identity_sha256: published.evidence_identity_sha256().to_owned(),
            configured_population: published.configured_population(),
            resolved_population: published.resolved_population(),
            population_cap: published.population_cap(),
            hard_growth_cap: published.hard_growth_cap(),
            term_cap: published.term_cap(),
            selected_device_ordinal: published.selected_device_ordinal(),
            engine: published.engine().to_owned(),
            parent_h2d_bytes: published.parent_h2d_bytes(),
            adaptive_h2d_bytes: published.adaptive_h2d_bytes(),
            metric_rows: published.metric_rows(),
            metric_bytes: published.metric_bytes(),
            consumer_completion_confirmed: published.consumer_completion_confirmed(),
            replay_identity_sealed: published.replay_identity_sealed(),
        });
    Ok(Json(EnginesDto {
        discovery: discovery.state.as_str().to_string(),
        discovery_start_available,
        discovery_start_unavailable_reason,
        discovery_start_mode: "ResearchOnly",
        historical_evaluation_available: financial_gate.is_ok(),
        historical_evaluation_unavailable_reason: financial_gate
            .err()
            .map(|error| error.to_string()),
        training: state
            .engine_state(JobKind::Training)
            .await
            .as_str()
            .to_string(),
        canonical_native_research: CanonicalNativeResearchStatusDto {
            available: native_available,
            availability_detail: native_availability_detail.to_owned(),
            state: native_snapshot
                .map(|snapshot| snapshot.state().as_str())
                .unwrap_or("Idle")
                .to_owned(),
            stage: native_snapshot
                .map(|snapshot| snapshot.stage().to_owned())
                .unwrap_or_default(),
            percent: native_snapshot
                .map(|snapshot| f64::from(snapshot.percent_basis_points()) / 100.0)
                .unwrap_or(0.0),
            lease_token: native_snapshot
                .filter(|snapshot| !snapshot.state().is_terminal())
                .map(|snapshot| snapshot.lease_token().to_string()),
            cancellation_requested: native_slot.cancellation_requested(),
            failure_stage: native_failure.map(|failure| failure.stable_stage().to_owned()),
            failure_code: native_failure.map(|failure| failure.stable_code().to_owned()),
            failure_detail: native_failure.map(|failure| failure.detail().to_owned()),
            published: native_published,
        },
        auto_trader: auto_trader_status(&state).to_owned(),
        discovery_summary: discovery.summary,
        training_summary: state.engine_summary(JobKind::Training).await,
        discovery_stage: discovery.stage,
        // Fraction → percent (see DTO field doc — fixes the "stuck at 1%" display).
        discovery_percent: discovery.percent.map(|fraction| fraction * 100.0),
        discovery_counters: discovery
            .counters
            .into_iter()
            .map(|(name, value)| EngineCounterDto { name, value })
            .collect(),
        ram_total_gb: neoethos_core::total_memory_bytes() as f64 / 1e9,
        ram_available_gb: neoethos_core::available_memory_bytes() as f64 / 1e9,
        feature_store_mb,
    }))
}

#[cfg(test)]
mod research_status_tests {
    use super::*;
    use crate::app_services::jobs::CancellationFlag;
    use neoethos_search::{ProcessExecutionKindV1, try_acquire_process_execution_lease_v1};

    #[tokio::test(flavor = "current_thread")]
    async fn engine_disk_scan_runs_off_the_async_worker_and_preserves_the_measurement() {
        let async_thread = std::thread::current().id();
        let Json(dto) = engines_with_disk_scan(AppApiState::new(), move || {
            assert_ne!(std::thread::current().id(), async_thread);
            23
        })
        .await
        .unwrap();
        assert_eq!(serde_json::to_value(dto).unwrap()["featureStoreMb"], 23);
    }

    #[tokio::test]
    async fn engine_disk_scan_join_failure_is_http_error_not_zero_usage() {
        let response = engines_with_disk_scan(AppApiState::new(), || {
            panic!("synthetic feature disk scan failure")
        })
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let bytes = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            error["error"]
                .as_str()
                .unwrap()
                .contains("feature-store disk usage")
        );
        assert!(
            error["detail"]
                .as_str()
                .unwrap()
                .contains("synthetic feature disk scan failure")
        );
        assert!(error.get("featureStoreMb").is_none());
    }

    #[test]
    fn auto_trader_status_never_turns_running_or_unobservable_handles_into_idle() {
        assert_eq!(summarize_live_engine_states([]), "Idle");
        assert_eq!(summarize_live_engine_states([Some(false)]), "Idle");
        assert_eq!(summarize_live_engine_states([None, Some(false)]), "Unknown");
        assert_eq!(summarize_live_engine_states([None, Some(true)]), "Running");
        assert_eq!(summarize_live_engine_states([Some(true), None]), "Running");
        let state = AppApiState::new();
        assert_eq!(auto_trader_status(&state), "Idle");
        let _held_by_start_or_stop = state.live_trading.lock().unwrap();
        assert_eq!(auto_trader_status(&state), "Unknown");
    }

    #[tokio::test]
    async fn research_readiness_does_not_acquire_or_bypass_an_execution_lease() {
        let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
        let state = AppApiState::new();
        let idle = engines(State(state.clone())).await.unwrap().0;
        assert!(idle.discovery_start_available);
        assert_eq!(idle.discovery_start_mode, "ResearchOnly");
        assert!(!idle.historical_evaluation_available);
        assert!(idle.historical_evaluation_unavailable_reason.is_some());
        assert!(neoethos_search::active_process_execution_kind_v1().is_none());

        let mut lease = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery)
            .expect("readiness did not take the lease");
        let token = lease.token();
        for kind in [
            ProcessExecutionKindV1::Discovery,
            ProcessExecutionKindV1::Training,
        ] {
            if kind == ProcessExecutionKindV1::Training {
                lease.transition_discovery_to_training_v1().unwrap();
            }
            let busy = engines(State(state.clone())).await.unwrap().0;
            assert!(!busy.discovery_start_available);
            assert!(
                busy.discovery_start_unavailable_reason
                    .unwrap()
                    .contains(&kind.to_string())
            );
            assert!(!busy.historical_evaluation_available);
            assert_eq!(
                neoethos_search::active_process_execution_kind_v1(),
                Some(kind)
            );
            assert_eq!(lease.token(), token);
            assert!(
                try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Migration).is_err()
            );
        }
        drop(lease);
        let ready = engines(State(state)).await.unwrap().0;
        assert!(ready.discovery_start_available);
        assert!(!ready.historical_evaluation_available);
    }

    #[tokio::test]
    async fn discovery_progress_distinguishes_unknown_zero_fraction_and_terminal_reset() {
        let state = AppApiState::new();
        state
            .install_engine(JobKind::Discovery, CancellationFlag::new(), 1)
            .await;
        for (fraction, expected) in [
            (None, None),
            (Some(0.0), Some(0.0)),
            (Some(0.375), Some(37.5)),
            (Some(2.0), Some(100.0)),
            (Some(-1.0), Some(0.0)),
            (Some(f64::NAN), None),
            (Some(f64::INFINITY), None),
        ] {
            let mut snapshot = crate::app_services::jobs::JobSnapshot::new(JobKind::Discovery);
            snapshot.state = crate::app_services::jobs::JobState::Running;
            snapshot.progress.stage = "search_generations".to_owned();
            snapshot.progress.percent = fraction.map(|value| value as f32);
            snapshot.report.counters = vec![("candidates_evaluated".to_owned(), 20_000)];
            state
                .update_engine_snapshot(JobKind::Discovery, &snapshot, 1)
                .await;
            let dto = engines(State(state.clone())).await.unwrap().0;
            assert_eq!(dto.discovery_percent, expected);
            assert_eq!(dto.discovery_counters[0].value, 20_000);
            assert_eq!(
                serde_json::to_value(dto).unwrap()["discoveryPercent"],
                serde_json::json!(expected)
            );
        }
        let mut completed = crate::app_services::jobs::JobSnapshot::new(JobKind::Discovery);
        completed.state = crate::app_services::jobs::JobState::Succeeded;
        completed.report.summary = "research complete".to_owned();
        state
            .update_engine_snapshot(JobKind::Discovery, &completed, 1)
            .await;
        let dto = engines(State(state.clone())).await.unwrap().0;
        assert_eq!(dto.discovery_summary, "research complete");
        assert!(dto.discovery_percent.is_none());
        assert!(dto.discovery_stage.is_empty());
        assert_eq!(dto.discovery_counters[0].value, 20_000);

        state
            .install_engine(JobKind::Discovery, CancellationFlag::new(), 2)
            .await;
        assert!(
            engines(State(state.clone()))
                .await
                .unwrap()
                .0
                .discovery_counters
                .is_empty()
        );
        let mut failed = crate::app_services::jobs::JobSnapshot::new(JobKind::Discovery);
        failed.state = crate::app_services::jobs::JobState::Failed;
        state
            .update_engine_snapshot(JobKind::Discovery, &failed, 2)
            .await;
        assert_eq!(
            state.engine_progress(JobKind::Discovery).await,
            (String::new(), None, Vec::new())
        );
    }
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
    /// The uniquely enabled, valid execution account; `(none)` if selection is invalid.
    pub account_id: String,
    /// A recent successful observation of this exact account/environment with no
    /// subsequent refresh failure. This is status, not permission to trade.
    pub connected: bool,
    pub last_snapshot_at_unix_ms: Option<i64>,
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

    let (account, failure) = state.account_observation().await;
    Json(broker_status_from_observation(
        &settings,
        account.as_ref(),
        failure.is_some(),
        chrono::Utc::now().timestamp_millis(),
    ))
    .into_response()
}

fn broker_status_from_observation(
    settings: &crate::app_services::broker_config::BrokerSettingsState,
    account: Option<&super::state::AccountSnapshotPayload>,
    refresh_failed: bool,
    now_ms: i64,
) -> BrokerStatusDto {
    use crate::app_services::ctrader_live_auth::CTraderEnvironment;
    let ct = &settings.ctrader;
    let selected_account = super::bridge::execution_account_id(settings).ok();
    let expected_account_id = selected_account
        .as_deref()
        .and_then(|id| id.parse::<i64>().ok());
    let account_id = selected_account.unwrap_or_else(|| "(none)".to_string());
    let (environment, expected_environment) = match ct.environment {
        crate::app_services::broker_config::CTraderBrokerEnvironment::Demo => {
            ("Demo", CTraderEnvironment::Demo)
        }
        crate::app_services::broker_config::CTraderBrokerEnvironment::Live => {
            ("Live", CTraderEnvironment::Live)
        }
    };
    // A late successful refresh may belong to the previous account or endpoint.
    // Its freshness is not an observation of the newly selected account.
    let last_snapshot_at_unix_ms = account
        .filter(|snapshot| {
            expected_account_id == Some(snapshot.source_account_id)
                && snapshot.source_environment == expected_environment
        })
        .map(|snapshot| snapshot.fetched_at_unix_ms);
    // The bridge polls every 5 seconds and retires retained snapshots
    // after 3 consecutive failures. The status must turn disconnected
    // on the first recorded failure, and also expire if polling stalls.
    // This display-only horizon changes no risk or execution admission.
    const STATUS_SNAPSHOT_MAX_AGE_MS: i64 = 15_000;
    let connected = !refresh_failed
        && last_snapshot_at_unix_ms.is_some_and(|observed| {
            observed > 0
                && now_ms
                    .checked_sub(observed)
                    .is_some_and(|age| (0..=STATUS_SNAPSHOT_MAX_AGE_MS).contains(&age))
        });

    let client_id_prefix = ct
        .client_id
        .split_once('_')
        .map(|(prefix, _)| format!("{prefix}_…"))
        .unwrap_or_else(|| "(unset)".to_string());

    BrokerStatusDto {
        adapter: "cTrader".to_string(),
        environment: environment.to_string(),
        account_id,
        connected,
        last_snapshot_at_unix_ms,
        client_id_prefix,
    }
}

#[cfg(test)]
mod broker_status_tests {
    use super::*;
    use crate::app_services::broker_config::{
        BrokerAccountTarget, BrokerSettingsState, CTraderBrokerEnvironment,
    };
    use crate::app_services::ctrader_live_auth::CTraderEnvironment;
    use crate::server::state::AccountSnapshotPayload;

    fn configured_accounts() -> BrokerSettingsState {
        let mut settings = BrokerSettingsState::default();
        settings.ctrader.environment = CTraderBrokerEnvironment::Demo;
        settings.ctrader.accounts = vec![
            BrokerAccountTarget {
                account_id: "11".into(),
                ..Default::default()
            },
            BrokerAccountTarget {
                account_id: "22".into(),
                enabled_for_execution: true,
                ..Default::default()
            },
        ];
        settings
    }

    fn observed(
        account_id: i64,
        environment: CTraderEnvironment,
        timestamp: i64,
    ) -> AccountSnapshotPayload {
        AccountSnapshotPayload {
            source_account_id: account_id,
            source_environment: environment,
            balance: 1000.0,
            equity: 1000.0,
            free_margin: 1000.0,
            used_margin: 0.0,
            currency: "EUR".into(),
            fetched_at_unix_ms: timestamp,
            positions: Vec::new(),
        }
    }

    #[test]
    fn broker_status_reports_the_same_account_as_the_execution_selector() {
        let mut settings = configured_accounts();
        let snapshot = observed(22, CTraderEnvironment::Demo, 100_000);
        let status = broker_status_from_observation(&settings, Some(&snapshot), false, 100_001);
        assert_eq!(status.account_id, "22");
        assert!(status.connected);
        settings.ctrader.accounts[1].enabled_for_execution = false;
        let unselected = broker_status_from_observation(&settings, Some(&snapshot), false, 100_001);
        assert_eq!(unselected.account_id, "(none)");
        assert!(!unselected.connected);
        assert_eq!(unselected.last_snapshot_at_unix_ms, None);
        settings.ctrader.accounts.clear();
        let empty = broker_status_from_observation(&settings, Some(&snapshot), false, 100_001);
        assert_eq!(empty.account_id, "(none)");
        assert!(!empty.connected);
    }

    #[test]
    fn failed_missing_stalled_or_future_account_observations_are_not_connected() {
        let settings = configured_accounts();
        for (timestamp, failed) in [
            (Some(100_000), true),
            (None, false),
            (Some(0), false),
            (Some(84_999), false),
            (Some(100_001), false),
            (Some(i64::MIN), false),
        ] {
            let snapshot = timestamp.map(|value| observed(22, CTraderEnvironment::Demo, value));
            let status =
                broker_status_from_observation(&settings, snapshot.as_ref(), failed, 100_000);
            assert!(
                !status.connected,
                "observation={timestamp:?}, failed={failed}"
            );
            assert_eq!(status.last_snapshot_at_unix_ms, timestamp);
        }
        let boundary = observed(22, CTraderEnvironment::Demo, 85_000);
        assert!(
            broker_status_from_observation(&settings, Some(&boundary), false, 100_000).connected
        );
        let overflow = observed(22, CTraderEnvironment::Demo, 1);
        assert!(
            !broker_status_from_observation(&settings, Some(&overflow), false, i64::MIN).connected
        );
    }

    #[test]
    fn fresh_snapshot_never_connects_a_different_account_or_environment() {
        let mut settings = configured_accounts();
        for (configured, expected, other) in [
            (
                CTraderBrokerEnvironment::Demo,
                CTraderEnvironment::Demo,
                CTraderEnvironment::Live,
            ),
            (
                CTraderBrokerEnvironment::Live,
                CTraderEnvironment::Live,
                CTraderEnvironment::Demo,
            ),
        ] {
            settings.ctrader.environment = configured;
            for snapshot in [
                observed(11, expected, 100_000),
                observed(22, other, 100_000),
            ] {
                let status =
                    broker_status_from_observation(&settings, Some(&snapshot), false, 100_001);
                assert_eq!(status.account_id, "22");
                assert_eq!(status.environment, configured.as_str());
                assert!(!status.connected);
                assert_eq!(status.last_snapshot_at_unix_ms, None);
            }
            let snapshot = observed(22, expected, 100_000);
            assert!(
                broker_status_from_observation(&settings, Some(&snapshot), false, 100_001)
                    .connected
            );
        }
    }

    #[test]
    fn ambiguous_or_invalid_selection_never_connects_a_retained_snapshot() {
        let snapshot = observed(22, CTraderEnvironment::Demo, 100_000);
        let mut settings = configured_accounts();
        settings.ctrader.accounts[0].enabled_for_execution = true;
        assert!(
            !broker_status_from_observation(&settings, Some(&snapshot), false, 100_001).connected
        );
        settings.ctrader.accounts[0].enabled_for_execution = false;
        for invalid in ["", "not-numeric", "0", "-22", "9223372036854775808"] {
            settings.ctrader.accounts[1].account_id = invalid.into();
            assert!(
                !broker_status_from_observation(&settings, Some(&snapshot), false, 100_001)
                    .connected
            );
        }
    }

    #[tokio::test]
    async fn cached_identity_is_preserved_for_late_old_refresh_and_matching_recovery() {
        let settings = configured_accounts();
        let state = AppApiState::new();
        state
            .set_account(observed(11, CTraderEnvironment::Demo, 100_000))
            .await;
        let (old, failure) = state.account_observation().await;
        assert!(
            !broker_status_from_observation(&settings, old.as_ref(), failure.is_some(), 100_001)
                .connected
        );
        state
            .set_account_failure(&anyhow::anyhow!("refresh rejected"))
            .await;
        let (retained, failure) = state.account_observation().await;
        assert_eq!(retained.unwrap().source_account_id, 11);
        assert!(failure.is_some());
        state
            .set_account(observed(22, CTraderEnvironment::Demo, 100_002))
            .await;
        let (matching, failure) = state.account_observation().await;
        assert!(
            broker_status_from_observation(
                &settings,
                matching.as_ref(),
                failure.is_some(),
                100_003
            )
            .connected
        );
        let wire = serde_json::to_value(matching.unwrap()).unwrap();
        assert!(wire.get("sourceAccountId").is_none());
        assert!(wire.get("sourceEnvironment").is_none());
    }
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
        .filter(|entry| !is_app_inventory_sidecar(&dir, entry))
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

/// The data root also holds application state. The generic data library
/// correctly refuses loose files as runtime datasets, but the desktop must
/// not tell users to import its own supervisor config or metadata as prices.
/// Keep this exact root-level allowlist at the application boundary; arbitrary
/// JSON/JSONL files and every rejected canonical generation remain visible.
fn is_app_inventory_sidecar(
    root: &std::path::Path,
    entry: &neoethos_data::core::discover::SkippedFile,
) -> bool {
    entry.path.parent() == Some(root)
        && matches!(
            entry.reason,
            neoethos_data::core::discover::SkipReason::ImportRequired(_)
        )
        && matches!(
            entry.path.file_name().and_then(|name| name.to_str()),
            Some(
                "symbol_metadata.json"
                    | "strategy_blacklist.json"
                    | "supervisor.json"
                    | "supervisor_log.jsonl"
                    | "spread_stats.json"
            )
        )
}

#[cfg(test)]
mod inventory_sidecar_tests {
    use super::*;

    #[test]
    fn app_sidecars_are_not_price_imports_but_unknown_json_remains_visible() {
        let root = std::env::temp_dir().join(format!(
            "neoethos-inventory-sidecars-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(root.clone());
        for name in [
            "symbol_metadata.json",
            "supervisor.json",
            "supervisor_log.jsonl",
            "spread_stats.json",
            "EURUSD.json",
        ] {
            std::fs::write(root.join(name), "{}").unwrap();
        }
        std::fs::write(root.join("strategy_blacklist.json"), "[]\n").unwrap();
        use neoethos_data::core::discover::{SkipReason, SkippedFile};
        for entry in [
            SkippedFile {
                path: root.join("nested/strategy_blacklist.json"),
                reason: SkipReason::ImportRequired("not root application state".into()),
            },
            SkippedFile {
                path: root.join("strategy_blacklist.json"),
                reason: SkipReason::Unreadable("I/O failure must remain visible".into()),
            },
        ] {
            assert!(!is_app_inventory_sidecar(&root, &entry));
        }
        std::fs::create_dir(root.join("d1-invalid")).unwrap();
        let inventory = scan_data_dir(root).unwrap();
        assert_eq!(inventory.dataset_count, 0);
        assert_eq!(inventory.skipped.len(), 2);
        assert!(inventory.skipped.iter().any(
            |entry| entry.path.ends_with("EURUSD.json") && entry.category == "import_required"
        ));
        assert!(
            inventory
                .skipped
                .iter()
                .any(|entry| entry.path.ends_with("d1-invalid")
                    && entry.category == "invalid_canonical_identity")
        );
    }
}
