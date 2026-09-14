//! Lease-owning typed adapters for the legacy Discovery/Training workers.
//!
//! These are intentionally crate-private. Frontends pass lightweight intent;
//! Settings and canonical dataset resolution happen only after admission.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use neoethos_core::Settings;
use neoethos_data::{
    CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe, ExactDatasetGenerationConflict,
    SelectedDatasetGenerationV1,
};
#[cfg(test)]
use neoethos_search::DiscoveryConfig;
use neoethos_search::{
    ProcessExecutionBusyV1, ProcessExecutionKindV1, ProcessExecutionLeaseV1, PropFirmRiskRules,
    try_acquire_process_execution_lease_v1,
};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use crate::app_services::ServiceEvent;
use crate::app_services::discovery::{
    DirectTimeframeAcquisitionRequired, DiscoveryRequest, DiscoverySettingsSource,
    pin_current_discovery_input, pin_discovery_input, resolve_unique_background_dataset_identity,
    start_discovery_job,
};
use crate::app_services::jobs::{CancellationFlag, JobKind, JobSnapshot, JobState};
use crate::app_services::training::{
    TrainingRequest, handoff, start_discovery_training_job, start_strategy_research_job,
    start_training_job,
};
use crate::server::state::AppApiState;

#[cfg(test)]
use super::EngineRunState;

const MAX_TYPED_EXECUTION_DETAIL_BYTES_V1: usize = 1_024;

type TypedLegacyAdmissionSenderV1 =
    oneshot::Sender<Result<TypedLegacyExecutionAdmissionV1, TypedLegacyExecutionAdmissionErrorV1>>;

pub(crate) use crate::app_services::discovery::{
    TypedDiscoveryGenerationOverrideV1, TypedDiscoveryOverridesV1,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TypedHigherTimeframePolicyV1 {
    Configured,
    Exact(Vec<CanonicalTimeframe>),
}

impl TypedHigherTimeframePolicyV1 {
    fn resolve_for_discovery(
        &self,
        settings: &Settings,
        base: CanonicalTimeframe,
    ) -> Result<Vec<String>, TypedLegacyExecutionAdmissionErrorV1> {
        let configured = matches!(self, Self::Configured);
        let timeframes = match self {
            TypedHigherTimeframePolicyV1::Configured => {
                let system = &settings.system;
                let active = if system.multi_resolution_enabled
                    && !system.multi_resolution_timeframes.is_empty()
                {
                    &system.multi_resolution_timeframes
                } else {
                    &system.higher_timeframes
                };
                active
                    .iter()
                    .map(|raw| raw.trim().to_uppercase())
                    .filter(|label| !label.is_empty())
                    .map(|label| {
                        label.parse::<CanonicalTimeframe>().map_err(|error| {
                            TypedLegacyExecutionAdmissionErrorV1::BadRequest(format!(
                                "invalid configured Discovery timeframe {label:?}: {error}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            Self::Exact(timeframes) => timeframes.clone(),
        };
        let mut higher = Vec::with_capacity(timeframes.len());
        for timeframe in timeframes {
            if timeframe <= base {
                if configured {
                    // The shared model-context resolver deliberately retains lower
                    // resolutions. Discovery pins a strictly top-down input ladder;
                    // do not change model context or silently rewrite exact API intent.
                    tracing::info!(%base, %timeframe, "excluded non-higher configured timeframe from Discovery top-down context");
                    continue;
                }
                return Err(TypedLegacyExecutionAdmissionErrorV1::BadRequest(format!(
                    "higher timeframe {timeframe} must be strictly above base {base}"
                )));
            }
            if higher.contains(&timeframe) {
                return Err(TypedLegacyExecutionAdmissionErrorV1::BadRequest(format!(
                    "duplicate higher timeframe {timeframe}"
                )));
            }
            higher.push(timeframe);
        }
        Ok(higher
            .into_iter()
            .map(|timeframe| timeframe.as_str().to_owned())
            .collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TypedDiscoverySettingsGateV1 {
    None,
    RequireAutoRediscoveryEnabled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TypedDiscoveryDatasetPolicyV1 {
    Current,
    Exact(SelectedDatasetGenerationV1),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TypedDiscoveryExecutionIntentV1 {
    pub(crate) symbol: String,
    pub(crate) base_timeframe: CanonicalTimeframe,
    pub(crate) higher_timeframes: TypedHigherTimeframePolicyV1,
    pub(crate) overrides: TypedDiscoveryOverridesV1,
    pub(crate) settings_gate: TypedDiscoverySettingsGateV1,
    pub(crate) dataset_policy: TypedDiscoveryDatasetPolicyV1,
    pub(crate) training_after_success: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TypedTrainingSelectionPolicyV1 {
    DiscoveryHandoff {
        identity_sha256: String,
    },
    StrategyResearchHandoff {
        identity_sha256: String,
    },
    Exact {
        symbol: String,
        base_timeframe: CanonicalTimeframe,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TypedTrainingExecutionIntentV1 {
    pub(crate) selection: TypedTrainingSelectionPolicyV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TypedLegacyExecutionAdmissionV1 {
    Discovery {
        selected_generation: SelectedDatasetGenerationV1,
    },
    Training {
        symbol: String,
        base_timeframe: CanonicalTimeframe,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TypedLegacyExecutionAdmissionErrorV1 {
    BadRequest(String),
    Conflict(String),
    UnprocessableEntity(String),
    ServiceUnavailable(String),
    Cancelled(String),
    Internal(String),
}

impl TypedLegacyExecutionAdmissionErrorV1 {
    pub(crate) fn detail(&self) -> &str {
        match self {
            Self::BadRequest(detail)
            | Self::Conflict(detail)
            | Self::UnprocessableEntity(detail)
            | Self::ServiceUnavailable(detail)
            | Self::Cancelled(detail)
            | Self::Internal(detail) => detail,
        }
    }
}

impl fmt::Display for TypedLegacyExecutionAdmissionErrorV1 {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output.write_str(self.detail())
    }
}

impl std::error::Error for TypedLegacyExecutionAdmissionErrorV1 {}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TypedLegacyExecutionSnapshotV1 {
    lease_token: u64,
    lease_kind: ProcessExecutionKindV1,
    job_snapshot: JobSnapshot,
}

impl TypedLegacyExecutionSnapshotV1 {
    fn new(
        lease_token: u64,
        lease_kind: ProcessExecutionKindV1,
        job_snapshot: JobSnapshot,
    ) -> Self {
        Self {
            lease_token,
            lease_kind,
            job_snapshot,
        }
    }

    pub(crate) const fn state(&self) -> JobState {
        self.job_snapshot.state
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TypedLegacyExecutionTerminalV1 {
    Succeeded {
        final_snapshot: JobSnapshot,
        lease_token: u64,
        completed_kind: JobKind,
    },
    Failed {
        final_snapshot: JobSnapshot,
        lease_token: u64,
        detail: String,
    },
    Cancelled {
        final_snapshot: JobSnapshot,
        lease_token: u64,
    },
    WorkerPanicked {
        lease_token: u64,
        job_kind: JobKind,
        detail: String,
    },
}

#[derive(Debug)]
pub(crate) enum TypedLegacyExecutionStartErrorV1 {
    Busy(ProcessExecutionBusyV1),
    RuntimeUnavailable(String),
}

impl fmt::Display for TypedLegacyExecutionStartErrorV1 {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy(error) => error.fmt(output),
            Self::RuntimeUnavailable(detail) => {
                write!(output, "typed engine runtime is unavailable: {detail}")
            }
        }
    }
}

impl std::error::Error for TypedLegacyExecutionStartErrorV1 {}

pub(crate) struct TypedLegacyExecutionJobHandleV1 {
    lease_token: u64,
    initial_kind: JobKind,
    cancel: CancellationFlag,
    snapshots: watch::Receiver<TypedLegacyExecutionSnapshotV1>,
    admission: oneshot::Receiver<
        Result<TypedLegacyExecutionAdmissionV1, TypedLegacyExecutionAdmissionErrorV1>,
    >,
    terminal: oneshot::Receiver<TypedLegacyExecutionTerminalV1>,
    worker: JoinHandle<()>,
}

impl TypedLegacyExecutionJobHandleV1 {
    pub(crate) fn cancel(&self) {
        self.cancel.request();
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.worker.is_finished()
    }

    pub(crate) fn snapshot_receiver_mut(
        &mut self,
    ) -> &mut watch::Receiver<TypedLegacyExecutionSnapshotV1> {
        &mut self.snapshots
    }

    pub(crate) async fn await_admission_v1(
        &mut self,
    ) -> Result<TypedLegacyExecutionAdmissionV1, TypedLegacyExecutionAdmissionErrorV1> {
        (&mut self.admission).await.unwrap_or_else(|_| {
            Err(TypedLegacyExecutionAdmissionErrorV1::Internal(
                "typed engine worker exited before admission evidence".to_owned(),
            ))
        })
    }

    pub(crate) async fn await_terminal(self) -> TypedLegacyExecutionTerminalV1 {
        let Self {
            lease_token,
            initial_kind: _,
            cancel: _,
            snapshots,
            admission: _,
            terminal,
            worker,
        } = self;
        let signalled = terminal.await.ok();
        let joined = worker.await;
        let job_kind = snapshots.borrow().job_snapshot.kind;
        match joined {
            Ok(()) => signalled.unwrap_or(TypedLegacyExecutionTerminalV1::WorkerPanicked {
                lease_token,
                job_kind,
                detail: "typed engine worker exited without terminal evidence".to_owned(),
            }),
            Err(error) => TypedLegacyExecutionTerminalV1::WorkerPanicked {
                lease_token,
                job_kind,
                detail: bounded_detail_v1(error),
            },
        }
    }
}

pub(crate) fn start_typed_discovery_execution_v1(
    state: AppApiState,
    intent: TypedDiscoveryExecutionIntentV1,
) -> Result<TypedLegacyExecutionJobHandleV1, TypedLegacyExecutionStartErrorV1> {
    require_runtime_v1()?;
    let lease = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery)
        .map_err(TypedLegacyExecutionStartErrorV1::Busy)?;
    Ok(spawn_discovery_worker_v1(state, intent, lease))
}

pub(crate) fn start_typed_training_execution_v1(
    state: AppApiState,
    intent: TypedTrainingExecutionIntentV1,
) -> Result<TypedLegacyExecutionJobHandleV1, TypedLegacyExecutionStartErrorV1> {
    require_runtime_v1()?;
    let lease = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Training)
        .map_err(TypedLegacyExecutionStartErrorV1::Busy)?;
    Ok(spawn_training_worker_v1(state, intent, lease))
}

fn require_runtime_v1() -> Result<(), TypedLegacyExecutionStartErrorV1> {
    tokio::runtime::Handle::try_current()
        .map(|_| ())
        .map_err(|error| {
            TypedLegacyExecutionStartErrorV1::RuntimeUnavailable(bounded_detail_v1(error))
        })
}

fn spawn_discovery_worker_v1(
    state: AppApiState,
    intent: TypedDiscoveryExecutionIntentV1,
    lease: ProcessExecutionLeaseV1,
) -> TypedLegacyExecutionJobHandleV1 {
    let lease_token = lease.token();
    let cancel = CancellationFlag::new();
    let initial = queued_snapshot_v1(JobKind::Discovery);
    let (snapshot_tx, snapshots) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
        lease_token,
        ProcessExecutionKindV1::Discovery,
        initial,
    ));
    let (terminal_tx, terminal) = oneshot::channel();
    let (admission_tx, admission) = oneshot::channel();
    let worker_cancel = cancel.clone();
    let worker = tokio::spawn(async move {
        // One owner retains the same token through Discovery and its exact
        // candidate Training continuation. Never reacquire between phases.
        let mut lease = lease;
        state
            .install_engine(JobKind::Discovery, worker_cancel.clone(), lease_token)
            .await;
        let discovery = prepare_discovery_request_v1(&state, &intent, &worker_cancel).await;
        let final_result = match discovery {
            Ok((request, receipt)) => {
                let expected_series = request.pinned_input.receipt().clone();
                let discovery_result = run_discovery_job_v1(
                    &state,
                    request,
                    &worker_cancel,
                    &snapshot_tx,
                    lease_token,
                    admission_tx,
                    receipt,
                )
                .await;
                if intent.training_after_success {
                    match discovery_result {
                        Ok(snapshot) => {
                            continue_discovery_training_v1(
                                snapshot,
                                &worker_cancel,
                                &mut lease,
                                &snapshot_tx,
                                |training_intent| {
                                    run_training_intent_v1(
                                        &state,
                                        training_intent,
                                        &worker_cancel,
                                        &snapshot_tx,
                                        lease_token,
                                        None,
                                        Some(expected_series),
                                    )
                                },
                            )
                            .await
                        }
                        Err(snapshot) => Err(snapshot),
                    }
                } else {
                    discovery_result
                }
            }
            Err(error) => {
                let _ = admission_tx.send(Err(error.clone()));
                Err(preparation_error_snapshot_v1(
                    JobKind::Discovery,
                    &worker_cancel,
                    error,
                ))
            }
        };
        let final_snapshot = match final_result {
            Ok(snapshot) | Err(snapshot) => snapshot,
        };
        let completed_kind = final_snapshot.kind;
        let terminal_value = terminal_from_snapshot_v1(final_snapshot, lease_token, completed_kind);
        persist_terminal_state_v1(&state, &terminal_value, JobKind::Discovery, lease_token).await;
        let _ = terminal_tx.send(terminal_value);
    });
    TypedLegacyExecutionJobHandleV1 {
        lease_token,
        initial_kind: JobKind::Discovery,
        cancel,
        snapshots,
        admission,
        terminal,
        worker,
    }
}

fn spawn_training_worker_v1(
    state: AppApiState,
    intent: TypedTrainingExecutionIntentV1,
    lease: ProcessExecutionLeaseV1,
) -> TypedLegacyExecutionJobHandleV1 {
    let lease_token = lease.token();
    let cancel = CancellationFlag::new();
    let initial = queued_snapshot_v1(JobKind::Training);
    let (snapshot_tx, snapshots) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
        lease_token,
        ProcessExecutionKindV1::Training,
        initial,
    ));
    let (terminal_tx, terminal) = oneshot::channel();
    let (admission_tx, admission) = oneshot::channel();
    let worker_cancel = cancel.clone();
    let worker = tokio::spawn(async move {
        let _lease = lease;
        let final_snapshot = match run_training_intent_v1(
            &state,
            intent,
            &worker_cancel,
            &snapshot_tx,
            lease_token,
            Some(admission_tx),
            None,
        )
        .await
        {
            Ok(snapshot) | Err(snapshot) => snapshot,
        };
        let terminal_value =
            terminal_from_snapshot_v1(final_snapshot, lease_token, JobKind::Training);
        persist_terminal_state_v1(&state, &terminal_value, JobKind::Training, lease_token).await;
        let _ = terminal_tx.send(terminal_value);
    });
    TypedLegacyExecutionJobHandleV1 {
        lease_token,
        initial_kind: JobKind::Training,
        cancel,
        snapshots,
        admission,
        terminal,
        worker,
    }
}

async fn continue_discovery_training_v1<F, Fut>(
    discovery_snapshot: JobSnapshot,
    cancel: &CancellationFlag,
    lease: &mut ProcessExecutionLeaseV1,
    snapshot_tx: &watch::Sender<TypedLegacyExecutionSnapshotV1>,
    run_training: F,
) -> Result<JobSnapshot, JobSnapshot>
where
    F: FnOnce(TypedTrainingExecutionIntentV1) -> Fut,
    Fut: std::future::Future<Output = Result<JobSnapshot, JobSnapshot>>,
{
    if discovery_snapshot.kind != JobKind::Discovery {
        return Err(failed_snapshot_v1(
            JobKind::Discovery,
            "automatic candidate Training requires a Discovery terminal snapshot",
        ));
    }
    if discovery_snapshot.state != JobState::Succeeded {
        return Err(discovery_snapshot);
    }
    if cancel.is_requested() {
        return Err(cancelled_snapshot_v1(JobKind::Discovery));
    }
    if let Some((_, published)) = discovery_snapshot
        .report
        .counters
        .iter()
        .find(|(key, _)| key == "working_set_training_handoffs")
        && *published > 1
    {
        let detail = format!(
            "Discovery published {published} batch handoffs. Automatic Training currently accepts one exact result; select a published result in Training. No result was selected automatically."
        );
        // The requested continuation failed, not the already completed research.
        // Keep its evidence and counters visible instead of replacing the report.
        let mut snapshot = discovery_snapshot;
        snapshot.state = JobState::Failed;
        snapshot.report.summary = detail;
        return Err(snapshot);
    }
    let mut handoffs = discovery_snapshot
        .report
        .highlights
        .iter()
        .filter(|(key, _)| key == "training_handoff");
    let Some((_, identity_sha256)) = handoffs.next() else {
        return Err(failed_snapshot_v1(
            JobKind::Discovery,
            "successful Discovery did not publish a training_handoff identity",
        ));
    };
    if handoffs.next().is_some() {
        return Err(failed_snapshot_v1(
            JobKind::Discovery,
            "successful Discovery published ambiguous training_handoff identities",
        ));
    }
    // Reuse the same canonical identity validation as the durable loader; this
    // only builds a path and performs no filesystem access.
    if let Err(error) = handoff::handoff_path(std::path::Path::new(""), identity_sha256) {
        return Err(failed_snapshot_v1(JobKind::Discovery, error));
    }
    let intent = TypedTrainingExecutionIntentV1 {
        selection: TypedTrainingSelectionPolicyV1::DiscoveryHandoff {
            identity_sha256: identity_sha256.clone(),
        },
    };
    if cancel.is_requested() {
        return Err(cancelled_snapshot_v1(JobKind::Discovery));
    }
    if let Err(error) = lease.transition_discovery_to_training_v1() {
        return Err(failed_snapshot_v1(JobKind::Discovery, error));
    }
    // Publish the new phase before invoking its worker, including preparation.
    // A panic now belongs to Training, not to the initial Discovery phase.
    snapshot_tx.send_replace(TypedLegacyExecutionSnapshotV1::new(
        lease.token(),
        ProcessExecutionKindV1::Training,
        queued_snapshot_v1(JobKind::Training),
    ));
    run_training(intent).await
}

async fn prepare_discovery_request_v1(
    state: &AppApiState,
    intent: &TypedDiscoveryExecutionIntentV1,
    cancel: &CancellationFlag,
) -> Result<(DiscoveryRequest, TypedLegacyExecutionAdmissionV1), TypedLegacyExecutionAdmissionErrorV1>
{
    if cancel.is_requested() {
        return Err(TypedLegacyExecutionAdmissionErrorV1::Cancelled(
            "typed Discovery was cancelled before Settings load".to_owned(),
        ));
    }
    let config_path = state.config_path().to_path_buf();
    let settings_source =
        tokio::task::spawn_blocking(move || DiscoverySettingsSource::load(&config_path))
            .await
            .map_err(|error| {
                TypedLegacyExecutionAdmissionErrorV1::Internal(bounded_detail_v1(error))
            })?
            .map_err(|error| {
                TypedLegacyExecutionAdmissionErrorV1::ServiceUnavailable(bounded_detail_v1(error))
            })?;
    let settings_source = Arc::new(settings_source);
    let settings = settings_source.settings();
    if intent.settings_gate == TypedDiscoverySettingsGateV1::RequireAutoRediscoveryEnabled
        && !settings.system.auto_rediscover_on_cull
    {
        return Err(TypedLegacyExecutionAdmissionErrorV1::ServiceUnavailable(
            "automatic rediscovery is disabled in current Settings".to_owned(),
        ));
    }
    if cancel.is_requested() {
        return Err(TypedLegacyExecutionAdmissionErrorV1::Cancelled(
            "typed Discovery was cancelled before dataset resolution".to_owned(),
        ));
    }
    let data_root = settings.system.data_dir.clone();
    let symbol = intent.symbol.trim().to_uppercase();
    let base_tf = intent.base_timeframe.as_str().to_owned();
    let higher = intent
        .higher_timeframes
        .resolve_for_discovery(settings, intent.base_timeframe)?;
    let pin_root = data_root.clone();
    let pin_higher = higher.clone();
    let pinned_input = match &intent.dataset_policy {
        TypedDiscoveryDatasetPolicyV1::Current => {
            let identity_root = data_root.clone();
            let identity_symbol = symbol.clone();
            let identity_tf = base_tf.clone();
            let identity = tokio::task::spawn_blocking(move || {
                resolve_unique_background_dataset_identity(
                    &identity_root,
                    &identity_symbol,
                    &identity_tf,
                )
            })
            .await
            .map_err(|error| {
                TypedLegacyExecutionAdmissionErrorV1::Internal(bounded_detail_v1(error))
            })?
            .map_err(|error| {
                TypedLegacyExecutionAdmissionErrorV1::BadRequest(bounded_detail_v1(error))
            })?;
            if cancel.is_requested() {
                return Err(TypedLegacyExecutionAdmissionErrorV1::Cancelled(
                    "typed Discovery was cancelled before exact dataset pin".to_owned(),
                ));
            }
            tokio::task::spawn_blocking(move || {
                pin_current_discovery_input(&pin_root, &identity, &pin_higher)
            })
            .await
            .map_err(|error| {
                TypedLegacyExecutionAdmissionErrorV1::Internal(bounded_detail_v1(error))
            })?
            .map_err(classify_discovery_pin_error_v1)?
        }
        TypedDiscoveryDatasetPolicyV1::Exact(selected) => {
            selected.validate().map_err(|error| {
                TypedLegacyExecutionAdmissionErrorV1::BadRequest(bounded_detail_v1(error))
            })?;
            if !selected
                .identity()
                .symbol_name()
                .eq_ignore_ascii_case(&symbol)
                || selected.identity().timeframe() != intent.base_timeframe
            {
                return Err(TypedLegacyExecutionAdmissionErrorV1::BadRequest(
                    "exact Discovery dataset selection disagrees with typed symbol/timeframe"
                        .to_owned(),
                ));
            }
            if cancel.is_requested() {
                return Err(TypedLegacyExecutionAdmissionErrorV1::Cancelled(
                    "typed Discovery was cancelled before exact dataset pin".to_owned(),
                ));
            }
            let selected = selected.clone();
            tokio::task::spawn_blocking(move || {
                pin_discovery_input(&pin_root, selected, &pin_higher)
            })
            .await
            .map_err(|error| {
                TypedLegacyExecutionAdmissionErrorV1::Internal(bounded_detail_v1(error))
            })?
            .map_err(classify_discovery_pin_error_v1)?
        }
    };
    // Admission owns intent and exact source selection only. The actual worker
    // resolves financial configuration after its real feature receipt exists.
    let selected_generation = pinned_input.receipt().anchor().clone();
    let request = DiscoveryRequest {
        data_root,
        settings_source,
        pinned_input: Arc::new(pinned_input),
        higher_tfs: higher,
        config: None,
        overrides: intent.overrides.clone(),
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    Ok((
        request,
        TypedLegacyExecutionAdmissionV1::Discovery {
            selected_generation,
        },
    ))
}

fn classify_discovery_pin_error_v1(error: anyhow::Error) -> TypedLegacyExecutionAdmissionErrorV1 {
    let detail = bounded_detail_v1(&error);
    if error
        .downcast_ref::<ExactDatasetGenerationConflict>()
        .is_some()
    {
        TypedLegacyExecutionAdmissionErrorV1::Conflict(detail)
    } else if error
        .downcast_ref::<DirectTimeframeAcquisitionRequired>()
        .is_some()
    {
        TypedLegacyExecutionAdmissionErrorV1::UnprocessableEntity(detail)
    } else {
        TypedLegacyExecutionAdmissionErrorV1::BadRequest(detail)
    }
}

async fn run_discovery_job_v1(
    state: &AppApiState,
    request: DiscoveryRequest,
    cancel: &CancellationFlag,
    snapshot_tx: &watch::Sender<TypedLegacyExecutionSnapshotV1>,
    lease_token: u64,
    admission_tx: TypedLegacyAdmissionSenderV1,
    admission: TypedLegacyExecutionAdmissionV1,
) -> Result<JobSnapshot, JobSnapshot> {
    if cancel.is_requested() {
        let _ = admission_tx.send(Err(TypedLegacyExecutionAdmissionErrorV1::Cancelled(
            "typed Discovery was cancelled before child start".to_owned(),
        )));
        return Err(cancelled_snapshot_v1(JobKind::Discovery));
    }
    let Some(execution) = state.execution_state() else {
        let error = TypedLegacyExecutionAdmissionErrorV1::ServiceUnavailable(
            "Discovery requires the installed application CPU execution budget".to_owned(),
        );
        let snapshot = failed_snapshot_v1(JobKind::Discovery, &error);
        let _ = admission_tx.send(Err(error));
        return Err(snapshot);
    };
    let (tx, mut rx) = mpsc::channel::<ServiceEvent>(1_000);
    let child = match start_discovery_job(request, execution, tx) {
        Ok(child) => child,
        Err(error) => {
            let error = TypedLegacyExecutionAdmissionErrorV1::BadRequest(bounded_detail_v1(error));
            let snapshot = failed_snapshot_v1(JobKind::Discovery, &error);
            let _ = admission_tx.send(Err(error));
            return Err(snapshot);
        }
    };
    let _ = admission_tx.send(Ok(admission));
    drain_job_events_v1(
        state,
        JobKind::Discovery,
        cancel,
        &child.cancel,
        &mut rx,
        snapshot_tx,
        lease_token,
        ProcessExecutionKindV1::Discovery,
    )
    .await
}

async fn run_training_intent_v1(
    state: &AppApiState,
    intent: TypedTrainingExecutionIntentV1,
    cancel: &CancellationFlag,
    snapshot_tx: &watch::Sender<TypedLegacyExecutionSnapshotV1>,
    lease_token: u64,
    admission_tx: Option<TypedLegacyAdmissionSenderV1>,
    expected_series: Option<CanonicalDatasetSeriesReceiptV1>,
) -> Result<JobSnapshot, JobSnapshot> {
    // Admission starts a new Training census even when cancellation precedes IO.
    state
        .install_engine(JobKind::Training, cancel.clone(), lease_token)
        .await;
    if cancel.is_requested() {
        if let Some(admission_tx) = admission_tx {
            let _ = admission_tx.send(Err(TypedLegacyExecutionAdmissionErrorV1::Cancelled(
                "typed Training was cancelled before Settings resolution".to_owned(),
            )));
        }
        return Err(cancelled_snapshot_v1(JobKind::Training));
    }
    let strategy_only = matches!(
        &intent.selection,
        TypedTrainingSelectionPolicyV1::StrategyResearchHandoff { .. }
    );
    let handoff_identity = match &intent.selection {
        TypedTrainingSelectionPolicyV1::DiscoveryHandoff { identity_sha256 }
        | TypedTrainingSelectionPolicyV1::StrategyResearchHandoff { identity_sha256 } => {
            Some(identity_sha256.clone())
        }
        _ => None,
    };
    let (request, admission) =
        match prepare_training_request_v1(state, intent, cancel, expected_series).await {
            Ok(prepared) => prepared,
            Err(error) => {
                if let Some(admission_tx) = admission_tx {
                    let _ = admission_tx.send(Err(error.clone()));
                }
                return Err(preparation_error_snapshot_v1(
                    JobKind::Training,
                    cancel,
                    error,
                ));
            }
        };
    let (tx, mut rx) = mpsc::channel::<ServiceEvent>(1_000);
    let start = match handoff_identity {
        Some(identity) if strategy_only => start_strategy_research_job(request, identity, tx),
        Some(identity) => start_discovery_training_job(request, identity, tx),
        None => start_training_job(request, tx),
    };
    let child = match start {
        Ok(child) => child,
        Err(error) => {
            let error = TypedLegacyExecutionAdmissionErrorV1::BadRequest(bounded_detail_v1(error));
            let snapshot = failed_snapshot_v1(JobKind::Training, &error);
            if let Some(admission_tx) = admission_tx {
                let _ = admission_tx.send(Err(error));
            }
            return Err(snapshot);
        }
    };
    if let Some(admission_tx) = admission_tx {
        let _ = admission_tx.send(Ok(admission));
    }
    drain_job_events_v1(
        state,
        JobKind::Training,
        cancel,
        &child.cancel,
        &mut rx,
        snapshot_tx,
        lease_token,
        ProcessExecutionKindV1::Training,
    )
    .await
}

async fn prepare_training_request_v1(
    state: &AppApiState,
    intent: TypedTrainingExecutionIntentV1,
    cancel: &CancellationFlag,
    expected_series: Option<CanonicalDatasetSeriesReceiptV1>,
) -> Result<(TrainingRequest, TypedLegacyExecutionAdmissionV1), TypedLegacyExecutionAdmissionErrorV1>
{
    let strategy_only = matches!(
        &intent.selection,
        TypedTrainingSelectionPolicyV1::StrategyResearchHandoff { .. }
    );
    let (symbol, base_timeframe) = match intent.selection {
        TypedTrainingSelectionPolicyV1::DiscoveryHandoff { identity_sha256 }
        | TypedTrainingSelectionPolicyV1::StrategyResearchHandoff { identity_sha256 } => {
            let config_path = state.config_path().to_path_buf();
            tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
                let settings = Settings::from_yaml(&config_path)?;
                let selected = handoff::load(&settings.system.data_dir, &identity_sha256)?;
                validate_follow_on_series_v1(
                    expected_series.as_ref(),
                    selected.canonical_series(),
                )?;
                let settings = handoff::settings_for_series(&settings, selected.canonical_series());
                if !strategy_only {
                    selected.validate_against_settings_v1(&settings)?;
                }
                Ok((
                    selected
                        .canonical_series()
                        .anchor()
                        .identity()
                        .symbol_name()
                        .to_owned(),
                    selected.base_timeframe(),
                ))
            })
            .await
            .map_err(|error| {
                TypedLegacyExecutionAdmissionErrorV1::Internal(bounded_detail_v1(error))
            })?
            .map_err(|error| {
                TypedLegacyExecutionAdmissionErrorV1::BadRequest(bounded_detail_v1(error))
            })?
        }
        TypedTrainingSelectionPolicyV1::Exact {
            symbol,
            base_timeframe,
        } => {
            if expected_series.is_some() {
                return Err(TypedLegacyExecutionAdmissionErrorV1::BadRequest(
                    "automatic candidate Training cannot use a symbol-only selection".to_owned(),
                ));
            }
            (symbol.trim().to_uppercase(), base_timeframe)
        }
    };
    if symbol.is_empty() {
        return Err(TypedLegacyExecutionAdmissionErrorV1::BadRequest(
            "Training symbol is empty".to_owned(),
        ));
    }
    if cancel.is_requested() {
        return Err(TypedLegacyExecutionAdmissionErrorV1::Cancelled(
            "typed Training was cancelled before child start".to_owned(),
        ));
    }
    let request = TrainingRequest {
        config_path: state.config_path().display().to_string(),
        models_dir: PathBuf::from("models"),
        symbol: symbol.clone(),
        base_tf: base_timeframe.as_str().to_owned(),
    };
    Ok((
        request,
        TypedLegacyExecutionAdmissionV1::Training {
            symbol,
            base_timeframe,
        },
    ))
}

fn validate_follow_on_series_v1(
    expected: Option<&CanonicalDatasetSeriesReceiptV1>,
    actual: &CanonicalDatasetSeriesReceiptV1,
) -> anyhow::Result<()> {
    if let Some(expected) = expected {
        anyhow::ensure!(
            actual == expected,
            "automatic candidate Training handoff differs from the completed Discovery canonical series"
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn drain_job_events_v1(
    state: &AppApiState,
    kind: JobKind,
    root_cancel: &CancellationFlag,
    child_cancel: &CancellationFlag,
    rx: &mut mpsc::Receiver<ServiceEvent>,
    snapshot_tx: &watch::Sender<TypedLegacyExecutionSnapshotV1>,
    lease_token: u64,
    lease_kind: ProcessExecutionKindV1,
) -> Result<JobSnapshot, JobSnapshot> {
    loop {
        if root_cancel.is_requested() {
            child_cancel.request();
        }
        let event = tokio::time::timeout(std::time::Duration::from_millis(25), rx.recv()).await;
        let Some(event) = (match event {
            Ok(event) => event,
            Err(_) => continue,
        }) else {
            break;
        };
        let snapshot = match (kind, event) {
            (JobKind::Discovery, ServiceEvent::DiscoveryUpdated(snapshot))
            | (JobKind::Training, ServiceEvent::TrainingUpdated(snapshot)) => snapshot,
            _ => continue,
        };
        state
            .update_engine_snapshot(kind, &snapshot, lease_token)
            .await;
        snapshot_tx.send_replace(TypedLegacyExecutionSnapshotV1::new(
            lease_token,
            lease_kind,
            snapshot.clone(),
        ));
        if !matches!(snapshot.state, JobState::Queued | JobState::Running) {
            return if snapshot.state == JobState::Succeeded {
                Ok(snapshot)
            } else {
                Err(snapshot)
            };
        }
    }
    let failure = failed_snapshot_v1(
        kind,
        format!("{kind:?} event channel closed without terminal evidence"),
    );
    state
        .update_engine_snapshot(kind, &failure, lease_token)
        .await;
    Err(failure)
}

pub(crate) fn detach_typed_legacy_execution_observer_v1(
    state: AppApiState,
    mut handle: TypedLegacyExecutionJobHandleV1,
) {
    tokio::spawn(async move {
        let initial_kind = handle.initial_kind;
        let lease_token = handle.lease_token;
        loop {
            if handle.snapshot_receiver_mut().changed().await.is_err() {
                break;
            }
            if !matches!(
                handle.snapshot_receiver_mut().borrow().state(),
                JobState::Queued | JobState::Running
            ) {
                break;
            }
        }
        let terminal = handle.await_terminal().await;
        let (kind, snapshot) = terminal_kind_and_snapshot_v1(&terminal, initial_kind);
        if let Some(snapshot) = snapshot {
            state
                .update_engine_snapshot(kind, snapshot, lease_token)
                .await;
        } else if let TypedLegacyExecutionTerminalV1::WorkerPanicked { detail, .. } = terminal {
            state
                .update_engine_snapshot(kind, &failed_snapshot_v1(kind, detail), lease_token)
                .await;
        }
    });
}

fn queued_snapshot_v1(kind: JobKind) -> JobSnapshot {
    JobSnapshot::new(kind)
}

fn failed_snapshot_v1(kind: JobKind, detail: impl fmt::Display) -> JobSnapshot {
    let mut snapshot = JobSnapshot::new(kind);
    snapshot.state = JobState::Failed;
    snapshot.report.summary = bounded_detail_v1(detail);
    snapshot
}

fn cancelled_snapshot_v1(kind: JobKind) -> JobSnapshot {
    let mut snapshot = JobSnapshot::new(kind);
    snapshot.state = JobState::Cancelled;
    snapshot.report.summary = "typed engine execution cancelled".to_owned();
    snapshot
}

fn preparation_error_snapshot_v1(
    kind: JobKind,
    cancel: &CancellationFlag,
    detail: impl fmt::Display,
) -> JobSnapshot {
    if cancel.is_requested() {
        cancelled_snapshot_v1(kind)
    } else {
        failed_snapshot_v1(kind, detail)
    }
}

fn terminal_from_snapshot_v1(
    snapshot: JobSnapshot,
    lease_token: u64,
    kind: JobKind,
) -> TypedLegacyExecutionTerminalV1 {
    match snapshot.state {
        JobState::Succeeded => TypedLegacyExecutionTerminalV1::Succeeded {
            final_snapshot: snapshot,
            lease_token,
            completed_kind: kind,
        },
        JobState::Cancelled => TypedLegacyExecutionTerminalV1::Cancelled {
            final_snapshot: snapshot,
            lease_token,
        },
        _ => TypedLegacyExecutionTerminalV1::Failed {
            detail: snapshot.report.summary.clone(),
            final_snapshot: snapshot,
            lease_token,
        },
    }
}

fn terminal_kind_and_snapshot_v1(
    terminal: &TypedLegacyExecutionTerminalV1,
    _fallback_kind: JobKind,
) -> (JobKind, Option<&JobSnapshot>) {
    match terminal {
        TypedLegacyExecutionTerminalV1::Succeeded {
            final_snapshot,
            completed_kind,
            ..
        } => (*completed_kind, Some(final_snapshot)),
        TypedLegacyExecutionTerminalV1::Failed { final_snapshot, .. }
        | TypedLegacyExecutionTerminalV1::Cancelled { final_snapshot, .. } => {
            (final_snapshot.kind, Some(final_snapshot))
        }
        TypedLegacyExecutionTerminalV1::WorkerPanicked { job_kind, .. } => (*job_kind, None),
    }
}

async fn persist_terminal_state_v1(
    state: &AppApiState,
    terminal: &TypedLegacyExecutionTerminalV1,
    fallback_kind: JobKind,
    lease_token: u64,
) {
    let (kind, snapshot) = terminal_kind_and_snapshot_v1(terminal, fallback_kind);
    if let Some(snapshot) = snapshot {
        state
            .update_engine_snapshot(kind, snapshot, lease_token)
            .await;
    }
}

fn bounded_detail_v1(detail: impl ToString) -> String {
    let mut detail = detail.to_string();
    if detail.len() <= MAX_TYPED_EXECUTION_DETAIL_BYTES_V1 {
        return detail;
    }
    let mut end = MAX_TYPED_EXECUTION_DETAIL_BYTES_V1;
    while !detail.is_char_boundary(end) {
        end -= 1;
    }
    detail.truncate(end);
    detail
}

#[cfg(test)]
#[path = "typed_execution_v1_tests.rs"]
mod tests;
