//! Lease-owning typed adapters for the legacy Discovery/Training workers.
//!
//! These are intentionally crate-private. Frontends pass lightweight intent;
//! Settings and canonical dataset resolution happen only after admission.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use neoethos_core::Settings;
use neoethos_data::{
    CanonicalTimeframe, ExactDatasetGenerationConflict, SelectedDatasetGenerationV1,
};
use neoethos_search::{
    DiscoveryConfig, ProcessExecutionBusyV1, ProcessExecutionKindV1, ProcessExecutionLeaseV1,
    PropFirmRiskRules, try_acquire_process_execution_lease_v1,
};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use crate::app_services::ServiceEvent;
use crate::app_services::discovery::{
    DirectTimeframeAcquisitionRequired, DiscoveryRequest, pin_current_discovery_input,
    pin_discovery_input, resolve_unique_background_dataset_identity, start_discovery_job,
};
use crate::app_services::jobs::{CancellationFlag, JobKind, JobReport, JobSnapshot, JobState};
use crate::app_services::training::{TrainingRequest, start_training_job};
use crate::server::state::AppApiState;

use super::EngineRunState;

const MAX_TYPED_EXECUTION_DETAIL_BYTES_V1: usize = 1_024;

type TypedLegacyAdmissionSenderV1 =
    oneshot::Sender<Result<TypedLegacyExecutionAdmissionV1, TypedLegacyExecutionAdmissionErrorV1>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TypedDiscoveryGenerationOverrideV1 {
    Exact(usize),
    Floor(usize),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TypedDiscoveryOverridesV1 {
    population: Option<usize>,
    generation_policy: Option<TypedDiscoveryGenerationOverrideV1>,
    max_indicators: Option<usize>,
    max_rows: Option<usize>,
    target_candidates: Option<usize>,
    portfolio_size: Option<usize>,
}

impl TypedDiscoveryOverridesV1 {
    pub(crate) fn checked_new(
        population: Option<usize>,
        generation_policy: Option<TypedDiscoveryGenerationOverrideV1>,
        max_indicators: Option<usize>,
        max_rows: Option<usize>,
        target_candidates: Option<usize>,
        portfolio_size: Option<usize>,
    ) -> Result<Self, &'static str> {
        let values = [
            population,
            max_indicators,
            max_rows,
            target_candidates,
            portfolio_size,
        ];
        if values.into_iter().flatten().any(|value| value == 0) {
            return Err("typed Discovery overrides must be nonzero when supplied");
        }
        if generation_policy.is_some_and(|policy| match policy {
            TypedDiscoveryGenerationOverrideV1::Exact(value)
            | TypedDiscoveryGenerationOverrideV1::Floor(value) => value == 0,
        }) {
            return Err("typed Discovery generation override must be nonzero");
        }
        Ok(Self {
            population,
            generation_policy,
            max_indicators,
            max_rows,
            target_candidates,
            portfolio_size,
        })
    }

    pub(crate) fn exact_generations(generations: usize) -> Result<Self, &'static str> {
        Self::checked_new(
            None,
            Some(TypedDiscoveryGenerationOverrideV1::Exact(generations)),
            None,
            None,
            None,
            None,
        )
    }

    pub(crate) fn minimum_generations(generations: usize) -> Result<Self, &'static str> {
        Self::checked_new(
            None,
            Some(TypedDiscoveryGenerationOverrideV1::Floor(generations)),
            None,
            None,
            None,
            None,
        )
    }

    fn apply(&self, config: &mut DiscoveryConfig) {
        if let Some(population) = self.population {
            config.population = population;
        }
        if let Some(policy) = self.generation_policy {
            config.generations = match policy {
                TypedDiscoveryGenerationOverrideV1::Exact(value) => value,
                TypedDiscoveryGenerationOverrideV1::Floor(value) => config.generations.max(value),
            };
        }
        if let Some(max_indicators) = self.max_indicators {
            config.max_indicators = max_indicators;
        }
        if let Some(max_rows) = self.max_rows {
            config.max_rows = max_rows;
        }
        if let Some(target_candidates) = self.target_candidates {
            config.candidate_count = target_candidates;
        }
        if let Some(portfolio_size) = self.portfolio_size {
            config.portfolio_size = portfolio_size;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TypedHigherTimeframePolicyV1 {
    Configured,
    Exact(Vec<CanonicalTimeframe>),
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
    Configured,
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

    pub(crate) const fn lease_token(&self) -> u64 {
        self.lease_token
    }

    pub(crate) const fn lease_kind(&self) -> ProcessExecutionKindV1 {
        self.lease_kind
    }

    pub(crate) fn job_snapshot(&self) -> &JobSnapshot {
        &self.job_snapshot
    }

    pub(crate) const fn state(&self) -> JobState {
        self.job_snapshot.state
    }

    pub(crate) fn report(&self) -> &JobReport {
        &self.job_snapshot.report
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
            snapshots: _,
            admission: _,
            terminal,
            worker,
        } = self;
        let signalled = terminal.await.ok();
        match worker.await {
            Ok(()) => signalled.unwrap_or(TypedLegacyExecutionTerminalV1::WorkerPanicked {
                lease_token,
                detail: "typed engine worker exited without terminal evidence".to_owned(),
            }),
            Err(error) => TypedLegacyExecutionTerminalV1::WorkerPanicked {
                lease_token,
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
    mut lease: ProcessExecutionLeaseV1,
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
        state
            .install_engine(JobKind::Discovery, worker_cancel.clone())
            .await;
        let discovery = prepare_discovery_request_v1(&state, &intent, &worker_cancel).await;
        let final_result = match discovery {
            Ok((request, receipt)) => {
                run_discovery_job_v1(
                    &state,
                    request,
                    &worker_cancel,
                    &snapshot_tx,
                    lease_token,
                    admission_tx,
                    receipt,
                )
                .await
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
        let terminal_value = match final_result {
            Ok(discovery_final)
                if discovery_final.state == JobState::Succeeded
                    && intent.training_after_success
                    && !worker_cancel.is_requested() =>
            {
                match lease.transition_discovery_to_training_v1() {
                    Ok(()) => {
                        let training_intent = TypedTrainingExecutionIntentV1 {
                            selection: TypedTrainingSelectionPolicyV1::Exact {
                                symbol: intent.symbol,
                                base_timeframe: intent.base_timeframe,
                            },
                        };
                        match run_training_intent_v1(
                            &state,
                            training_intent,
                            &worker_cancel,
                            &snapshot_tx,
                            lease_token,
                            None,
                        )
                        .await
                        {
                            Ok(training_final) => terminal_from_snapshot_v1(
                                training_final,
                                lease_token,
                                JobKind::Training,
                            ),
                            Err(failed) => {
                                terminal_from_snapshot_v1(failed, lease_token, JobKind::Training)
                            }
                        }
                    }
                    Err(error) => TypedLegacyExecutionTerminalV1::Failed {
                        final_snapshot: failed_snapshot_v1(JobKind::Discovery, &error),
                        lease_token,
                        detail: bounded_detail_v1(error),
                    },
                }
            }
            Ok(final_snapshot) => {
                terminal_from_snapshot_v1(final_snapshot, lease_token, JobKind::Discovery)
            }
            Err(final_snapshot) => {
                terminal_from_snapshot_v1(final_snapshot, lease_token, JobKind::Discovery)
            }
        };
        persist_terminal_state_v1(&state, &terminal_value, JobKind::Discovery).await;
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
        )
        .await
        {
            Ok(snapshot) | Err(snapshot) => snapshot,
        };
        let terminal_value =
            terminal_from_snapshot_v1(final_snapshot, lease_token, JobKind::Training);
        persist_terminal_state_v1(&state, &terminal_value, JobKind::Training).await;
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
    let settings = tokio::task::spawn_blocking(move || Settings::from_yaml(&config_path))
        .await
        .map_err(|error| TypedLegacyExecutionAdmissionErrorV1::Internal(bounded_detail_v1(error)))?
        .map_err(|error| {
            TypedLegacyExecutionAdmissionErrorV1::ServiceUnavailable(bounded_detail_v1(error))
        })?;
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
    let higher = match &intent.higher_timeframes {
        TypedHigherTimeframePolicyV1::Configured => {
            settings.system.resolve_higher_timeframes(&base_tf)
        }
        TypedHigherTimeframePolicyV1::Exact(timeframes) => timeframes
            .iter()
            .map(|timeframe| timeframe.as_str().to_owned())
            .collect(),
    };
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
    let mut config = DiscoveryConfig::try_from_settings(&settings).map_err(|error| {
        TypedLegacyExecutionAdmissionErrorV1::ServiceUnavailable(bounded_detail_v1(error))
    })?;
    config.evaluation_symbol = symbol;
    intent.overrides.apply(&mut config);
    config = config.apply_mode_overrides();
    let selected_generation = pinned_input.receipt().anchor().clone();
    let request = DiscoveryRequest {
        data_root,
        pinned_input: Arc::new(pinned_input),
        higher_tfs: higher,
        config,
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
    let (tx, mut rx) = mpsc::channel::<ServiceEvent>(1_000);
    let child = match start_discovery_job(request, tx) {
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
) -> Result<JobSnapshot, JobSnapshot> {
    if cancel.is_requested() {
        if let Some(admission_tx) = admission_tx {
            let _ = admission_tx.send(Err(TypedLegacyExecutionAdmissionErrorV1::Cancelled(
                "typed Training was cancelled before Settings resolution".to_owned(),
            )));
        }
        return Err(cancelled_snapshot_v1(JobKind::Training));
    }
    state
        .install_engine(JobKind::Training, cancel.clone())
        .await;
    let (request, admission) = match prepare_training_request_v1(state, intent, cancel).await {
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
    let child = match start_training_job(request, tx) {
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
) -> Result<(TrainingRequest, TypedLegacyExecutionAdmissionV1), TypedLegacyExecutionAdmissionErrorV1>
{
    let (symbol, base_timeframe) = match intent.selection {
        TypedTrainingSelectionPolicyV1::Configured => {
            let config_path = state.config_path().to_path_buf();
            let settings = tokio::task::spawn_blocking(move || Settings::from_yaml(&config_path))
                .await
                .map_err(|error| {
                    TypedLegacyExecutionAdmissionErrorV1::Internal(bounded_detail_v1(error))
                })?
                .map_err(|error| {
                    TypedLegacyExecutionAdmissionErrorV1::BadRequest(bounded_detail_v1(error))
                })?;
            let symbol = settings.system.resolve_symbol().trim().to_uppercase();
            let base_timeframe = settings
                .system
                .resolve_base_timeframe()
                .parse::<CanonicalTimeframe>()
                .map_err(|error| {
                    TypedLegacyExecutionAdmissionErrorV1::BadRequest(bounded_detail_v1(error))
                })?;
            (symbol, base_timeframe)
        }
        TypedTrainingSelectionPolicyV1::Exact {
            symbol,
            base_timeframe,
        } => (symbol.trim().to_uppercase(), base_timeframe),
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
        let run_state = EngineRunState::from(snapshot.state);
        state
            .update_engine(kind, run_state, snapshot.report.summary.clone())
            .await;
        if run_state == EngineRunState::Running {
            state
                .set_engine_progress(
                    kind,
                    snapshot.progress.stage.clone(),
                    f64::from(snapshot.progress.percent.unwrap_or(0.0)),
                    snapshot.report.counters.clone(),
                )
                .await;
        }
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
    state.finalize_engine_if_running(kind).await;
    Err(failed_snapshot_v1(
        kind,
        format!("{kind:?} event channel closed without terminal evidence"),
    ))
}

pub(crate) fn detach_typed_legacy_execution_observer_v1(
    state: AppApiState,
    mut handle: TypedLegacyExecutionJobHandleV1,
) {
    tokio::spawn(async move {
        let initial_kind = handle.initial_kind;
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
                .update_engine(
                    kind,
                    EngineRunState::from(snapshot.state),
                    snapshot.report.summary.clone(),
                )
                .await;
        } else if let TypedLegacyExecutionTerminalV1::WorkerPanicked { detail, .. } = terminal {
            state
                .update_engine(kind, EngineRunState::Failed, detail)
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
    fallback_kind: JobKind,
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
        TypedLegacyExecutionTerminalV1::WorkerPanicked { .. } => (fallback_kind, None),
    }
}

async fn persist_terminal_state_v1(
    state: &AppApiState,
    terminal: &TypedLegacyExecutionTerminalV1,
    fallback_kind: JobKind,
) {
    let (kind, snapshot) = terminal_kind_and_snapshot_v1(terminal, fallback_kind);
    if let Some(snapshot) = snapshot {
        state
            .update_engine(
                kind,
                EngineRunState::from(snapshot.state),
                snapshot.report.summary.clone(),
            )
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
