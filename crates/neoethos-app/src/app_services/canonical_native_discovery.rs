//! Dedicated application lane for canonical native Generation-zero research.
//!
//! This module deliberately does not reuse the legacy Discovery job/result
//! vocabulary: native research publishes a sealed research artifact and never
//! produces model targets or starts Training.

use std::fmt;
use std::sync::Arc;

use neoethos_core::Settings;
use neoethos_search::{
    CanonicalNativeCancellationTokenV1, CanonicalNativeDiscoveryExecutionErrorCodeV1,
    CanonicalNativeDiscoveryExecutionStageV1, CanonicalNativeGenerationZeroOverridesV1,
    CanonicalNativeRuntimeInstallReceiptV1, CanonicalResearchContractArtifactRefV1,
    DiscoveryProgress, ProcessExecutionBusyV1, ProcessExecutionKindV1, ProcessExecutionLeaseV1,
    PublishedCanonicalNativeGenerationZeroResearchV1,
    run_canonical_native_discovery_generation_zero_from_ref_v1,
    try_acquire_process_execution_lease_v1,
};
use tokio::sync::{oneshot, watch};
use tokio::task::JoinHandle;

const MAX_NATIVE_RESEARCH_DETAIL_BYTES_V1: usize = 1_024;

#[derive(Debug)]
pub struct CanonicalNativeResearchIntentV1 {
    contract_ref: CanonicalResearchContractArtifactRefV1,
    overrides: CanonicalNativeGenerationZeroOverridesV1,
}

impl CanonicalNativeResearchIntentV1 {
    pub fn new(
        contract_ref: CanonicalResearchContractArtifactRefV1,
        overrides: CanonicalNativeGenerationZeroOverridesV1,
    ) -> Self {
        Self {
            contract_ref,
            overrides,
        }
    }

    pub fn contract_ref(&self) -> &CanonicalResearchContractArtifactRefV1 {
        &self.contract_ref
    }

    pub fn overrides(&self) -> &CanonicalNativeGenerationZeroOverridesV1 {
        &self.overrides
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalNativeResearchStateV1 {
    Queued,
    Running,
    Published,
    Failed,
    Cancelled,
    WorkerPanicked,
}

impl CanonicalNativeResearchStateV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "Queued",
            Self::Running => "Running",
            Self::Published => "Published",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
            Self::WorkerPanicked => "WorkerPanicked",
        }
    }

    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Published | Self::Failed | Self::Cancelled | Self::WorkerPanicked
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalNativeResearchFailureV1 {
    stage: Option<CanonicalNativeDiscoveryExecutionStageV1>,
    code: Option<CanonicalNativeDiscoveryExecutionErrorCodeV1>,
    detail: String,
}

impl CanonicalNativeResearchFailureV1 {
    fn from_execution(error: &neoethos_search::CanonicalNativeDiscoveryExecutionErrorV1) -> Self {
        Self {
            stage: Some(error.stage()),
            code: Some(error.code()),
            detail: bounded_detail_v1(error.detail()),
        }
    }

    fn worker_panicked(detail: impl fmt::Display) -> Self {
        Self {
            stage: None,
            code: None,
            detail: bounded_detail_v1(detail.to_string()),
        }
    }

    pub const fn stage(&self) -> Option<CanonicalNativeDiscoveryExecutionStageV1> {
        self.stage
    }

    pub const fn code(&self) -> Option<CanonicalNativeDiscoveryExecutionErrorCodeV1> {
        self.code
    }

    pub fn stable_stage(&self) -> &'static str {
        self.stage.map(stage_name_v1).unwrap_or("native_worker")
    }

    pub fn stable_code(&self) -> &'static str {
        self.code.map(code_name_v1).unwrap_or("worker_panicked")
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalNativeResearchTerminalSnapshotV1 {
    Published(PublishedCanonicalNativeGenerationZeroResearchV1),
    Failed(CanonicalNativeResearchFailureV1),
    Cancelled(CanonicalNativeResearchFailureV1),
    WorkerPanicked(CanonicalNativeResearchFailureV1),
}

impl CanonicalNativeResearchTerminalSnapshotV1 {
    pub const fn state(&self) -> CanonicalNativeResearchStateV1 {
        match self {
            Self::Published(_) => CanonicalNativeResearchStateV1::Published,
            Self::Failed(_) => CanonicalNativeResearchStateV1::Failed,
            Self::Cancelled(_) => CanonicalNativeResearchStateV1::Cancelled,
            Self::WorkerPanicked(_) => CanonicalNativeResearchStateV1::WorkerPanicked,
        }
    }

    pub fn published(&self) -> Option<&PublishedCanonicalNativeGenerationZeroResearchV1> {
        match self {
            Self::Published(published) => Some(published),
            _ => None,
        }
    }

    pub fn failure(&self) -> Option<&CanonicalNativeResearchFailureV1> {
        match self {
            Self::Failed(failure) | Self::Cancelled(failure) | Self::WorkerPanicked(failure) => {
                Some(failure)
            }
            Self::Published(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalNativeResearchSnapshotV1 {
    lease_token: u64,
    state: CanonicalNativeResearchStateV1,
    stage: String,
    percent_basis_points: u16,
    terminal: Option<CanonicalNativeResearchTerminalSnapshotV1>,
}

impl CanonicalNativeResearchSnapshotV1 {
    fn queued(lease_token: u64) -> Self {
        Self {
            lease_token,
            state: CanonicalNativeResearchStateV1::Queued,
            stage: "queued".to_owned(),
            percent_basis_points: 0,
            terminal: None,
        }
    }

    fn running(lease_token: u64, stage: impl Into<String>, percent_basis_points: u16) -> Self {
        Self {
            lease_token,
            state: CanonicalNativeResearchStateV1::Running,
            stage: stage.into(),
            percent_basis_points: percent_basis_points.min(10_000),
            terminal: None,
        }
    }

    pub(crate) fn from_terminal(
        lease_token: u64,
        terminal: CanonicalNativeResearchTerminalSnapshotV1,
    ) -> Self {
        Self {
            lease_token,
            state: terminal.state(),
            stage: "terminal".to_owned(),
            percent_basis_points: 10_000,
            terminal: Some(terminal),
        }
    }

    pub const fn lease_token(&self) -> u64 {
        self.lease_token
    }

    pub const fn state(&self) -> CanonicalNativeResearchStateV1 {
        self.state
    }

    pub fn stage(&self) -> &str {
        &self.stage
    }

    pub const fn percent_basis_points(&self) -> u16 {
        self.percent_basis_points
    }

    pub fn terminal(&self) -> Option<&CanonicalNativeResearchTerminalSnapshotV1> {
        self.terminal.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalNativeResearchEventV1 {
    snapshot: CanonicalNativeResearchSnapshotV1,
}

impl CanonicalNativeResearchEventV1 {
    #[allow(
        dead_code,
        reason = "constructed by the Chunk4C-observed native state reducer"
    )]
    pub(crate) fn new(snapshot: CanonicalNativeResearchSnapshotV1) -> Self {
        Self { snapshot }
    }

    pub fn snapshot(&self) -> &CanonicalNativeResearchSnapshotV1 {
        &self.snapshot
    }
}

#[derive(Debug)]
pub enum CanonicalNativeResearchStartErrorV1 {
    Busy(ProcessExecutionBusyV1),
    RuntimeUnavailable(String),
}

impl fmt::Display for CanonicalNativeResearchStartErrorV1 {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy(error) => error.fmt(output),
            Self::RuntimeUnavailable(detail) => {
                write!(output, "native research runtime is unavailable: {detail}")
            }
        }
    }
}

impl std::error::Error for CanonicalNativeResearchStartErrorV1 {}

pub struct CanonicalNativeResearchJobHandleV1 {
    cancellation: CanonicalNativeCancellationTokenV1,
    snapshots: watch::Receiver<CanonicalNativeResearchSnapshotV1>,
    terminal: oneshot::Receiver<CanonicalNativeResearchTerminalSnapshotV1>,
    worker: JoinHandle<()>,
}

impl CanonicalNativeResearchJobHandleV1 {
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn cancellation_token(&self) -> &CanonicalNativeCancellationTokenV1 {
        &self.cancellation
    }

    pub fn snapshot_receiver_mut(
        &mut self,
    ) -> &mut watch::Receiver<CanonicalNativeResearchSnapshotV1> {
        &mut self.snapshots
    }

    pub async fn await_terminal(self) -> CanonicalNativeResearchTerminalSnapshotV1 {
        let Self {
            cancellation: _,
            snapshots: _,
            terminal,
            worker,
        } = self;
        let signalled = terminal.await.ok();
        match worker.await {
            Ok(()) => signalled.unwrap_or_else(|| {
                CanonicalNativeResearchTerminalSnapshotV1::WorkerPanicked(
                    CanonicalNativeResearchFailureV1::worker_panicked(
                        "native research worker exited without terminal evidence",
                    ),
                )
            }),
            Err(error) => CanonicalNativeResearchTerminalSnapshotV1::WorkerPanicked(
                CanonicalNativeResearchFailureV1::worker_panicked(error),
            ),
        }
    }
}

pub fn start_canonical_native_research_lane_v1(
    startup_settings: Arc<Settings>,
    runtime_install_receipt: Arc<CanonicalNativeRuntimeInstallReceiptV1>,
    intent: CanonicalNativeResearchIntentV1,
) -> Result<CanonicalNativeResearchJobHandleV1, CanonicalNativeResearchStartErrorV1> {
    tokio::runtime::Handle::try_current().map_err(|error| {
        CanonicalNativeResearchStartErrorV1::RuntimeUnavailable(bounded_detail_v1(error))
    })?;
    let lease = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::NativeResearch)
        .map_err(CanonicalNativeResearchStartErrorV1::Busy)?;
    Ok(spawn_native_worker_v1(
        startup_settings,
        runtime_install_receipt,
        intent,
        lease,
    ))
}

fn spawn_native_worker_v1(
    startup_settings: Arc<Settings>,
    runtime_install_receipt: Arc<CanonicalNativeRuntimeInstallReceiptV1>,
    intent: CanonicalNativeResearchIntentV1,
    lease: ProcessExecutionLeaseV1,
) -> CanonicalNativeResearchJobHandleV1 {
    let lease_token = lease.token();
    let cancellation = CanonicalNativeCancellationTokenV1::new();
    let worker_cancellation = cancellation.clone();
    let (snapshot_tx, snapshots) =
        watch::channel(CanonicalNativeResearchSnapshotV1::queued(lease_token));
    let (terminal_tx, terminal) = oneshot::channel();
    let worker = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        snapshot_tx.send_replace(CanonicalNativeResearchSnapshotV1::running(
            lease_token,
            "preparing_native_research",
            100,
        ));
        let CanonicalNativeResearchIntentV1 {
            contract_ref,
            overrides,
        } = intent;
        let progress_tx = snapshot_tx.clone();
        let result = run_canonical_native_discovery_generation_zero_from_ref_v1(
            startup_settings.as_ref(),
            runtime_install_receipt.as_ref(),
            contract_ref,
            overrides,
            &worker_cancellation,
            move |progress| {
                let (stage, percent) = progress_snapshot_v1(&progress);
                progress_tx.send_replace(CanonicalNativeResearchSnapshotV1::running(
                    lease_token,
                    stage,
                    percent,
                ));
            },
        );
        let terminal_snapshot = match result {
            Ok(published) => CanonicalNativeResearchTerminalSnapshotV1::Published(published),
            Err(error) => {
                let cancelled =
                    error.code() == CanonicalNativeDiscoveryExecutionErrorCodeV1::Cancelled;
                let failure = CanonicalNativeResearchFailureV1::from_execution(&error);
                if cancelled {
                    CanonicalNativeResearchTerminalSnapshotV1::Cancelled(failure)
                } else {
                    CanonicalNativeResearchTerminalSnapshotV1::Failed(failure)
                }
            }
        };
        snapshot_tx.send_replace(CanonicalNativeResearchSnapshotV1::from_terminal(
            lease_token,
            terminal_snapshot.clone(),
        ));
        let _ = terminal_tx.send(terminal_snapshot);
    });
    CanonicalNativeResearchJobHandleV1 {
        cancellation,
        snapshots,
        terminal,
        worker,
    }
}

fn progress_snapshot_v1(progress: &DiscoveryProgress) -> (&'static str, u16) {
    match progress {
        DiscoveryProgress::SearchStarted { .. } => ("generation_zero_started", 1_000),
        DiscoveryProgress::GenerationCompleted {
            generation,
            total_generations,
            ..
        } => {
            let denominator = (*total_generations).max(1);
            let percent = generation.saturating_mul(8_000) / denominator;
            (
                "generation_completed",
                u16::try_from(percent.min(8_000)).unwrap_or(8_000),
            )
        }
        DiscoveryProgress::CandidatesRanked { .. } => ("candidates_ranked", 8_300),
        DiscoveryProgress::CandidatesFiltered { .. } => ("candidates_filtered", 8_600),
        DiscoveryProgress::QualityScreened { .. } => ("quality_screened", 8_900),
        DiscoveryProgress::PortfolioSelected { .. } => ("portfolio_selected", 9_200),
        DiscoveryProgress::StageAdvanced { stage, .. } => (stage, 9_400),
        DiscoveryProgress::Completed { .. } => ("generation_zero_completed", 9_700),
    }
}

fn bounded_detail_v1(detail: impl ToString) -> String {
    let mut detail = detail.to_string();
    if detail.len() <= MAX_NATIVE_RESEARCH_DETAIL_BYTES_V1 {
        return detail;
    }
    let mut end = MAX_NATIVE_RESEARCH_DETAIL_BYTES_V1;
    while !detail.is_char_boundary(end) {
        end -= 1;
    }
    detail.truncate(end);
    detail
}

fn stage_name_v1(stage: CanonicalNativeDiscoveryExecutionStageV1) -> &'static str {
    use CanonicalNativeDiscoveryExecutionStageV1 as Stage;
    match stage {
        Stage::NativeCapabilityGate => "native_capability_gate",
        Stage::RuntimeInstallReceipt => "runtime_install_receipt",
        Stage::SearchGpuExecutionLease => "search_gpu_execution_lease",
        Stage::ContractReferenceValidation => "contract_reference_validation",
        Stage::ContractArtifactRead => "contract_artifact_read",
        Stage::ContractArtifactHash => "contract_artifact_hash",
        Stage::ContractSchemaValidation => "contract_schema_validation",
        Stage::ExactSourcePin => "exact_source_pin",
        Stage::NativePreflight => "native_preflight",
        Stage::NativeAdmission => "native_admission",
        Stage::ResidentDataMaterialization => "resident_data_materialization",
        Stage::NativeReceiptBinding => "native_receipt_binding",
        Stage::GenerationZeroEvaluation => "generation_zero_evaluation",
        Stage::ConsumerCompletion => "consumer_completion",
        Stage::ResultPublication => "result_publication",
    }
}

fn code_name_v1(code: CanonicalNativeDiscoveryExecutionErrorCodeV1) -> &'static str {
    use CanonicalNativeDiscoveryExecutionErrorCodeV1 as Code;
    match code {
        Code::UnsupportedPlatform => "unsupported_platform",
        Code::NativeCudaRequired => "native_cuda_required",
        Code::Cancelled => "cancelled",
        Code::InvalidRequest => "invalid_request",
        Code::RuntimeAuthorityInvalid => "runtime_authority_invalid",
        Code::ArtifactUnavailable => "artifact_unavailable",
        Code::ArtifactHashMismatch => "artifact_hash_mismatch",
        Code::ContractInvalid => "contract_invalid",
        Code::ExactGenerationConflict => "exact_generation_conflict",
        Code::PreflightRejected => "preflight_rejected",
        Code::AdmissionRejected => "admission_rejected",
        Code::MaterializationRejected => "materialization_rejected",
        Code::ReceiptRejected => "receipt_rejected",
        Code::EvaluationRejected => "evaluation_rejected",
        Code::CompletionRejected => "completion_rejected",
        Code::ResultSealingRejected => "result_sealing_rejected",
        Code::PublicationRejected => "publication_rejected",
    }
}

#[cfg(test)]
#[path = "canonical_native_discovery_tests.rs"]
mod tests;
