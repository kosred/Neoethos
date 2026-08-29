use std::fmt;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::canonical_native_discovery_request_v1::{
    CanonicalNativeGenerationZeroOverridesV1, CanonicalResearchContractArtifactRefV1,
};
use crate::canonical_native_runtime_authority_v1::CanonicalNativeRuntimeInstallReceiptV1;
use crate::discovery::DiscoveryProgress;

#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
use crate::canonical_native_discovery_request_v1::{
    CanonicalNativeDiscoveryRequestErrorV1, resolve_canonical_native_discovery_request_v1,
};
#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
use crate::canonical_native_generation_zero_publication_v1::{
    CanonicalNativeGenerationZeroPublicationErrorKindV1,
    CanonicalNativeGenerationZeroPublicationGateRejectionV1,
    publish_canonical_native_generation_zero_research_result_v1,
};
#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
use crate::canonical_native_generation_zero_result_v1::{
    preflight_canonical_native_generation_zero_result_v1,
    seal_canonical_native_generation_zero_research_result_v1,
};
#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
use crate::data_selection::CanonicalGpuResidentSearchInputReceiptV3;
#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
use crate::prepared_discovery_run_input_v3::{
    ResidentGenerationZeroStageErrorV1,
    prepare_prepared_canonical_trendbar_research_run_input_capped_v5,
    run_prepared_canonical_trendbar_research_generation_zero_gated_typed_v5,
};
#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
use crate::resident_population_auto_sizing_receipt_v2::canonical_pinned_source_projection_from_search_receipt_v1;
#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
use anyhow::Context;

const MAX_EXECUTION_ERROR_DETAIL_BYTES_V1: usize = 1_024;

#[derive(Clone, Debug, Default)]
pub struct CanonicalNativeCancellationTokenV1 {
    cancelled: Arc<AtomicBool>,
}

impl CanonicalNativeCancellationTokenV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalNativeDiscoveryExecutionStageV1 {
    NativeCapabilityGate,
    RuntimeInstallReceipt,
    SearchGpuExecutionLease,
    ContractReferenceValidation,
    ContractArtifactRead,
    ContractArtifactHash,
    ContractSchemaValidation,
    ExactSourcePin,
    NativePreflight,
    NativeAdmission,
    ResidentDataMaterialization,
    NativeReceiptBinding,
    GenerationZeroEvaluation,
    ConsumerCompletion,
    ResultPublication,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalNativeDiscoveryExecutionErrorCodeV1 {
    UnsupportedPlatform,
    NativeCudaRequired,
    Cancelled,
    InvalidRequest,
    RuntimeAuthorityInvalid,
    ArtifactUnavailable,
    ArtifactHashMismatch,
    ContractInvalid,
    ExactGenerationConflict,
    PreflightRejected,
    AdmissionRejected,
    MaterializationRejected,
    ReceiptRejected,
    EvaluationRejected,
    CompletionRejected,
    ResultSealingRejected,
    PublicationRejected,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalNativeDiscoveryExecutionErrorV1 {
    stage: CanonicalNativeDiscoveryExecutionStageV1,
    code: CanonicalNativeDiscoveryExecutionErrorCodeV1,
    detail: String,
}

impl CanonicalNativeDiscoveryExecutionErrorV1 {
    pub const fn stage(&self) -> CanonicalNativeDiscoveryExecutionStageV1 {
        self.stage
    }

    pub const fn code(&self) -> CanonicalNativeDiscoveryExecutionErrorCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for CanonicalNativeDiscoveryExecutionErrorV1 {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(output, "{:?}/{:?}: {}", self.stage, self.code, self.detail)
    }
}

impl std::error::Error for CanonicalNativeDiscoveryExecutionErrorV1 {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishedCanonicalNativeGenerationZeroResearchV1 {
    relative_path: String,
    byte_count: u64,
    file_sha256: String,
    reused_identical: bool,
    evidence_identity_sha256: String,
    financial_input_receipt_identity_sha256: String,
    native_input_receipt_identity_sha256: String,
    population_sizing_receipt_identity_sha256: String,
    configured_population: usize,
    resolved_population: usize,
    population_cap: usize,
    hard_growth_cap: usize,
    term_cap: usize,
    stage1_row_start: usize,
    stage1_row_end: usize,
    selected_device_ordinal: u32,
    engine: String,
    parent_h2d_bytes: u64,
    adaptive_h2d_bytes: u64,
    metric_rows: u64,
    metric_bytes: u64,
    gene_count: usize,
    metric_row_count: usize,
    metric_value_count_per_row: usize,
    consumer_completion_confirmed: bool,
    replay_identity_sealed: bool,
}

impl PublishedCanonicalNativeGenerationZeroResearchV1 {
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }
    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }
    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
    pub const fn reused_identical(&self) -> bool {
        self.reused_identical
    }
    pub fn evidence_identity_sha256(&self) -> &str {
        &self.evidence_identity_sha256
    }
    pub fn financial_input_receipt_identity_sha256(&self) -> &str {
        &self.financial_input_receipt_identity_sha256
    }
    pub fn native_input_receipt_identity_sha256(&self) -> &str {
        &self.native_input_receipt_identity_sha256
    }
    pub fn population_sizing_receipt_identity_sha256(&self) -> &str {
        &self.population_sizing_receipt_identity_sha256
    }
    pub const fn configured_population(&self) -> usize {
        self.configured_population
    }
    pub const fn resolved_population(&self) -> usize {
        self.resolved_population
    }
    pub const fn population_cap(&self) -> usize {
        self.population_cap
    }
    pub const fn hard_growth_cap(&self) -> usize {
        self.hard_growth_cap
    }
    pub const fn term_cap(&self) -> usize {
        self.term_cap
    }
    pub const fn stage1_row_start(&self) -> usize {
        self.stage1_row_start
    }
    pub const fn stage1_row_end(&self) -> usize {
        self.stage1_row_end
    }
    pub const fn selected_device_ordinal(&self) -> u32 {
        self.selected_device_ordinal
    }
    pub fn engine(&self) -> &str {
        &self.engine
    }
    pub const fn parent_h2d_bytes(&self) -> u64 {
        self.parent_h2d_bytes
    }
    pub const fn adaptive_h2d_bytes(&self) -> u64 {
        self.adaptive_h2d_bytes
    }
    pub const fn metric_rows(&self) -> u64 {
        self.metric_rows
    }
    pub const fn metric_bytes(&self) -> u64 {
        self.metric_bytes
    }
    pub const fn gene_count(&self) -> usize {
        self.gene_count
    }
    pub const fn metric_row_count(&self) -> usize {
        self.metric_row_count
    }
    pub const fn metric_value_count_per_row(&self) -> usize {
        self.metric_value_count_per_row
    }
    pub const fn consumer_completion_confirmed(&self) -> bool {
        self.consumer_completion_confirmed
    }
    pub const fn replay_identity_sealed(&self) -> bool {
        self.replay_identity_sealed
    }
}

fn bounded_detail_v1(mut detail: String) -> String {
    if detail.len() <= MAX_EXECUTION_ERROR_DETAIL_BYTES_V1 {
        return detail;
    }
    let mut end = MAX_EXECUTION_ERROR_DETAIL_BYTES_V1;
    while !detail.is_char_boundary(end) {
        end -= 1;
    }
    detail.truncate(end);
    detail
}

fn execution_error_v1(
    stage: CanonicalNativeDiscoveryExecutionStageV1,
    code: CanonicalNativeDiscoveryExecutionErrorCodeV1,
    detail: impl fmt::Display,
) -> CanonicalNativeDiscoveryExecutionErrorV1 {
    CanonicalNativeDiscoveryExecutionErrorV1 {
        stage,
        code,
        detail: bounded_detail_v1(detail.to_string()),
    }
}

#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
fn probe_cancellation_v1(
    cancellation: &CanonicalNativeCancellationTokenV1,
    stage: CanonicalNativeDiscoveryExecutionStageV1,
) -> Result<(), CanonicalNativeDiscoveryExecutionErrorV1> {
    if cancellation.is_cancelled() {
        Err(execution_error_v1(
            stage,
            CanonicalNativeDiscoveryExecutionErrorCodeV1::Cancelled,
            "canonical native Generation-zero execution was cancelled",
        ))
    } else {
        Ok(())
    }
}

pub fn run_canonical_native_discovery_generation_zero_from_ref_v1<F>(
    startup_settings: &neoethos_core::Settings,
    runtime_install_receipt: &CanonicalNativeRuntimeInstallReceiptV1,
    contract_ref: CanonicalResearchContractArtifactRefV1,
    overrides: CanonicalNativeGenerationZeroOverridesV1,
    cancellation: &CanonicalNativeCancellationTokenV1,
    progress: F,
) -> Result<
    PublishedCanonicalNativeGenerationZeroResearchV1,
    CanonicalNativeDiscoveryExecutionErrorV1,
>
where
    F: FnMut(DiscoveryProgress),
{
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (
            startup_settings,
            runtime_install_receipt,
            contract_ref,
            overrides,
            cancellation,
            progress,
        );
        Err(execution_error_v1(
            CanonicalNativeDiscoveryExecutionStageV1::NativeCapabilityGate,
            CanonicalNativeDiscoveryExecutionErrorCodeV1::UnsupportedPlatform,
            "canonical native Generation-zero execution V1 requires Linux",
        ))
    }
    #[cfg(all(target_os = "linux", not(feature = "gpu-cuda")))]
    {
        let _ = (
            startup_settings,
            runtime_install_receipt,
            contract_ref,
            overrides,
            cancellation,
            progress,
        );
        Err(execution_error_v1(
            CanonicalNativeDiscoveryExecutionStageV1::NativeCapabilityGate,
            CanonicalNativeDiscoveryExecutionErrorCodeV1::NativeCudaRequired,
            "canonical native Generation-zero execution requires the gpu-cuda feature",
        ))
    }
    #[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
    {
        run_canonical_native_discovery_generation_zero_cuda_v1(
            startup_settings,
            runtime_install_receipt,
            contract_ref,
            overrides,
            cancellation,
            progress,
        )
    }
}

#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
#[derive(Debug)]
struct ExecutorCancellationMarkerV1;

#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
impl fmt::Display for ExecutorCancellationMarkerV1 {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output.write_str("canonical native Generation-zero execution was cancelled")
    }
}

#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
impl std::error::Error for ExecutorCancellationMarkerV1 {}

#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
fn map_request_error_v1(
    error: CanonicalNativeDiscoveryRequestErrorV1,
) -> CanonicalNativeDiscoveryExecutionErrorV1 {
    use CanonicalNativeDiscoveryExecutionErrorCodeV1 as Code;
    use CanonicalNativeDiscoveryExecutionStageV1 as Stage;
    let (stage, code) = match &error {
        CanonicalNativeDiscoveryRequestErrorV1::UnsupportedPlatform => {
            (Stage::NativeCapabilityGate, Code::UnsupportedPlatform)
        }
        CanonicalNativeDiscoveryRequestErrorV1::RuntimeAuthority(_)
        | CanonicalNativeDiscoveryRequestErrorV1::MigrationEnabled => {
            (Stage::RuntimeInstallReceipt, Code::RuntimeAuthorityInvalid)
        }
        CanonicalNativeDiscoveryRequestErrorV1::InvalidArtifactReference(_)
        | CanonicalNativeDiscoveryRequestErrorV1::InvalidGenerationZeroOverrides(_)
        | CanonicalNativeDiscoveryRequestErrorV1::RequestLimitExceeded { .. }
        | CanonicalNativeDiscoveryRequestErrorV1::UnsupportedGenerationZeroPolicy { .. } => {
            (Stage::ContractReferenceValidation, Code::InvalidRequest)
        }
        CanonicalNativeDiscoveryRequestErrorV1::ArtifactHashMismatch { .. } => {
            (Stage::ContractArtifactHash, Code::ArtifactHashMismatch)
        }
        CanonicalNativeDiscoveryRequestErrorV1::ContractDecode(_)
        | CanonicalNativeDiscoveryRequestErrorV1::ContractValidation(_)
        | CanonicalNativeDiscoveryRequestErrorV1::ContractSettingsMismatch(_) => {
            (Stage::ContractSchemaValidation, Code::ContractInvalid)
        }
        CanonicalNativeDiscoveryRequestErrorV1::ExactDatasetGenerationConflict(_) => {
            (Stage::ExactSourcePin, Code::ExactGenerationConflict)
        }
        CanonicalNativeDiscoveryRequestErrorV1::DatasetSeries(_) => {
            (Stage::ExactSourcePin, Code::ArtifactUnavailable)
        }
        CanonicalNativeDiscoveryRequestErrorV1::CanonicalRootUnavailable(_)
        | CanonicalNativeDiscoveryRequestErrorV1::SecureResolutionUnavailable(_)
        | CanonicalNativeDiscoveryRequestErrorV1::UnsafeLink
        | CanonicalNativeDiscoveryRequestErrorV1::EscapeOrMount
        | CanonicalNativeDiscoveryRequestErrorV1::RaceDetected
        | CanonicalNativeDiscoveryRequestErrorV1::NonRegularArtifact
        | CanonicalNativeDiscoveryRequestErrorV1::ArtifactTooLarge { .. }
        | CanonicalNativeDiscoveryRequestErrorV1::ArtifactIo(_) => {
            (Stage::ContractArtifactRead, Code::ArtifactUnavailable)
        }
    };
    execution_error_v1(stage, code, error)
}

#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
fn map_exact_pin_error_v1(error: anyhow::Error) -> CanonicalNativeDiscoveryExecutionErrorV1 {
    let code = if error
        .downcast_ref::<neoethos_data::ExactDatasetGenerationConflict>()
        .is_some()
    {
        CanonicalNativeDiscoveryExecutionErrorCodeV1::ExactGenerationConflict
    } else {
        CanonicalNativeDiscoveryExecutionErrorCodeV1::ArtifactUnavailable
    };
    execution_error_v1(
        CanonicalNativeDiscoveryExecutionStageV1::ExactSourcePin,
        code,
        format!("{error:#}"),
    )
}

#[cfg(all(target_os = "linux", feature = "gpu-cuda"))]
fn run_canonical_native_discovery_generation_zero_cuda_v1<F>(
    startup_settings: &neoethos_core::Settings,
    runtime_install_receipt: &CanonicalNativeRuntimeInstallReceiptV1,
    contract_ref: CanonicalResearchContractArtifactRefV1,
    overrides: CanonicalNativeGenerationZeroOverridesV1,
    cancellation: &CanonicalNativeCancellationTokenV1,
    progress: F,
) -> Result<
    PublishedCanonicalNativeGenerationZeroResearchV1,
    CanonicalNativeDiscoveryExecutionErrorV1,
>
where
    F: FnMut(DiscoveryProgress),
{
    use CanonicalNativeDiscoveryExecutionErrorCodeV1 as Code;
    use CanonicalNativeDiscoveryExecutionStageV1 as Stage;

    if !neoethos_gpu_cuda::runtime_available() || neoethos_gpu_cuda::device_count() == 0 {
        return Err(execution_error_v1(
            Stage::NativeCapabilityGate,
            Code::NativeCudaRequired,
            "no physical CUDA runtime/device is available",
        ));
    }

    probe_cancellation_v1(cancellation, Stage::ContractArtifactRead)?;
    let request = resolve_canonical_native_discovery_request_v1(
        startup_settings,
        runtime_install_receipt,
        contract_ref,
        overrides,
    )
    .map_err(map_request_error_v1)?;

    probe_cancellation_v1(cancellation, Stage::ExactSourcePin)?;
    let pinned = neoethos_data::pin_exact_canonical_series_v1(
        startup_settings.system.data_dir.as_path(),
        request.exact_series().clone(),
    )
    .map_err(map_exact_pin_error_v1)?;
    let base_timeframe = request
        .loaded_contract()
        .source_projection()
        .base_timeframe();
    let pinned_parent_rows = pinned.row_count(base_timeframe).map_err(|error| {
        execution_error_v1(Stage::ExactSourcePin, Code::ArtifactUnavailable, error)
    })?;
    let projected_parent_rows = usize::try_from(
        request
            .loaded_contract()
            .source_projection()
            .parent_row_count(),
    )
    .map_err(|_| {
        execution_error_v1(
            Stage::NativePreflight,
            Code::PreflightRejected,
            "projected parent row count does not fit this process",
        )
    })?;
    if pinned_parent_rows != projected_parent_rows {
        return Err(execution_error_v1(
            Stage::NativePreflight,
            Code::PreflightRejected,
            "pinned parent rows disagree with the contract source projection",
        ));
    }

    probe_cancellation_v1(cancellation, Stage::NativePreflight)?;
    request
        .revalidate_before_native_preflight_v1(startup_settings)
        .map_err(map_request_error_v1)?;
    let workspace = neoethos_data::preflight_gpu_only_feature_workspace_v3(
        pinned,
        base_timeframe,
        request.feature_profile(),
        pinned_parent_rows,
    )
    .map_err(|error| execution_error_v1(Stage::NativePreflight, Code::PreflightRejected, error))?;

    probe_cancellation_v1(cancellation, Stage::NativePreflight)?;
    let prepared =
        neoethos_data::prepare_gpu_only_feature_materialization_v3(workspace).map_err(|error| {
            execution_error_v1(Stage::NativePreflight, Code::PreflightRejected, error)
        })?;
    let cpu_projection = canonical_pinned_source_projection_from_search_receipt_v1(
        request.loaded_contract().contract().input_receipt(),
    )
    .map_err(|error| execution_error_v1(Stage::NativePreflight, Code::PreflightRejected, error))?;
    if prepared.pinned_source_projection_v1() != &cpu_projection {
        return Err(execution_error_v1(
            Stage::NativePreflight,
            Code::PreflightRejected,
            "prepared Data source projection disagrees with financial input authority",
        ));
    }

    let feature_count =
        usize::try_from(prepared.workspace_extent().column_count()).map_err(|_| {
            execution_error_v1(
                Stage::NativePreflight,
                Code::PreflightRejected,
                "prepared feature count does not fit this process",
            )
        })?;
    let preflight = preflight_canonical_native_generation_zero_result_v1(&request, feature_count)
        .map_err(|error| {
        execution_error_v1(Stage::NativePreflight, Code::PreflightRejected, error)
    })?;
    let configured_population = preflight.configured_population();
    let population_cap = preflight.population_cap();
    let preflight_term_cap = preflight.term_cap();
    let mut native_config = request.config().clone();
    if preflight.raw_configured_max_indicators() == 0 {
        native_config.max_indicators = preflight.resolved_max_indicators();
    }

    let execution_stage = std::cell::Cell::new(Stage::NativeAdmission);
    let anchor = request.exact_series().anchor().identity().clone();
    let prepared_v5 = prepare_prepared_canonical_trendbar_research_run_input_capped_v5(
        &native_config,
        request.loaded_contract().contract(),
        prepared,
        population_cap,
        |prepared, admitted| {
            execution_stage.set(Stage::ResidentDataMaterialization);
            if cancellation.is_cancelled() {
                return Err(anyhow::Error::new(ExecutorCancellationMarkerV1));
            }
            let store =
                neoethos_data::materialize_prepared_gpu_only_feature_store_for_data_population_v3(
                    prepared, admitted,
                )
                .context("materialize exact admitted resident Data store")?;
            if store.pinned_source_projection_v1() != &cpu_projection {
                anyhow::bail!(
                    "materialized Data source projection disagrees with financial input authority"
                );
            }
            execution_stage.set(Stage::NativeReceiptBinding);
            let receipt =
                CanonicalGpuResidentSearchInputReceiptV3::from_resident_store(&anchor, &store)
                    .map_err(anyhow::Error::new)
                    .context("seal native GPU-resident Search input receipt")?;
            Ok((receipt, store))
        },
    )
    .map_err(|error| {
        let stage = execution_stage.get();
        let code = if error
            .downcast_ref::<ExecutorCancellationMarkerV1>()
            .is_some()
        {
            Code::Cancelled
        } else {
            match stage {
                Stage::ResidentDataMaterialization => Code::MaterializationRejected,
                Stage::NativeReceiptBinding => Code::ReceiptRejected,
                _ => Code::AdmissionRejected,
            }
        };
        execution_error_v1(stage, code, format!("{error:#}"))
    })?;

    let native_receipt_v3 = prepared_v5.native_receipt_v3().clone();
    let sizing_receipt_v2 = prepared_v5.population_sizing_receipt_v2().clone();
    let financial_contract = prepared_v5.financial_contract_v3().clone();
    let evaluation_config = prepared_v5.exact_evaluation_config_v2().clone();
    let hard_growth_cap = sizing_receipt_v2.hard_growth_cap();
    let stage1_row_start = sizing_receipt_v2.stage1_row_start();
    let stage1_row_end = sizing_receipt_v2.stage1_row_end();

    probe_cancellation_v1(cancellation, Stage::GenerationZeroEvaluation)?;
    let milestone = run_prepared_canonical_trendbar_research_generation_zero_gated_typed_v5(
        prepared_v5,
        progress,
        || {
            if cancellation.is_cancelled() {
                Err(anyhow::Error::new(ExecutorCancellationMarkerV1))
            } else {
                Ok(())
            }
        },
    )
    .map_err(|error| match error {
        ResidentGenerationZeroStageErrorV1::PreLaunchGate(source) => {
            let code = if source
                .downcast_ref::<ExecutorCancellationMarkerV1>()
                .is_some()
            {
                Code::Cancelled
            } else {
                Code::EvaluationRejected
            };
            execution_error_v1(Stage::GenerationZeroEvaluation, code, format!("{source:#}"))
        }
        ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation(source) => execution_error_v1(
            Stage::GenerationZeroEvaluation,
            Code::EvaluationRejected,
            format!("{source:#}"),
        ),
        ResidentGenerationZeroStageErrorV1::ConsumerCompletion(source) => execution_error_v1(
            Stage::ConsumerCompletion,
            Code::CompletionRejected,
            format!("{source:#}"),
        ),
    })?;

    probe_cancellation_v1(cancellation, Stage::ResultPublication)?;
    let gene_count = milestone.search_result().genes.len();
    let metric_row_count = milestone.search_result().metrics.len();
    let metric_value_count_per_row = milestone
        .search_result()
        .metrics
        .first()
        .map_or(0, |row| row.len());
    let (view, compact_seal) = seal_canonical_native_generation_zero_research_result_v1(
        &request,
        preflight,
        financial_contract,
        native_receipt_v3,
        sizing_receipt_v2,
        evaluation_config,
        &milestone,
    )
    .map_err(|error| {
        execution_error_v1(Stage::ResultPublication, Code::ResultSealingRejected, error)
    })?;
    let publication = publish_canonical_native_generation_zero_research_result_v1(
        request.canonical_root(),
        &view,
        &compact_seal,
        || {
            if cancellation.is_cancelled() {
                Err(CanonicalNativeGenerationZeroPublicationGateRejectionV1::Cancelled)
            } else {
                Ok(())
            }
        },
    )
    .map_err(|error| {
        let code = match error.kind() {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::PreInstallRejected(
                CanonicalNativeGenerationZeroPublicationGateRejectionV1::Cancelled,
            ) => Code::Cancelled,
            _ => Code::PublicationRejected,
        };
        execution_error_v1(Stage::ResultPublication, code, error)
    })?;

    Ok(PublishedCanonicalNativeGenerationZeroResearchV1 {
        relative_path: publication.relative_path().to_owned(),
        byte_count: publication.byte_count(),
        file_sha256: publication.file_sha256().to_owned(),
        reused_identical: publication.reused_identical(),
        evidence_identity_sha256: publication.evidence_identity_sha256().to_owned(),
        financial_input_receipt_identity_sha256: publication
            .financial_input_receipt_identity_sha256()
            .to_owned(),
        native_input_receipt_identity_sha256: publication
            .native_input_receipt_identity_sha256()
            .to_owned(),
        population_sizing_receipt_identity_sha256: publication
            .population_sizing_receipt_identity_sha256()
            .to_owned(),
        configured_population,
        resolved_population: publication.resolved_population(),
        population_cap,
        hard_growth_cap,
        term_cap: preflight_term_cap,
        stage1_row_start,
        stage1_row_end,
        selected_device_ordinal: publication.selected_device_ordinal(),
        engine: publication.engine().to_owned(),
        parent_h2d_bytes: publication.parent_h2d_bytes(),
        adaptive_h2d_bytes: publication.adaptive_h2d_bytes(),
        metric_rows: publication.metric_rows(),
        metric_bytes: publication.metric_bytes(),
        gene_count,
        metric_row_count,
        metric_value_count_per_row,
        consumer_completion_confirmed: publication.consumer_completion_confirmed(),
        replay_identity_sealed: publication.replay_identity_sealed(),
    })
}
