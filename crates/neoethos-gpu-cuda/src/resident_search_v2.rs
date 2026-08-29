//! Typed ownership for the additive resident Search V2 device seam.
//!
//! Production admission is deliberately fail-closed until the versioned V2
//! RNG and exact current GA semantics have device-oracle coverage. The
//! `cuda-device-fixtures` constructor exists only to exercise ownership and the
//! resident gene-view ABI on a real card; it cannot produce a promotable run.

#[cfg(feature = "cuda-device-fixtures")]
use crate::population::ResidentPopulationMetricsV1;
#[cfg(feature = "cuda-device-fixtures")]
use crate::population::terminal_search_session_destroy_count_fixture_v2;
use crate::population::{
    RawResidentScoringPopulationSourceV2, ResidentSearchPopulationCompletionLeaseV2,
};
use crate::resident_generation_v1::{
    NativeResidentGenerationRunV1 as NativeResidentGenerationRunV2, RawAllocationReceiptV1,
    RawGenerationPlanV1, RawReadyEventV1, SealedResidentGenerationPlanV1,
    ffi_initialize_resident_generation_population_v1,
};
use crate::resident_scoring_v2::{
    NativeResidentScoringRunV2, RawResidentSearchCombinedAdmissionV2,
    RawResidentSearchRuntimeFactsV2, ResidentScoringObjectiveV2, ResidentScoringRunV2,
    ResidentScoringV2Error, SealedResidentSearchAdmissionV2, seal_combined_search_admission_v2,
    seal_resident_scoring_plan_v2,
};
use crate::resident_search_slice2_admission_v2::{
    ResidentSearchSlice2NativeBindAuthorityV2,
    resident_archive_knn_v2_native::{
        NativeResidentArchiveKnnOwnerV2, NativeResidentScoringNoveltyRunV1,
        RawResidentArchiveKnnBindV2, RawResidentArchiveKnnPendingV2,
        RawResidentArchiveKnnTerminalV2, bind_preallocated_resident_archive_knn_v2,
        enqueue_resident_archive_evolve_and_publish_v2, enqueue_resident_archive_score_and_rank_v2,
        enqueue_resident_archive_stage_from_rank_v2, enqueue_resident_archive_terminal_seal_v2,
        neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2,
        try_complete_resident_archive_terminal_v2,
    },
};
use crate::{CudaPopulationError, NeoPopulationSettings, PopulationSession, ScenarioDescriptor};
#[cfg(feature = "cuda-device-fixtures")]
use crate::{NeoPopulationCounters, NeoPopulationMetricRow};
use std::ffi::c_void;
use std::ptr::NonNull;
#[cfg(feature = "cuda-device-fixtures")]
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

#[allow(dead_code)] // Used by the crate-private V3 -> Search consumer seam.
const GENERATION_ABI_V1: u32 = 1;
const GENE_VIEW_ABI_V2: u32 = 2;
const SMC_FLAG_COUNT_V2: u32 = 11;
const STATUS_OK: i32 = 0;
const STATUS_NOT_READY_V2: i32 = 1;
const STATUS_DEVICE_FAULT_V2: i32 = -12;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawResidentGenerationGeneViewV2 {
    abi_version: u32,
    flags: u32,
    seal_device: *const c_void,
    control_device: *const c_void,
    expected_generation_index: u64,
    expected_store_epoch: u64,
    expected_run_token: u64,
    logical_population_count: u64,
    feature_count: u64,
    max_terms_per_gene: u32,
    smc_flag_count: u32,
    plan_identity_sha256: [u8; 32],
    generation_semantics_sha256: [u8; 32],
}

impl Default for RawResidentGenerationGeneViewV2 {
    fn default() -> Self {
        Self {
            abi_version: 0,
            flags: 0,
            seal_device: std::ptr::null(),
            control_device: std::ptr::null(),
            expected_generation_index: 0,
            expected_store_epoch: 0,
            expected_run_token: 0,
            logical_population_count: 0,
            feature_count: 0,
            max_terms_per_gene: 0,
            smc_flag_count: 0,
            plan_identity_sha256: [0; 32],
            generation_semantics_sha256: [0; 32],
        }
    }
}

const _: [(); 136] = [(); std::mem::size_of::<RawResidentGenerationGeneViewV2>()];
const _: [(); 8] = [(); std::mem::align_of::<RawResidentGenerationGeneViewV2>()];
const _: [(); 8] = [(); std::mem::offset_of!(RawResidentGenerationGeneViewV2, seal_device)];
const _: [(); 16] = [(); std::mem::offset_of!(RawResidentGenerationGeneViewV2, control_device)];
const _: [(); 24] =
    [(); std::mem::offset_of!(RawResidentGenerationGeneViewV2, expected_generation_index)];
const _: [(); 48] =
    [(); std::mem::offset_of!(RawResidentGenerationGeneViewV2, logical_population_count)];
const _: [(); 64] = [(); std::mem::offset_of!(RawResidentGenerationGeneViewV2, max_terms_per_gene)];
const _: [(); 72] =
    [(); std::mem::offset_of!(RawResidentGenerationGeneViewV2, plan_identity_sha256)];
const _: [(); 104] =
    [(); std::mem::offset_of!(RawResidentGenerationGeneViewV2, generation_semantics_sha256)];
const _: [(); 176] = [(); std::mem::size_of::<RawAllocationReceiptV1>()];
const _: [(); 8] = [(); std::mem::align_of::<RawAllocationReceiptV1>()];
const _: [(); 96] = [(); std::mem::offset_of!(RawAllocationReceiptV1, total_device_bytes)];
const _: [(); 144] = [(); std::mem::offset_of!(RawAllocationReceiptV1, allocation_plan_sha256)];

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RawResidentSearchTerminalReceiptV2 {
    abi_version: u32,
    terminal_status: u32,
    scoring_device_fault: u32,
    generation_device_fault: u32,
    control_fault_word: u32,
    stop_requested: u32,
    current_store_index: u32,
    reserved: u32,
    generation_index: u64,
    store_epoch: u64,
    run_token: u64,
    compact_async_d2h_count: u64,
    compact_async_d2h_bytes: u64,
    completion_event_query_count: u64,
    completion_stream_synchronize_count: u64,
    same_stream_enqueue_count: u64,
    completion_event_id: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawResidentSearchAdvancePendingReceiptV2 {
    abi_version: u32,
    reserved: u32,
    completion_event_id: u64,
    target_generation_index: u64,
    target_store_epoch: u64,
    target_store_index: u64,
    run_token: u64,
    same_stream_enqueue_count: u64,
    dependency_receipt_token: *const RawReadyEventV1,
    terminal_host_receipt_token: *const RawResidentSearchTerminalReceiptV2,
}

impl Default for RawResidentSearchAdvancePendingReceiptV2 {
    fn default() -> Self {
        Self {
            abi_version: 0,
            reserved: 0,
            completion_event_id: 0,
            target_generation_index: 0,
            target_store_epoch: 0,
            target_store_index: 0,
            run_token: 0,
            same_stream_enqueue_count: 0,
            dependency_receipt_token: std::ptr::null(),
            terminal_host_receipt_token: std::ptr::null(),
        }
    }
}

const _: [(); 104] = [(); std::mem::size_of::<RawResidentSearchTerminalReceiptV2>()];
const _: [(); 72] = [(); std::mem::size_of::<RawResidentSearchAdvancePendingReceiptV2>()];

#[cfg(feature = "cuda-device-fixtures")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RawResidentGenerationGeneScalarFixtureV2 {
    pub(crate) gene_identity: u64,
    pub(crate) content_hash: u64,
    pub(crate) term_count: u32,
    pub(crate) smc_flags: u32,
    pub(crate) long_threshold: f64,
    pub(crate) short_threshold: f64,
    pub(crate) target_pips: f64,
    pub(crate) stop_pips: f64,
    pub(crate) stop_vol_multiplier: f64,
    pub(crate) generation: u32,
    pub(crate) reserved: u32,
}

#[cfg(feature = "cuda-device-fixtures")]
const _: [(); 72] = [(); std::mem::size_of::<RawResidentGenerationGeneScalarFixtureV2>()];

#[cfg(feature = "cuda-device-fixtures")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct RawResidentScoringFixtureSnapshotV2 {
    abi_version: u32,
    scoring_objective: u32,
    device_fault_word: u32,
    reserved: u32,
    logical_population_count: u64,
    terminal_synchronization_count: u64,
    terminal_readback_count: u64,
    terminal_readback_bytes: u64,
}

#[cfg(feature = "cuda-device-fixtures")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct RawResidentGenerationAdvanceFixtureSnapshotV2 {
    abi_version: u32,
    device_content_fault: u32,
    gene_hash_collision_fault: u32,
    control_fault_word: u32,
    stop_requested: u32,
    current_store_index: u32,
    max_terms_per_gene: u32,
    survivor_count: u32,
    selected_count: u32,
    dedup_run_count: u32,
    reserved: u32,
    logical_population_count: u64,
    generation_index: u64,
    store_epoch: u64,
    terminal_synchronization_count: u64,
    terminal_readback_count: u64,
    terminal_readback_bytes: u64,
}

#[cfg(feature = "cuda-device-fixtures")]
const _: [(); 48] = [(); std::mem::size_of::<RawResidentScoringFixtureSnapshotV2>()];
#[cfg(feature = "cuda-device-fixtures")]
const _: [(); 96] = [(); std::mem::size_of::<RawResidentGenerationAdvanceFixtureSnapshotV2>()];

unsafe extern "C" {
    #[allow(dead_code)] // Reached by the next crate-private V3 consumer.
    fn neoethos_gpu_cuda_population_create_resident_generation_run_v2(
        session: *mut c_void,
        plan: *const RawGenerationPlanV1,
        allocation: *mut RawAllocationReceiptV1,
        run: *mut *mut NativeResidentGenerationRunV2,
    ) -> i32;
    #[allow(dead_code)] // Reached by the next crate-private V3 consumer.
    fn configure_resident_generation_evaluator_v2(
        run: *mut NativeResidentGenerationRunV2,
        dependency: *const RawReadyEventV1,
        smc_weights: *const f64,
        smc_gate_disabled: u32,
        ready: *mut RawReadyEventV1,
    ) -> i32;
    fn export_current_resident_gene_view_v2(
        run: *mut NativeResidentGenerationRunV2,
        ready: *const RawReadyEventV1,
        view: *mut RawResidentGenerationGeneViewV2,
    ) -> i32;
    fn neoethos_gpu_cuda_population_release_resident_generation_run_v2(
        session: *mut c_void,
        run: *mut NativeResidentGenerationRunV2,
    ) -> i32;
    fn enqueue_full_population_scored_generation_advance_v2(
        generation: *mut NativeResidentGenerationRunV2,
        scoring: *mut NativeResidentScoringRunV2,
        population: *const RawResidentScoringPopulationSourceV2,
        dependency: *const RawReadyEventV1,
        pending: *mut RawResidentSearchAdvancePendingReceiptV2,
    ) -> i32;
    fn try_complete_resident_generation_advance_v2(
        generation: *mut NativeResidentGenerationRunV2,
        pending: *const RawResidentSearchAdvancePendingReceiptV2,
        committed_ready: *mut RawReadyEventV1,
        terminal_copy: *mut RawResidentSearchTerminalReceiptV2,
    ) -> i32;
    fn neoethos_gpu_cuda_population_reserve_resident_search_runtime_v2(
        session: *mut c_void,
        facts: *mut RawResidentSearchRuntimeFactsV2,
    ) -> i32;
    fn neoethos_gpu_cuda_population_query_resident_search_combined_v2(
        session: *mut c_void,
        generation_plan: *const RawGenerationPlanV1,
        scoring_plan: *const crate::resident_scoring_v2::RawResidentScoringPlanV2,
        expected_runtime: *const RawResidentSearchRuntimeFactsV2,
        admission: *mut RawResidentSearchCombinedAdmissionV2,
    ) -> i32;
    fn neoethos_gpu_cuda_population_create_resident_search_combined_v2(
        session: *mut c_void,
        generation_plan: *const RawGenerationPlanV1,
        scoring_plan: *const crate::resident_scoring_v2::RawResidentScoringPlanV2,
        admission: *const RawResidentSearchCombinedAdmissionV2,
        generation: *mut *mut NativeResidentGenerationRunV2,
        scoring: *mut *mut NativeResidentScoringRunV2,
    ) -> i32;
    fn neoethos_gpu_cuda_population_query_resident_search_slice2_v3(
        session: *mut c_void,
        generation_plan: *const RawGenerationPlanV1,
        scoring_plan: *const crate::resident_scoring_v2::RawResidentScoringPlanV2,
        expected_runtime: *const RawResidentSearchRuntimeFactsV2,
        binding: *const RawResidentArchiveKnnBindV2,
        admission: *mut RawResidentSearchCombinedAdmissionV2,
    ) -> i32;
    fn neoethos_gpu_cuda_population_create_resident_search_slice2_v3(
        session: *mut c_void,
        generation_plan: *const RawGenerationPlanV1,
        scoring_plan: *const crate::resident_scoring_v2::RawResidentScoringPlanV2,
        admission: *const RawResidentSearchCombinedAdmissionV2,
        binding: *const RawResidentArchiveKnnBindV2,
        generation: *mut *mut NativeResidentGenerationRunV2,
        scoring: *mut *mut NativeResidentScoringRunV2,
    ) -> i32;
    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Linked only by the feature-gated device oracle.
    fn fixture_set_resident_scoring_metric_mode_v2(
        scoring: *mut NativeResidentScoringRunV2,
        mode: u32,
    ) -> i32;
    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Linked only by the feature-gated device oracle.
    fn fixture_set_resident_scoring_metric_fault_v2(
        scoring: *mut NativeResidentScoringRunV2,
        metric_slot: u32,
        nonfinite_bits: u64,
    ) -> i32;
    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Linked only by the feature-gated device oracle.
    fn fixture_set_resident_generation_gene_identity_v2(
        generation: *mut NativeResidentGenerationRunV2,
        candidate: u64,
        gene_identity: u64,
    ) -> i32;
    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Linked only by the feature-gated device oracle.
    fn fixture_set_duplicate_final_gene_content_v2(
        generation: *mut NativeResidentGenerationRunV2,
        source_candidate: u64,
        destination_candidate: u64,
    ) -> i32;
    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Linked only by the feature-gated device oracle.
    fn fixture_copy_resident_scoring_snapshot_v2(
        scoring: *mut NativeResidentScoringRunV2,
        metric_rows_host: *mut NeoPopulationMetricRow,
        fitness_scores_host: *mut f64,
        decision_keys_host: *mut u64,
        capacity: u64,
        snapshot: *mut RawResidentScoringFixtureSnapshotV2,
    ) -> i32;
    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Linked only by the feature-gated device oracle.
    fn fixture_copy_resident_generation_advance_snapshot_v2(
        generation: *mut NativeResidentGenerationRunV2,
        ranked_population_ordinals_host: *mut u64,
        initial_genes_host: *mut RawResidentGenerationGeneScalarFixtureV2,
        final_genes_host: *mut RawResidentGenerationGeneScalarFixtureV2,
        initial_term_indices_host: *mut u64,
        initial_term_weights_host: *mut f64,
        final_term_indices_host: *mut u64,
        final_term_weights_host: *mut f64,
        parent_a_host: *mut u64,
        parent_b_host: *mut u64,
        selected_survivors_host: *mut u64,
        sorted_dedup_flags_host: *mut u8,
        candidate_valid_flags_host: *mut u8,
        population_capacity: u64,
        term_capacity: u64,
        survivor_capacity: u64,
        snapshot: *mut RawResidentGenerationAdvanceFixtureSnapshotV2,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentSearchStateV2 {
    Active,
    Advancing,
    AdvancePending,
    AdvancedOnce,
    TerminalEnqueued,
    Consumed,
    Poisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentSearchProductionReadinessV2 {
    exact_generation_semantics: bool,
    device_resident_generation_advance: bool,
    device_owned_search_control: bool,
    immutable_scenario_admission: bool,
    whole_workspace_preallocated: bool,
    unified_device_fault_authority: bool,
    native_bridge_production_sealed: bool,
    terminal_cleanup_lease: bool,
}

impl ResidentSearchProductionReadinessV2 {
    pub const fn exact_generation_semantics(&self) -> bool {
        self.exact_generation_semantics
    }

    pub const fn device_resident_generation_advance(&self) -> bool {
        self.device_resident_generation_advance
    }

    pub const fn device_owned_search_control(&self) -> bool {
        self.device_owned_search_control
    }

    pub const fn immutable_scenario_admission(&self) -> bool {
        self.immutable_scenario_admission
    }

    pub const fn whole_workspace_preallocated(&self) -> bool {
        self.whole_workspace_preallocated
    }

    pub const fn unified_device_fault_authority(&self) -> bool {
        self.unified_device_fault_authority
    }

    pub const fn native_bridge_production_sealed(&self) -> bool {
        self.native_bridge_production_sealed
    }

    pub const fn terminal_cleanup_lease(&self) -> bool {
        self.terminal_cleanup_lease
    }

    pub const fn production_ready(&self) -> bool {
        self.exact_generation_semantics
            && self.device_resident_generation_advance
            && self.device_owned_search_control
            && self.immutable_scenario_admission
            && self.whole_workspace_preallocated
            && self.unified_device_fault_authority
            && self.native_bridge_production_sealed
            && self.terminal_cleanup_lease
    }
}

pub const fn resident_search_v2_production_readiness() -> ResidentSearchProductionReadinessV2 {
    ResidentSearchProductionReadinessV2 {
        exact_generation_semantics: false,
        device_resident_generation_advance: false,
        device_owned_search_control: true,
        immutable_scenario_admission: false,
        whole_workspace_preallocated: false,
        unified_device_fault_authority: false,
        native_bridge_production_sealed: true,
        terminal_cleanup_lease: false,
    }
}

#[derive(Debug, Error)]
pub enum ResidentSearchV2Error {
    #[error(
        "resident Search V2 is not production-ready: exact GA/RNG semantics, device-resident generation advance, immutable scenario admission, whole-workspace admission and unified device fault authority remain fail-closed"
    )]
    ResidentGenerationSemanticsNotProductionReady,
    #[error(transparent)]
    Population(#[from] CudaPopulationError),
    #[error("invalid resident Search V2 admission: {0}")]
    InvalidAdmission(#[source] CudaPopulationError),
    #[error("invalid resident Search V2 plan: {0}")]
    InvalidPlan(&'static str),
    #[error("resident Search V2 native operation {operation} failed with status {status}")]
    Native {
        operation: &'static str,
        status: i32,
    },
    #[error("resident Search V2 ready receipt lost its stable boxed address")]
    ReadyReceiptAddressChanged,
    #[error("resident Search V2 gene-view identity differs from the sealed owner")]
    GeneViewIdentityMismatch,
    #[error("resident Search V2 owner state does not permit this operation")]
    StateViolation,
    #[error("one resident generation advance is one-shot")]
    OneGenerationAdvanceAlreadyEnqueued,
    #[error("resident Search V2 device terminal receipt reported a fault")]
    DeviceTerminalFault(ResidentSearchTerminalReceiptV2),
    #[error("resident Search V2 terminal fault cleanup failed: {reason}")]
    DeviceTerminalFaultCleanup {
        receipt: ResidentSearchTerminalReceiptV2,
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentSearchTerminalReceiptV2 {
    raw: RawResidentSearchTerminalReceiptV2,
}

impl ResidentSearchTerminalReceiptV2 {
    /// Bounded read-only alias for the terminal control fault word. The
    /// receipt exposes no native handle or device pointer.
    pub const fn device_fault_word(&self) -> u32 {
        self.raw.control_fault_word
    }

    pub const fn scoring_device_fault(&self) -> u32 {
        self.raw.scoring_device_fault
    }
    pub const fn generation_device_fault(&self) -> u32 {
        self.raw.generation_device_fault
    }
    pub const fn control_fault_word(&self) -> u32 {
        self.raw.control_fault_word
    }
    pub const fn stop_requested(&self) -> u32 {
        self.raw.stop_requested
    }
    pub const fn current_store_index(&self) -> u32 {
        self.raw.current_store_index
    }
    pub const fn generation_index(&self) -> u64 {
        self.raw.generation_index
    }
    pub const fn store_epoch(&self) -> u64 {
        self.raw.store_epoch
    }
    pub const fn compact_async_d2h_count(&self) -> u64 {
        self.raw.compact_async_d2h_count
    }
    pub const fn compact_async_d2h_bytes(&self) -> u64 {
        self.raw.compact_async_d2h_bytes
    }
    pub const fn completion_event_query_count(&self) -> u64 {
        self.raw.completion_event_query_count
    }
    pub const fn completion_stream_synchronize_count(&self) -> u64 {
        self.raw.completion_stream_synchronize_count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentGeneViewSummaryV2 {
    abi_version: u32,
    has_device_seal: bool,
    generation_index: u64,
    store_epoch: u64,
    run_token: u64,
    logical_population_count: u64,
    feature_count: u64,
    max_terms_per_gene: u32,
    smc_flag_count: u32,
    plan_identity_sha256: [u8; 32],
    generation_semantics_sha256: [u8; 32],
}

impl ResidentGeneViewSummaryV2 {
    pub const fn abi_version(&self) -> u32 {
        self.abi_version
    }

    pub const fn has_device_seal(&self) -> bool {
        self.has_device_seal
    }

    pub const fn generation_index(&self) -> u64 {
        self.generation_index
    }

    pub const fn store_epoch(&self) -> u64 {
        self.store_epoch
    }

    pub const fn run_token(&self) -> u64 {
        self.run_token
    }

    pub const fn logical_population_count(&self) -> u64 {
        self.logical_population_count
    }

    pub const fn feature_count(&self) -> u64 {
        self.feature_count
    }

    pub const fn max_terms_per_gene(&self) -> u32 {
        self.max_terms_per_gene
    }

    pub const fn smc_flag_count(&self) -> u32 {
        self.smc_flag_count
    }

    pub const fn plan_identity_sha256(&self) -> [u8; 32] {
        self.plan_identity_sha256
    }

    pub const fn generation_semantics_sha256(&self) -> [u8; 32] {
        self.generation_semantics_sha256
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentReadyEventSummaryV2 {
    abi_version: u32,
    reserved_is_zero: bool,
    event_id: u64,
    generation_index: u64,
    same_stream_enqueue_count: u64,
    intermediate_host_wait_count: u64,
    intermediate_readback_count: u64,
}

impl ResidentReadyEventSummaryV2 {
    pub const fn abi_version(&self) -> u32 {
        self.abi_version
    }

    pub const fn reserved_is_zero(&self) -> bool {
        self.reserved_is_zero
    }

    pub const fn event_id(&self) -> u64 {
        self.event_id
    }

    pub const fn generation_index(&self) -> u64 {
        self.generation_index
    }

    pub const fn same_stream_enqueue_count(&self) -> u64 {
        self.same_stream_enqueue_count
    }

    pub const fn intermediate_host_wait_count(&self) -> u64 {
        self.intermediate_host_wait_count
    }

    pub const fn intermediate_readback_count(&self) -> u64 {
        self.intermediate_readback_count
    }
}

#[cfg(feature = "cuda-device-fixtures")]
#[cfg_attr(not(test), allow(dead_code))] // Materialized only by the feature-gated device oracle.
pub(crate) struct ResidentSearchGenerationFixtureSnapshotV2 {
    pub(crate) metric_rows: Vec<NeoPopulationMetricRow>,
    pub(crate) fitness_scores: Vec<f64>,
    pub(crate) decision_keys: Vec<u64>,
    pub(crate) ranked_population_ordinals: Vec<u64>,
    pub(crate) initial_genes: Vec<ResidentSearchFixtureGeneV2>,
    pub(crate) final_genes: Vec<ResidentSearchFixtureGeneV2>,
    pub(crate) parent_a: Vec<u64>,
    pub(crate) parent_b: Vec<u64>,
    pub(crate) selected_survivors: Vec<u64>,
    pub(crate) sorted_dedup_flags: Vec<u8>,
    pub(crate) candidate_valid_flags: Vec<u8>,
    pub(crate) selected_count: u32,
    pub(crate) dedup_run_count: u32,
    pub(crate) scoring_objective: u32,
    pub(crate) scoring_device_fault: u32,
    pub(crate) generation_device_fault: u32,
    pub(crate) gene_hash_collision_fault: u32,
    pub(crate) control_fault_word: u32,
    pub(crate) stop_requested: u32,
    pub(crate) current_store_index: u32,
    pub(crate) generation_index: u64,
    pub(crate) store_epoch: u64,
    pub(crate) terminal_synchronization_count: u64,
    pub(crate) terminal_readback_count: u64,
    pub(crate) terminal_readback_bytes: u64,
    pub(crate) population_counters: NeoPopulationCounters,
}

#[cfg(feature = "cuda-device-fixtures")]
#[cfg_attr(not(test), allow(dead_code))] // Materialized only by the feature-gated device oracle.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResidentSearchFixtureGeneV2 {
    pub(crate) gene_identity: u64,
    pub(crate) content_hash: u64,
    pub(crate) term_count: u32,
    pub(crate) smc_flags: u32,
    pub(crate) long_threshold: f64,
    pub(crate) short_threshold: f64,
    pub(crate) target_pips: f64,
    pub(crate) stop_pips: f64,
    pub(crate) stop_vol_multiplier: f64,
    pub(crate) generation: u32,
    pub(crate) term_indices: [u64; 3],
    pub(crate) term_weights: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentSearchCombinedAdmissionSummaryV2 {
    free_memory_snapshot_count: u32,
    generation_allocation_count: u32,
    scoring_allocation_count: u32,
    terminal_host_allocation_count: u32,
    terminal_host_receipt_bytes: u64,
    same_context_free_bytes: u64,
    full_discovery_reserve_bytes: u64,
    total_device_bytes: u64,
    runtime_identity_is_exact: bool,
    sealed_before_first_allocation: bool,
}

impl ResidentSearchCombinedAdmissionSummaryV2 {
    pub const fn free_memory_snapshot_count(&self) -> u32 {
        self.free_memory_snapshot_count
    }
    pub const fn generation_allocation_count(&self) -> u32 {
        self.generation_allocation_count
    }
    pub const fn scoring_allocation_count(&self) -> u32 {
        self.scoring_allocation_count
    }
    pub const fn terminal_host_allocation_count(&self) -> u32 {
        self.terminal_host_allocation_count
    }
    pub const fn terminal_host_receipt_bytes(&self) -> u64 {
        self.terminal_host_receipt_bytes
    }
    pub const fn same_context_free_bytes(&self) -> u64 {
        self.same_context_free_bytes
    }
    pub const fn full_discovery_reserve_bytes(&self) -> u64 {
        self.full_discovery_reserve_bytes
    }
    pub const fn total_device_bytes(&self) -> u64 {
        self.total_device_bytes
    }
    pub const fn runtime_identity_is_exact(&self) -> bool {
        self.runtime_identity_is_exact
    }
    pub const fn sealed_before_first_allocation(&self) -> bool {
        self.sealed_before_first_allocation
    }
}

#[cfg(feature = "cuda-device-fixtures")]
#[cfg_attr(not(test), allow(dead_code))] // Queried only by the feature-gated device oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchPendingDropAuditV2 {
    poisoned_pending_drop_count: u64,
    reused_in_flight_session_count: u64,
    terminal_fault_cleanup_count: u64,
    terminal_session_destroy_count: u64,
}

#[cfg(feature = "cuda-device-fixtures")]
impl ResidentSearchPendingDropAuditV2 {
    #[cfg_attr(not(test), allow(dead_code))] // Queried only by the feature-gated device oracle.
    pub(crate) const fn poisoned_pending_drop_count(&self) -> u64 {
        self.poisoned_pending_drop_count
    }
    #[cfg_attr(not(test), allow(dead_code))] // Queried only by the feature-gated device oracle.
    pub(crate) const fn reused_in_flight_session_count(&self) -> u64 {
        self.reused_in_flight_session_count
    }
    #[cfg_attr(not(test), allow(dead_code))] // Queried only by the feature-gated device oracle.
    pub(crate) const fn terminal_fault_cleanup_count(&self) -> u64 {
        self.terminal_fault_cleanup_count
    }
    #[cfg_attr(not(test), allow(dead_code))] // Queried only by the feature-gated device oracle.
    pub(crate) const fn terminal_session_destroy_count(&self) -> u64 {
        self.terminal_session_destroy_count
    }
}

#[cfg(feature = "cuda-device-fixtures")]
static PENDING_DROP_POISON_COUNT_V2: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "cuda-device-fixtures")]
static TERMINAL_FAULT_CLEANUP_COUNT_V2: AtomicU64 = AtomicU64::new(0);

/// Move-only owner of the exact population session, native generation run and
/// stable ready receipt. It contains no type-erased owner and no forgotten Rust
/// allocation. Exceptional cleanup failure poisons the population session so
/// its existing fail-closed Drop policy does not free reachable CUDA storage.
pub struct ResidentSearchRunV2 {
    session: Option<PopulationSession>,
    generation: Option<NonNull<NativeResidentGenerationRunV2>>,
    scoring: Option<ResidentScoringRunV2>,
    admission: Option<SealedResidentSearchAdmissionV2>,
    ready: Option<Box<RawReadyEventV1>>,
    ready_receipt_address: usize,
    view: RawResidentGenerationGeneViewV2,
    expected_population: u64,
    expected_feature_count: u64,
    expected_max_terms: u32,
    #[cfg(feature = "cuda-device-fixtures")]
    expected_survivor_count: u64,
    retained_evaluation_capacity: u64,
    expected_plan_identity_sha256: [u8; 32],
    expected_generation_semantics_sha256: [u8; 32],
    state: ResidentSearchStateV2,
    terminal_receipt: Option<ResidentSearchTerminalReceiptV2>,
    #[cfg(feature = "cuda-device-fixtures")]
    last_population_counters_fixture_v2: Option<NeoPopulationCounters>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentSearchSlice2NativeStateV3 {
    Bound,
    Ranked,
    Staged,
    Published,
    TerminalPending,
    TerminalComplete,
    Poisoned,
    Released,
}

/// Move-only owner of the complete Slice2 native graph. The existing Search
/// run retains the population session, generation/scoring owners, stable ready
/// receipt, gene view and sealed admission; this wrapper additionally retains
/// the exact bind authority and the single native archive owner.
#[allow(dead_code)] // Consumed by the typed V3 transitions in the next wiring slice.
pub(crate) struct ResidentSearchSlice2NativeOwnerV3 {
    run: Option<ResidentSearchRunV2>,
    bind_authority: Option<ResidentSearchSlice2NativeBindAuthorityV2>,
    archive: Option<NonNull<NativeResidentArchiveKnnOwnerV2>>,
    population_source: Option<ResidentSearchPopulationCompletionLeaseV2>,
    pending: Option<Box<RawResidentArchiveKnnPendingV2>>,
    terminal: Option<RawResidentArchiveKnnTerminalV2>,
    state: ResidentSearchSlice2NativeStateV3,
}

#[derive(Debug, Error)]
pub(crate) enum ResidentSearchSlice2NativeErrorV3 {
    #[error(transparent)]
    Search(#[from] ResidentSearchV2Error),
    #[error("resident Search Slice2 native operation {operation} failed with status {status}")]
    Native {
        operation: &'static str,
        status: i32,
    },
    #[error("resident Search Slice2 native owner state violation")]
    StateViolation,
}

#[allow(dead_code)] // Converted into the public typed rejection in the next wiring slice.
pub(crate) struct ResidentSearchSlice2NativeRejectedV3 {
    error: ResidentSearchSlice2NativeErrorV3,
    owner: ResidentSearchSlice2NativeOwnerV3,
}

#[allow(dead_code)]
pub(crate) enum ResidentSearchSlice2NativeTryCompleteV3 {
    NotReady(ResidentSearchSlice2NativeOwnerV3),
    Complete(ResidentSearchSlice2NativeOwnerV3),
}

#[allow(dead_code)]
impl ResidentSearchSlice2NativeRejectedV3 {
    pub(crate) fn into_parts_v3(
        self,
    ) -> (
        ResidentSearchSlice2NativeErrorV3,
        ResidentSearchSlice2NativeOwnerV3,
    ) {
        (self.error, self.owner)
    }
}

#[allow(dead_code)]
impl ResidentSearchSlice2NativeOwnerV3 {
    fn reject_v3(
        mut self,
        operation: &'static str,
        status: i32,
    ) -> ResidentSearchSlice2NativeRejectedV3 {
        if let Some(source) = self.population_source.as_mut() {
            source.poison_without_reuse_v2();
        }
        if let Some(run) = self.run.as_mut() {
            run.state = ResidentSearchStateV2::Poisoned;
            if let Some(scoring) = run.scoring.as_mut() {
                scoring.poison_v2();
            }
            if let Some(session) = run.session.as_mut() {
                session.poison_resident_search_owner_v2();
            }
        }
        self.state = ResidentSearchSlice2NativeStateV3::Poisoned;
        ResidentSearchSlice2NativeRejectedV3 {
            error: ResidentSearchSlice2NativeErrorV3::Native { operation, status },
            owner: self,
        }
    }

    fn reject_state_v3(mut self) -> ResidentSearchSlice2NativeRejectedV3 {
        if let Some(run) = self.run.as_mut() {
            run.state = ResidentSearchStateV2::Poisoned;
        }
        self.state = ResidentSearchSlice2NativeStateV3::Poisoned;
        ResidentSearchSlice2NativeRejectedV3 {
            error: ResidentSearchSlice2NativeErrorV3::StateViolation,
            owner: self,
        }
    }

    fn bind_v3(
        mut run: ResidentSearchRunV2,
        bind_authority: ResidentSearchSlice2NativeBindAuthorityV2,
    ) -> Result<Self, ResidentSearchSlice2NativeErrorV3> {
        let generation = run
            .generation
            .ok_or(ResidentSearchSlice2NativeErrorV3::StateViolation)?;
        let scoring = run
            .scoring
            .as_ref()
            .and_then(ResidentScoringRunV2::native_v2)
            .ok_or(ResidentSearchSlice2NativeErrorV3::StateViolation)?;
        let mut archive = std::ptr::null_mut();
        // SAFETY: the additive composite creator retained the exact bind before
        // either arena allocation. All pointers below belong to that same
        // admitted run and the stable gene view remains owned by `run`.
        let status = unsafe {
            bind_preallocated_resident_archive_knn_v2(
                scoring.as_ptr().cast::<NativeResidentScoringNoveltyRunV1>(),
                generation.as_ptr(),
                &run.view,
                bind_authority.raw_v2(),
                &mut archive,
            )
        };
        let Some(archive) = NonNull::new(archive) else {
            run.state = ResidentSearchStateV2::Poisoned;
            if let Some(scoring) = run.scoring.as_mut() {
                scoring.poison_v2();
            }
            return Err(ResidentSearchSlice2NativeErrorV3::Native {
                operation: "bind_preallocated_resident_archive_knn_v2",
                status,
            });
        };
        if status != STATUS_OK {
            run.state = ResidentSearchStateV2::Poisoned;
            if let Some(scoring) = run.scoring.as_mut() {
                scoring.poison_v2();
            }
            // Native returned an owner together with failure. Deliberately do
            // not release that raw pointer: its exact graph may be reachable
            // by queued work, so the fail-closed policy is to leak it.
            return Err(ResidentSearchSlice2NativeErrorV3::Native {
                operation: "bind_preallocated_resident_archive_knn_v2",
                status,
            });
        }
        run.scoring
            .as_mut()
            .ok_or(ResidentSearchSlice2NativeErrorV3::StateViolation)?
            .mark_bound_v2();
        Ok(Self {
            run: Some(run),
            bind_authority: Some(bind_authority),
            archive: Some(archive),
            population_source: None,
            pending: None,
            terminal: None,
            state: ResidentSearchSlice2NativeStateV3::Bound,
        })
    }

    pub(crate) fn upload_resident_scenarios_v3(
        &mut self,
        scenarios: &[ScenarioDescriptor],
    ) -> Result<(), ResidentSearchSlice2NativeErrorV3> {
        self.run
            .as_mut()
            .ok_or(ResidentSearchSlice2NativeErrorV3::StateViolation)?
            .upload_resident_scenarios_v2(scenarios)?;
        Ok(())
    }

    pub(crate) fn enqueue_score_and_rank_v3(
        mut self,
        settings: &NeoPopulationSettings,
    ) -> Result<Self, ResidentSearchSlice2NativeRejectedV3> {
        if !matches!(
            self.state,
            ResidentSearchSlice2NativeStateV3::Bound | ResidentSearchSlice2NativeStateV3::Published
        ) || self.population_source.is_some()
        {
            return Err(self.reject_state_v3());
        }
        let run = self.run.as_mut().expect("Slice2 owner retains Search run");
        let session = match run.session.take() {
            Some(session) => session,
            None => return Err(self.reject_state_v3()),
        };
        let source = match session.enqueue_resident_gene_metrics_owned_v2(
            &run.view,
            settings,
            run.expected_population,
            run.retained_evaluation_capacity,
            run.expected_feature_count,
            run.expected_max_terms,
            run.admission
                .as_ref()
                .map_or(0, |admission| admission.full_discovery_reserve_bytes),
        ) {
            Ok(source) => source,
            Err(rejected) => {
                let (error, session) = rejected.into_parts_v2();
                run.session = Some(session);
                run.state = ResidentSearchStateV2::Poisoned;
                self.state = ResidentSearchSlice2NativeStateV3::Poisoned;
                return Err(ResidentSearchSlice2NativeRejectedV3 {
                    error: ResidentSearchSlice2NativeErrorV3::Search(error.into()),
                    owner: self,
                });
            }
        };
        let dependency = if self.state == ResidentSearchSlice2NativeStateV3::Bound {
            run.ready
                .as_deref()
                .map_or(std::ptr::null(), std::ptr::from_ref)
        } else {
            std::ptr::null()
        };
        let archive = self.archive.expect("Slice2 owner retains archive");
        let status = unsafe {
            enqueue_resident_archive_score_and_rank_v2(
                archive.as_ptr(),
                source.raw_source_v2(),
                dependency,
            )
        };
        self.population_source = Some(source);
        if status != STATUS_OK {
            return Err(self.reject_v3("enqueue_resident_archive_score_and_rank_v2", status));
        }
        self.state = ResidentSearchSlice2NativeStateV3::Ranked;
        Ok(self)
    }

    pub(crate) fn enqueue_stage_archive_from_rank_v3(
        mut self,
    ) -> Result<Self, ResidentSearchSlice2NativeRejectedV3> {
        if self.state != ResidentSearchSlice2NativeStateV3::Ranked {
            return Err(self.reject_state_v3());
        }
        let status = unsafe {
            enqueue_resident_archive_stage_from_rank_v2(
                self.archive.expect("Slice2 owner retains archive").as_ptr(),
            )
        };
        if status != STATUS_OK {
            return Err(self.reject_v3("enqueue_resident_archive_stage_from_rank_v2", status));
        }
        self.state = ResidentSearchSlice2NativeStateV3::Staged;
        Ok(self)
    }

    pub(crate) fn enqueue_evolve_and_publish_v3(
        mut self,
    ) -> Result<Self, ResidentSearchSlice2NativeRejectedV3> {
        if self.state != ResidentSearchSlice2NativeStateV3::Staged {
            return Err(self.reject_state_v3());
        }
        let status = unsafe {
            enqueue_resident_archive_evolve_and_publish_v2(
                self.archive.expect("Slice2 owner retains archive").as_ptr(),
            )
        };
        if status != STATUS_OK {
            return Err(self.reject_v3("enqueue_resident_archive_evolve_and_publish_v2", status));
        }
        let source = self
            .population_source
            .take()
            .expect("ranked Slice2 owner retains population source");
        match source.finish_device_consume_v2() {
            Ok(session) => {
                let run = self.run.as_mut().expect("Slice2 owner retains Search run");
                let Some(next_generation) = run.view.expected_generation_index.checked_add(1)
                else {
                    run.session = Some(session);
                    return Err(self.reject_state_v3());
                };
                let Some(next_epoch) = run.view.expected_store_epoch.checked_add(1) else {
                    run.session = Some(session);
                    return Err(self.reject_state_v3());
                };
                run.view.expected_generation_index = next_generation;
                run.view.expected_store_epoch = next_epoch;
                run.session = Some(session);
            }
            Err(rejected) => {
                let (error, source) = rejected.into_parts_v2();
                self.population_source = Some(source);
                self.state = ResidentSearchSlice2NativeStateV3::Poisoned;
                return Err(ResidentSearchSlice2NativeRejectedV3 {
                    error: ResidentSearchSlice2NativeErrorV3::Search(error.into()),
                    owner: self,
                });
            }
        }
        self.state = ResidentSearchSlice2NativeStateV3::Published;
        Ok(self)
    }

    pub(crate) fn enqueue_terminal_seal_v3(
        mut self,
    ) -> Result<Self, ResidentSearchSlice2NativeRejectedV3> {
        if self.state != ResidentSearchSlice2NativeStateV3::Published || self.pending.is_some() {
            return Err(self.reject_state_v3());
        }
        let mut pending = Box::new(RawResidentArchiveKnnPendingV2::default());
        let status = unsafe {
            enqueue_resident_archive_terminal_seal_v2(
                self.archive.expect("Slice2 owner retains archive").as_ptr(),
                pending.as_mut(),
            )
        };
        if status != STATUS_OK {
            return Err(self.reject_v3("enqueue_resident_archive_terminal_seal_v2", status));
        }
        self.pending = Some(pending);
        self.state = ResidentSearchSlice2NativeStateV3::TerminalPending;
        Ok(self)
    }

    pub(crate) fn try_complete_terminal_v3(
        mut self,
    ) -> Result<ResidentSearchSlice2NativeTryCompleteV3, ResidentSearchSlice2NativeRejectedV3> {
        if self.state != ResidentSearchSlice2NativeStateV3::TerminalPending {
            return Err(self.reject_state_v3());
        }
        let mut committed = Box::new(RawReadyEventV1::default());
        let mut terminal = RawResidentArchiveKnnTerminalV2::default();
        let status = unsafe {
            try_complete_resident_archive_terminal_v2(
                self.archive.expect("Slice2 owner retains archive").as_ptr(),
                self.pending
                    .as_deref()
                    .expect("pending owner retains stable receipt"),
                committed.as_mut(),
                &mut terminal,
            )
        };
        if status == STATUS_NOT_READY_V2 {
            return Ok(ResidentSearchSlice2NativeTryCompleteV3::NotReady(self));
        }
        if status != STATUS_OK {
            return Err(self.reject_v3("try_complete_resident_archive_terminal_v2", status));
        }
        let pending = self
            .pending
            .as_deref()
            .expect("terminal owner retains pending receipt");
        let binding = self
            .bind_authority
            .as_ref()
            .expect("Slice2 owner retains bind authority")
            .raw_v2();
        if !terminal.validates_committed_v2(pending, binding, committed.as_ref()) {
            return Err(self.reject_v3("validate_resident_archive_terminal_v2", -6));
        }
        let run = self.run.as_mut().expect("Slice2 owner retains Search run");
        run.ready_receipt_address = std::ptr::from_ref(committed.as_ref()) as usize;
        run.ready = Some(committed);
        self.terminal = Some(terminal);
        self.state = ResidentSearchSlice2NativeStateV3::TerminalComplete;
        Ok(ResidentSearchSlice2NativeTryCompleteV3::Complete(self))
    }

    pub(crate) fn release_terminal_v3(
        mut self,
    ) -> Result<PopulationSession, ResidentSearchSlice2NativeRejectedV3> {
        if self.state != ResidentSearchSlice2NativeStateV3::TerminalComplete {
            return Err(self.reject_state_v3());
        }
        let Some(run) = self.run.as_mut() else {
            return Err(self.reject_state_v3());
        };
        let session_handle = run
            .session
            .as_ref()
            .map(PopulationSession::resident_search_native_handle_v2);
        let Some(session_handle) = session_handle else {
            return Err(self.reject_state_v3());
        };
        let Some(archive) = self.archive else {
            return Err(self.reject_state_v3());
        };
        let status = unsafe {
            neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2(
                session_handle,
                archive.as_ptr(),
            )
        };
        if status != STATUS_OK {
            return Err(self.reject_v3(
                "neoethos_gpu_cuda_population_release_resident_archive_knn_owner_v2",
                status,
            ));
        }
        self.archive = None;
        self.pending = None;
        self.bind_authority = None;
        self.state = ResidentSearchSlice2NativeStateV3::Released;
        let run = self
            .run
            .take()
            .expect("validated Slice2 owner retains Search run");
        match run.close_preserving_v3() {
            Ok(session) => Ok(session),
            Err((error, run)) => {
                self.run = Some(run);
                self.state = ResidentSearchSlice2NativeStateV3::Poisoned;
                Err(ResidentSearchSlice2NativeRejectedV3 {
                    error: error.into(),
                    owner: self,
                })
            }
        }
    }
}

impl Drop for ResidentSearchSlice2NativeOwnerV3 {
    fn drop(&mut self) {
        if self.archive.is_some() {
            if let Some(source) = self.population_source.as_mut() {
                source.poison_without_reuse_v2();
            }
            if let Some(run) = self.run.as_mut() {
                run.state = ResidentSearchStateV2::Poisoned;
                if let Some(scoring) = run.scoring.as_mut() {
                    scoring.poison_v2();
                }
                if let Some(session) = run.session.as_mut() {
                    session.poison_resident_search_owner_v2();
                }
            }
        }
    }
}

pub struct ResidentSearchAdvancePendingV2 {
    run: Option<ResidentSearchRunV2>,
    completion: Option<ResidentSearchPopulationCompletionLeaseV2>,
    dependency: Option<Box<RawReadyEventV1>>,
    pending: Option<Box<RawResidentSearchAdvancePendingReceiptV2>>,
    #[cfg(feature = "cuda-device-fixtures")]
    population_counters: NeoPopulationCounters,
    consumed: bool,
}

pub enum ResidentSearchTryCompleteV2 {
    NotReady(ResidentSearchAdvancePendingV2),
    Complete(ResidentSearchRunV2),
}

impl ResidentSearchRunV2 {
    pub const fn state_v2(&self) -> ResidentSearchStateV2 {
        self.state
    }

    pub fn selected_device_ordinal_v2(&self) -> Option<i32> {
        self.session.as_ref().map(PopulationSession::device)
    }

    pub fn ready_receipt_address_is_stable_v2(&self) -> bool {
        self.ready.as_ref().is_some_and(|ready| {
            std::ptr::from_ref::<RawReadyEventV1>(ready.as_ref()) as usize
                == self.ready_receipt_address
        })
    }

    pub fn ready_event_summary_v2(&self) -> Option<ResidentReadyEventSummaryV2> {
        self.ready
            .as_ref()
            .map(|ready| ResidentReadyEventSummaryV2 {
                abi_version: ready.abi_version,
                reserved_is_zero: ready.reserved == 0,
                event_id: ready.event_id,
                generation_index: ready.generation_index,
                same_stream_enqueue_count: ready.same_stream_enqueue_count,
                intermediate_host_wait_count: ready.intermediate_host_wait_count,
                intermediate_readback_count: ready.intermediate_readback_count,
            })
    }

    pub fn current_gene_view_summary_v2(&self) -> ResidentGeneViewSummaryV2 {
        ResidentGeneViewSummaryV2 {
            abi_version: self.view.abi_version,
            has_device_seal: !self.view.seal_device.is_null() && self.view.flags == 0,
            generation_index: self.view.expected_generation_index,
            store_epoch: self.view.expected_store_epoch,
            run_token: self.view.expected_run_token,
            logical_population_count: self.view.logical_population_count,
            feature_count: self.view.feature_count,
            max_terms_per_gene: self.view.max_terms_per_gene,
            smc_flag_count: self.view.smc_flag_count,
            plan_identity_sha256: self.view.plan_identity_sha256,
            generation_semantics_sha256: self.view.generation_semantics_sha256,
        }
    }

    pub fn terminal_receipt_summary_v2(&self) -> Option<ResidentSearchTerminalReceiptV2> {
        self.terminal_receipt
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Queried only by the feature-gated device oracle.
    pub(crate) fn combined_admission_summary_fixture_v2(
        &self,
    ) -> Option<ResidentSearchCombinedAdmissionSummaryV2> {
        self.admission.as_ref().map(|sealed| {
            let raw = &sealed.raw;
            ResidentSearchCombinedAdmissionSummaryV2 {
                free_memory_snapshot_count: raw.free_memory_snapshot_count,
                generation_allocation_count: raw.generation_allocation_count,
                scoring_allocation_count: raw.scoring_allocation_count,
                terminal_host_allocation_count: raw.terminal_host_allocation_count,
                terminal_host_receipt_bytes: raw.terminal_host_receipt_bytes,
                same_context_free_bytes: raw.same_context_free_bytes,
                full_discovery_reserve_bytes: raw.full_discovery_reserve_bytes,
                total_device_bytes: raw.total_device_bytes,
                runtime_identity_is_exact: raw.runtime.abi_version == 2
                    && raw.runtime.run_admission_ordinal != 0
                    && raw.runtime.device_uuid != [0; 16]
                    && raw.runtime.primary_context_id != 0
                    && raw.runtime.run_stream_id != 0
                    && raw.runtime.active_pool_is_default == 1
                    && raw.runtime.run_stream_process_token != [0; 32],
                sealed_before_first_allocation: sealed.receipt_identity_sha256
                    == raw.receipt_identity_sha256
                    && raw.receipt_identity_sha256 != [0; 32],
            }
        })
    }

    pub fn refresh_current_gene_view_v2(
        &mut self,
    ) -> Result<ResidentGeneViewSummaryV2, ResidentSearchV2Error> {
        if !matches!(
            self.state,
            ResidentSearchStateV2::Active | ResidentSearchStateV2::AdvancedOnce
        ) {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        if !self.ready_receipt_address_is_stable_v2() {
            self.state = ResidentSearchStateV2::Poisoned;
            return Err(ResidentSearchV2Error::ReadyReceiptAddressChanged);
        }
        let generation = self
            .generation
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let ready = self
            .ready
            .as_ref()
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let mut view = RawResidentGenerationGeneViewV2::default();
        // SAFETY: the typed owner retains the native run and the boxed receipt.
        // The Box address is checked immediately above and native also requires
        // exact pointer identity before exporting the device-only seal view.
        let status = unsafe {
            export_current_resident_gene_view_v2(generation.as_ptr(), ready.as_ref(), &mut view)
        };
        if status != STATUS_OK {
            self.state = ResidentSearchStateV2::Poisoned;
            return Err(native_error("export_current_resident_gene_view_v2", status));
        }
        if view.abi_version != GENE_VIEW_ABI_V2
            || view.flags != 0
            || view.seal_device.is_null()
            || view.control_device.is_null()
            || view.expected_store_epoch == 0
            || view.expected_run_token == 0
            || view.logical_population_count != self.expected_population
            || view.feature_count != self.expected_feature_count
            || view.max_terms_per_gene != self.expected_max_terms
            || view.smc_flag_count != SMC_FLAG_COUNT_V2
            || view.plan_identity_sha256 != self.expected_plan_identity_sha256
            || view.generation_semantics_sha256 != self.expected_generation_semantics_sha256
        {
            self.state = ResidentSearchStateV2::Poisoned;
            return Err(ResidentSearchV2Error::GeneViewIdentityMismatch);
        }
        self.view = view;
        Ok(self.current_gene_view_summary_v2())
    }

    pub(crate) fn close_v2(mut self) -> Result<PopulationSession, ResidentSearchV2Error> {
        if !matches!(
            self.state,
            ResidentSearchStateV2::Active | ResidentSearchStateV2::AdvancedOnce
        ) {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        self.release_search_resources_v2()?;
        self.ready = None;
        self.state = ResidentSearchStateV2::Consumed;
        self.session
            .take()
            .ok_or(ResidentSearchV2Error::StateViolation)
    }

    fn close_preserving_v3(mut self) -> Result<PopulationSession, (ResidentSearchV2Error, Self)> {
        if !matches!(
            self.state,
            ResidentSearchStateV2::Active | ResidentSearchStateV2::AdvancedOnce
        ) {
            return Err((ResidentSearchV2Error::StateViolation, self));
        }
        if let Err(error) = self.release_search_resources_v2() {
            return Err((error, self));
        }
        self.ready = None;
        self.state = ResidentSearchStateV2::Consumed;
        match self.session.take() {
            Some(session) => Ok(session),
            None => Err((ResidentSearchV2Error::StateViolation, self)),
        }
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub fn close_fixture_v2(self) -> Result<PopulationSession, ResidentSearchV2Error> {
        self.close_v2()
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub fn combined_admission_device_bytes_fixture_v2(&self) -> Option<u64> {
        self.admission
            .as_ref()
            .map(|admission| admission.total_device_bytes)
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn set_scoring_metric_mode_fixture_v2(
        &mut self,
        mode: u32,
    ) -> Result<(), ResidentSearchV2Error> {
        if self.state != ResidentSearchStateV2::Active || mode > 3 {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        let scoring = self
            .scoring
            .as_ref()
            .and_then(ResidentScoringRunV2::native_v2)
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        // SAFETY: the fixture mutates only a host-side mode before scoring is
        // bound; the production binary does not contain this symbol.
        let status = unsafe { fixture_set_resident_scoring_metric_mode_v2(scoring.as_ptr(), mode) };
        if status != STATUS_OK {
            return Err(native_error(
                "fixture_set_resident_scoring_metric_mode_v2",
                status,
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn set_scoring_metric_fault_fixture_v2(
        &mut self,
        metric_slot: u32,
        nonfinite: f64,
    ) -> Result<(), ResidentSearchV2Error> {
        if self.state != ResidentSearchStateV2::Active || metric_slot >= 11 || nonfinite.is_finite()
        {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        let scoring = self
            .scoring
            .as_ref()
            .and_then(ResidentScoringRunV2::native_v2)
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let status = unsafe {
            fixture_set_resident_scoring_metric_fault_v2(
                scoring.as_ptr(),
                metric_slot,
                nonfinite.to_bits(),
            )
        };
        if status != STATUS_OK {
            return Err(native_error(
                "fixture_set_resident_scoring_metric_fault_v2",
                status,
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn set_gene_identity_fixture_v2(
        &mut self,
        candidate: u64,
        gene_identity: u64,
    ) -> Result<(), ResidentSearchV2Error> {
        if self.state != ResidentSearchStateV2::Active {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        let generation = self
            .generation
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        // SAFETY: native validates the candidate extent and appends the
        // fixture-only mutation to the same admitted stream.
        let status = unsafe {
            fixture_set_resident_generation_gene_identity_v2(
                generation.as_ptr(),
                candidate,
                gene_identity,
            )
        };
        if status != STATUS_OK {
            return Err(native_error(
                "fixture_set_resident_generation_gene_identity_v2",
                status,
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn set_duplicate_final_gene_content_fixture_v2(
        &mut self,
        source_candidate: u64,
        destination_candidate: u64,
    ) -> Result<(), ResidentSearchV2Error> {
        if self.state != ResidentSearchStateV2::Active {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        let generation = self
            .generation
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let status = unsafe {
            fixture_set_duplicate_final_gene_content_v2(
                generation.as_ptr(),
                source_candidate,
                destination_candidate,
            )
        };
        if status != STATUS_OK {
            return Err(native_error(
                "fixture_set_duplicate_final_gene_content_v2",
                status,
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn pending_drop_audit_fixture_v2()
    -> Result<ResidentSearchPendingDropAuditV2, ResidentSearchV2Error> {
        Ok(ResidentSearchPendingDropAuditV2 {
            poisoned_pending_drop_count: PENDING_DROP_POISON_COUNT_V2.load(Ordering::SeqCst),
            reused_in_flight_session_count: 0,
            terminal_fault_cleanup_count: TERMINAL_FAULT_CLEANUP_COUNT_V2.load(Ordering::SeqCst),
            terminal_session_destroy_count: terminal_search_session_destroy_count_fixture_v2(),
        })
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn terminal_fixture_snapshot_v2(
        &mut self,
    ) -> Result<ResidentSearchGenerationFixtureSnapshotV2, ResidentSearchV2Error> {
        if self.state != ResidentSearchStateV2::AdvancedOnce {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        let generation = self
            .generation
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let scoring = self
            .scoring
            .as_ref()
            .and_then(ResidentScoringRunV2::native_v2)
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let count = usize::try_from(self.expected_population)
            .map_err(|_| ResidentSearchV2Error::InvalidPlan("population does not fit usize"))?;
        let mut metric_rows = vec![NeoPopulationMetricRow::default(); count];
        let mut fitness_scores = vec![0.0; count];
        let mut decision_keys = vec![0; count];
        let mut ranked_population_ordinals = vec![0; count];
        if self.expected_max_terms != 3 {
            return Err(ResidentSearchV2Error::InvalidPlan(
                "the full fixture oracle requires exactly three fixed-stride terms",
            ));
        }
        let term_count = count.checked_mul(self.expected_max_terms as usize).ok_or(
            ResidentSearchV2Error::InvalidPlan("fixture term extent overflowed"),
        )?;
        let survivor_count = usize::try_from(self.expected_survivor_count)
            .map_err(|_| ResidentSearchV2Error::InvalidPlan("survivor count does not fit usize"))?;
        let mut initial_scalars = vec![RawResidentGenerationGeneScalarFixtureV2::default(); count];
        let mut final_scalars = vec![RawResidentGenerationGeneScalarFixtureV2::default(); count];
        let mut initial_term_indices = vec![0; term_count];
        let mut initial_term_weights = vec![0.0; term_count];
        let mut final_term_indices = vec![0; term_count];
        let mut final_term_weights = vec![0.0; term_count];
        let mut parent_a = vec![0; count];
        let mut parent_b = vec![0; count];
        let mut selected_survivors = vec![0; survivor_count];
        let mut sorted_dedup_flags = vec![0; count];
        let mut candidate_valid_flags = vec![0; count];
        let mut generation_snapshot = RawResidentGenerationAdvanceFixtureSnapshotV2::default();
        let mut scoring_snapshot = RawResidentScoringFixtureSnapshotV2::default();
        // SAFETY: both calls are terminal fixture-only D2H boundaries. Each
        // capacity exactly matches the retained owner allocation.
        let generation_status = unsafe {
            fixture_copy_resident_generation_advance_snapshot_v2(
                generation.as_ptr(),
                ranked_population_ordinals.as_mut_ptr(),
                initial_scalars.as_mut_ptr(),
                final_scalars.as_mut_ptr(),
                initial_term_indices.as_mut_ptr(),
                initial_term_weights.as_mut_ptr(),
                final_term_indices.as_mut_ptr(),
                final_term_weights.as_mut_ptr(),
                parent_a.as_mut_ptr(),
                parent_b.as_mut_ptr(),
                selected_survivors.as_mut_ptr(),
                sorted_dedup_flags.as_mut_ptr(),
                candidate_valid_flags.as_mut_ptr(),
                self.expected_population,
                term_count as u64,
                self.expected_survivor_count,
                &mut generation_snapshot,
            )
        };
        if generation_status != STATUS_OK {
            return Err(native_error(
                "fixture_copy_resident_generation_advance_snapshot_v2",
                generation_status,
            ));
        }
        let scoring_status = unsafe {
            fixture_copy_resident_scoring_snapshot_v2(
                scoring.as_ptr(),
                metric_rows.as_mut_ptr(),
                fitness_scores.as_mut_ptr(),
                decision_keys.as_mut_ptr(),
                self.expected_population,
                &mut scoring_snapshot,
            )
        };
        if scoring_status != STATUS_OK {
            return Err(native_error(
                "fixture_copy_resident_scoring_snapshot_v2",
                scoring_status,
            ));
        }
        if generation_snapshot.abi_version != 2
            || scoring_snapshot.abi_version != 2
            || scoring_snapshot.reserved != 0
            || generation_snapshot.logical_population_count != self.expected_population
            || generation_snapshot.max_terms_per_gene != self.expected_max_terms
            || u64::from(generation_snapshot.survivor_count) != self.expected_survivor_count
            || generation_snapshot.reserved != 0
            || scoring_snapshot.logical_population_count != self.expected_population
        {
            return Err(ResidentSearchV2Error::GeneViewIdentityMismatch);
        }
        let combine_genes = |scalars: &[RawResidentGenerationGeneScalarFixtureV2],
                             indices: &[u64],
                             weights: &[f64]| {
            scalars
                .iter()
                .enumerate()
                .map(|(candidate, scalar)| {
                    let base = candidate * 3;
                    ResidentSearchFixtureGeneV2 {
                        gene_identity: scalar.gene_identity,
                        content_hash: scalar.content_hash,
                        term_count: scalar.term_count,
                        smc_flags: scalar.smc_flags,
                        long_threshold: scalar.long_threshold,
                        short_threshold: scalar.short_threshold,
                        target_pips: scalar.target_pips,
                        stop_pips: scalar.stop_pips,
                        stop_vol_multiplier: scalar.stop_vol_multiplier,
                        generation: scalar.generation,
                        term_indices: [indices[base], indices[base + 1], indices[base + 2]],
                        term_weights: [weights[base], weights[base + 1], weights[base + 2]],
                    }
                })
                .collect::<Vec<_>>()
        };
        let initial_genes = combine_genes(
            &initial_scalars,
            &initial_term_indices,
            &initial_term_weights,
        );
        let final_genes = combine_genes(&final_scalars, &final_term_indices, &final_term_weights);
        Ok(ResidentSearchGenerationFixtureSnapshotV2 {
            metric_rows,
            fitness_scores,
            decision_keys,
            ranked_population_ordinals,
            initial_genes,
            final_genes,
            parent_a,
            parent_b,
            selected_survivors,
            sorted_dedup_flags,
            candidate_valid_flags,
            selected_count: generation_snapshot.selected_count,
            dedup_run_count: generation_snapshot.dedup_run_count,
            scoring_objective: scoring_snapshot.scoring_objective,
            scoring_device_fault: scoring_snapshot.device_fault_word,
            generation_device_fault: generation_snapshot.device_content_fault,
            gene_hash_collision_fault: generation_snapshot.gene_hash_collision_fault,
            control_fault_word: generation_snapshot.control_fault_word,
            stop_requested: generation_snapshot.stop_requested,
            current_store_index: generation_snapshot.current_store_index,
            generation_index: generation_snapshot.generation_index,
            store_epoch: generation_snapshot.store_epoch,
            terminal_synchronization_count: generation_snapshot
                .terminal_synchronization_count
                .saturating_add(scoring_snapshot.terminal_synchronization_count),
            terminal_readback_count: generation_snapshot
                .terminal_readback_count
                .saturating_add(scoring_snapshot.terminal_readback_count),
            terminal_readback_bytes: generation_snapshot
                .terminal_readback_bytes
                .saturating_add(scoring_snapshot.terminal_readback_bytes),
            population_counters: self
                .last_population_counters_fixture_v2
                .ok_or(ResidentSearchV2Error::StateViolation)?,
        })
    }

    #[allow(dead_code)] // The next Search chunk consumes this private enqueue seam.
    pub(crate) fn upload_resident_scenarios_v2(
        &mut self,
        scenarios: &[ScenarioDescriptor],
    ) -> Result<(), ResidentSearchV2Error> {
        if self.state != ResidentSearchStateV2::Active {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        let session = self
            .session
            .as_mut()
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        session.upload_resident_scenarios_v2(
            scenarios,
            self.view.logical_population_count,
            self.view.expected_generation_index,
            self.view.plan_identity_sha256,
        )?;
        Ok(())
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub(crate) fn enqueue_resident_gene_metrics_fixture_v2(
        &mut self,
        settings: &NeoPopulationSettings,
    ) -> Result<ResidentPopulationMetricsV1<'_>, ResidentSearchV2Error> {
        if self.state != ResidentSearchStateV2::Active
            || self.view.control_device.is_null()
            || self.view.seal_device.is_null()
        {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        self.session
            .as_mut()
            .ok_or(ResidentSearchV2Error::StateViolation)?
            .enqueue_resident_gene_metrics_fixture_v2(&self.view, settings)
            .map_err(Into::into)
    }

    #[allow(dead_code)] // Exercised by the bounded real-device oracle before promotion.
    pub(crate) fn advance_one_full_population_generation_v2(
        mut self,
        settings: &NeoPopulationSettings,
    ) -> Result<ResidentSearchAdvancePendingV2, ResidentSearchV2Error> {
        if self.state == ResidentSearchStateV2::AdvancedOnce {
            return Err(ResidentSearchV2Error::OneGenerationAdvanceAlreadyEnqueued);
        }
        if self.state != ResidentSearchStateV2::Active
            || self.view.control_device.is_null()
            || self.view.seal_device.is_null()
        {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        if !self.ready_receipt_address_is_stable_v2() {
            self.state = ResidentSearchStateV2::Poisoned;
            return Err(ResidentSearchV2Error::ReadyReceiptAddressChanged);
        }
        let generation = self
            .generation
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let scoring = self
            .scoring
            .as_ref()
            .and_then(ResidentScoringRunV2::native_v2)
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let view = self.view;
        let expected_population = self.expected_population;
        let retained_evaluation_capacity = self.retained_evaluation_capacity;
        let expected_feature_count = self.expected_feature_count;
        let expected_max_terms = self.expected_max_terms;
        let expected_full_discovery_reserve_bytes = self
            .admission
            .as_ref()
            .ok_or(ResidentSearchV2Error::StateViolation)?
            .full_discovery_reserve_bytes;
        self.state = ResidentSearchStateV2::Advancing;

        let session = self
            .session
            .take()
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let mut source = session.enqueue_resident_gene_metrics_owned_v2(
            &view,
            settings,
            expected_population,
            retained_evaluation_capacity,
            expected_feature_count,
            expected_max_terms,
            expected_full_discovery_reserve_bytes,
        )?;
        #[cfg(feature = "cuda-device-fixtures")]
        let population_counters = source.counters_fixture_v2();
        // Take, but retain, the exact Box whose receipt address native sealed.
        // A value copy would change pointer identity and fail before enqueue.
        let dependency = self
            .ready
            .take()
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let mut pending = Box::new(RawResidentSearchAdvancePendingReceiptV2::default());
        // SAFETY: all opaque owners and the source receipt belong to the same
        // admitted stream. Native consumes one full-population device chunk,
        // seals scoring, advances once and records the distinct boxed receipt.
        let status = unsafe {
            enqueue_full_population_scored_generation_advance_v2(
                generation.as_ptr(),
                scoring.as_ptr(),
                source.raw_source_v2(),
                dependency.as_ref(),
                pending.as_mut(),
            )
        };
        if status != STATUS_OK {
            source.poison_without_reuse_v2();
            if let Some(scoring) = self.scoring.as_mut() {
                scoring.poison_v2();
            }
            self.state = ResidentSearchStateV2::Poisoned;
            return Err(native_error(
                "enqueue_full_population_scored_generation_advance_v2",
                status,
            ));
        }
        if pending.abi_version != GENE_VIEW_ABI_V2
            || pending.reserved != 0
            || pending.completion_event_id == 0
            || pending.target_generation_index != 1
            || pending.target_store_epoch != 2
            || pending.target_store_index != 1
            || pending.run_token != view.expected_run_token
            || pending.same_stream_enqueue_count <= dependency.same_stream_enqueue_count
            || !std::ptr::eq(pending.dependency_receipt_token, dependency.as_ref())
            || pending.terminal_host_receipt_token.is_null()
        {
            source.poison_without_reuse_v2();
            if let Some(scoring) = self.scoring.as_mut() {
                scoring.poison_v2();
            }
            self.state = ResidentSearchStateV2::Poisoned;
            return Err(ResidentSearchV2Error::GeneViewIdentityMismatch);
        }
        self.state = ResidentSearchStateV2::AdvancePending;
        Ok(ResidentSearchAdvancePendingV2 {
            run: Some(self),
            completion: Some(source),
            dependency: Some(dependency),
            pending: Some(pending),
            #[cfg(feature = "cuda-device-fixtures")]
            population_counters,
            consumed: false,
        })
    }

    fn release_search_resources_v2(&mut self) -> Result<(), ResidentSearchV2Error> {
        if self.state == ResidentSearchStateV2::Poisoned {
            // Error/drop leaves potentially reachable stream-ordered storage
            // owned by native. It must not be freed or reused without terminal
            // proof, so the poisoned owner intentionally leaks fail-closed.
            return Ok(());
        }
        if let Some(scoring) = self.scoring.as_mut() {
            if let Err(error) = scoring.release_v2() {
                self.state = ResidentSearchStateV2::Poisoned;
                if let Some(session) = self.session.as_mut() {
                    session.poison_resident_search_owner_v2();
                }
                return Err(scoring_error(error));
            }
        }
        self.scoring = None;
        let Some(generation) = self.generation else {
            return Ok(());
        };
        let session = self
            .session
            .as_mut()
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        // SAFETY: both opaque handles originate from the same native admission;
        // the generation run is released at most once and its boxed receipt is
        // kept alive until this call returns.
        let status = unsafe {
            neoethos_gpu_cuda_population_release_resident_generation_run_v2(
                session.resident_search_native_handle_v2(),
                generation.as_ptr(),
            )
        };
        if status != STATUS_OK {
            self.state = ResidentSearchStateV2::Poisoned;
            session.poison_resident_search_owner_v2();
            return Err(native_error(
                "neoethos_gpu_cuda_population_release_resident_generation_run_v2",
                status,
            ));
        }
        self.generation = None;
        Ok(())
    }

    fn release_terminal_proven_fault_resources_v2(&mut self) -> Result<(), ResidentSearchV2Error> {
        if self.state != ResidentSearchStateV2::Poisoned
            || self.terminal_receipt.as_ref().is_none_or(|receipt| {
                receipt.raw.terminal_status != 2
                    || receipt.raw.generation_index != 0
                    || receipt.raw.store_epoch != 1
                    || receipt.raw.current_store_index != 0
            })
        {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        // The exact terminal event is Ready. Release the scoring arena first;
        // if it fails, retain the generation owner as native proof that makes
        // the population session's Drop path leak-only.
        if let Some(scoring) = self.scoring.as_mut() {
            scoring.release_v2().map_err(scoring_error)?;
        }
        self.scoring = None;
        let generation = self
            .generation
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let session = self
            .session
            .as_mut()
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        // SAFETY: native recorded terminal_event_proven_v2 before returning the
        // device-fault status, and the population completion lease is StrictIdle.
        let status = unsafe {
            neoethos_gpu_cuda_population_release_resident_generation_run_v2(
                session.resident_search_native_handle_v2(),
                generation.as_ptr(),
            )
        };
        if status != STATUS_OK {
            session.poison_resident_search_owner_v2();
            return Err(native_error(
                "neoethos_gpu_cuda_population_release_resident_generation_run_v2",
                status,
            ));
        }
        self.generation = None;
        self.admission = None;
        Ok(())
    }
}

impl ResidentSearchAdvancePendingV2 {
    pub const fn state_v2(&self) -> ResidentSearchStateV2 {
        ResidentSearchStateV2::AdvancePending
    }

    pub fn committed_gene_view_summary_v2(&self) -> ResidentGeneViewSummaryV2 {
        self.run
            .as_ref()
            .expect("pending owner retains its Search run")
            .current_gene_view_summary_v2()
    }

    pub fn try_complete_one_generation_v2(
        mut self,
    ) -> Result<ResidentSearchTryCompleteV2, ResidentSearchV2Error> {
        let run = self
            .run
            .as_mut()
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        if run.state != ResidentSearchStateV2::AdvancePending {
            return Err(ResidentSearchV2Error::StateViolation);
        }
        let generation = run
            .generation
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let pending = self
            .pending
            .as_ref()
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        let mut committed = Box::new(RawReadyEventV1::default());
        let mut terminal = RawResidentSearchTerminalReceiptV2::default();
        let status = unsafe {
            try_complete_resident_generation_advance_v2(
                generation.as_ptr(),
                pending.as_ref(),
                committed.as_mut(),
                &mut terminal,
            )
        };
        if status == STATUS_NOT_READY_V2 {
            return Ok(ResidentSearchTryCompleteV2::NotReady(self));
        }
        if status == STATUS_DEVICE_FAULT_V2 {
            let receipt = ResidentSearchTerminalReceiptV2 { raw: terminal };
            run.state = ResidentSearchStateV2::Poisoned;
            run.terminal_receipt = Some(receipt);
            let cleanup = (|| -> Result<(), ResidentSearchV2Error> {
                let session = self
                    .completion
                    .take()
                    .ok_or(ResidentSearchV2Error::StateViolation)?
                    .finish_device_consume_v2()?;
                run.session = Some(session);
                run.release_terminal_proven_fault_resources_v2()?;
                run.session
                    .take()
                    .ok_or(ResidentSearchV2Error::StateViolation)?
                    .destroy_terminal_proven_resident_search_v2()?;
                Ok(())
            })();
            self.dependency = None;
            self.pending = None;
            self.consumed = true;
            if let Err(error) = cleanup {
                return Err(ResidentSearchV2Error::DeviceTerminalFaultCleanup {
                    receipt,
                    reason: error.to_string(),
                });
            }
            #[cfg(feature = "cuda-device-fixtures")]
            TERMINAL_FAULT_CLEANUP_COUNT_V2.fetch_add(1, Ordering::SeqCst);
            return Err(ResidentSearchV2Error::DeviceTerminalFault(receipt));
        }
        if status != STATUS_OK {
            if let Some(completion) = self.completion.as_mut() {
                completion.poison_without_reuse_v2();
            }
            if let Some(scoring) = run.scoring.as_mut() {
                scoring.poison_v2();
            }
            run.state = ResidentSearchStateV2::Poisoned;
            self.consumed = true;
            return Err(native_error(
                "try_complete_resident_generation_advance_v2",
                status,
            ));
        }
        if validate_ready_event(committed.as_ref()).is_err()
            || terminal.abi_version != GENE_VIEW_ABI_V2
            || terminal.terminal_status != 1
            || terminal.generation_index != 1
            || terminal.store_epoch != 2
            || terminal.current_store_index != 1
            || terminal.scoring_device_fault != 0
            || terminal.generation_device_fault != 0
            || terminal.control_fault_word != 0
            || terminal.stop_requested != 0
            || terminal.compact_async_d2h_count != 1
            || terminal.compact_async_d2h_bytes
                != std::mem::size_of::<RawResidentSearchTerminalReceiptV2>() as u64
            || terminal.completion_stream_synchronize_count != 0
            || terminal.completion_event_query_count == 0
        {
            if let Some(completion) = self.completion.as_mut() {
                completion.poison_without_reuse_v2();
            }
            if let Some(scoring) = run.scoring.as_mut() {
                scoring.poison_v2();
            }
            run.state = ResidentSearchStateV2::Poisoned;
            self.consumed = true;
            return Err(ResidentSearchV2Error::GeneViewIdentityMismatch);
        }
        let session = self
            .completion
            .take()
            .ok_or(ResidentSearchV2Error::StateViolation)?
            .finish_device_consume_v2()?;
        run.session = Some(session);
        run.ready_receipt_address = std::ptr::from_ref(committed.as_ref()) as usize;
        run.ready = Some(committed);
        run.terminal_receipt = Some(ResidentSearchTerminalReceiptV2 { raw: terminal });
        run.scoring
            .as_mut()
            .ok_or(ResidentSearchV2Error::StateViolation)?
            .mark_bound_v2();
        run.state = ResidentSearchStateV2::AdvancedOnce;
        #[cfg(feature = "cuda-device-fixtures")]
        {
            run.last_population_counters_fixture_v2 = Some(self.population_counters);
        }
        run.refresh_current_gene_view_v2()?;
        self.dependency = None;
        self.pending = None;
        self.consumed = true;
        Ok(ResidentSearchTryCompleteV2::Complete(
            self.run
                .take()
                .ok_or(ResidentSearchV2Error::StateViolation)?,
        ))
    }
}

impl Drop for ResidentSearchAdvancePendingV2 {
    fn drop(&mut self) {
        if self.consumed {
            return;
        }
        if let Some(completion) = self.completion.as_mut() {
            completion.poison_without_reuse_v2();
        }
        if let Some(run) = self.run.as_mut() {
            if let Some(scoring) = run.scoring.as_mut() {
                scoring.poison_v2();
            }
            run.state = ResidentSearchStateV2::Poisoned;
        }
        #[cfg(feature = "cuda-device-fixtures")]
        PENDING_DROP_POISON_COUNT_V2.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for ResidentSearchRunV2 {
    fn drop(&mut self) {
        if matches!(
            self.state,
            ResidentSearchStateV2::Advancing | ResidentSearchStateV2::AdvancePending
        ) {
            self.state = ResidentSearchStateV2::Poisoned;
            if let Some(session) = self.session.as_mut() {
                session.poison_resident_search_owner_v2();
            }
        }
        if self.generation.is_some() || self.scoring.is_some() {
            let _ = self.release_search_resources_v2();
        }
    }
}

impl PopulationSession {
    /// Production stays unavailable until V2 addressed RNG and exact current
    /// GA semantics have CPU-oracle/device parity, the strict evaluator can
    /// advance without a host readback, its control lives in the run arena and
    /// scenario identity/extent are frozen after one admission upload.
    pub fn begin_resident_search_v2(self) -> Result<ResidentSearchRunV2, ResidentSearchV2Error> {
        debug_assert!(!resident_search_v2_production_readiness().production_ready());
        drop(self);
        Err(ResidentSearchV2Error::ResidentGenerationSemanticsNotProductionReady)
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub fn begin_resident_search_fixture_v2(
        self,
        plan: ResidentSearchFixturePlanV2,
        smc_weights: [f64; 11],
        smc_gate_disabled: bool,
    ) -> Result<ResidentSearchRunV2, ResidentSearchV2Error> {
        self.begin_resident_search_sealed_v2(
            plan.sealed,
            smc_weights,
            smc_gate_disabled,
            ResidentScoringObjectiveV2::PropFirmV4,
            0.0,
        )
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn begin_resident_search_scoring_fixture_v2(
        self,
        plan: ResidentSearchFixturePlanV2,
        smc_weights: [f64; 11],
        smc_gate_disabled: bool,
        objective: ResidentScoringObjectiveV2,
        novelty_weight: f64,
    ) -> Result<ResidentSearchRunV2, ResidentSearchV2Error> {
        self.begin_resident_search_sealed_v2(
            plan.sealed,
            smc_weights,
            smc_gate_disabled,
            objective,
            novelty_weight,
        )
    }

    #[allow(dead_code)] // Called only by the crate-private V3 ownership bridge for now.
    pub(crate) fn begin_resident_search_from_plan_v2(
        self,
        plan: SealedResidentGenerationPlanV1,
        smc_weights: [f64; 11],
        smc_gate_disabled: bool,
    ) -> Result<ResidentSearchRunV2, ResidentSearchV2Error> {
        self.begin_resident_search_sealed_v2(
            plan,
            smc_weights,
            smc_gate_disabled,
            ResidentScoringObjectiveV2::PropFirmV4,
            0.0,
        )
    }

    #[allow(dead_code)] // Called by the typed V3 construction bridge in the next wiring slice.
    pub(crate) fn begin_resident_search_slice2_native_v3(
        self,
        plan: SealedResidentGenerationPlanV1,
        smc_weights: [f64; 11],
        smc_gate_disabled: bool,
        bind_authority: ResidentSearchSlice2NativeBindAuthorityV2,
    ) -> Result<ResidentSearchSlice2NativeOwnerV3, ResidentSearchSlice2NativeErrorV3> {
        let run = self.begin_resident_search_sealed_impl_v3(
            plan,
            smc_weights,
            smc_gate_disabled,
            ResidentScoringObjectiveV2::PropFirmV4,
            0.0,
            Some(bind_authority.raw_v2()),
        )?;
        ResidentSearchSlice2NativeOwnerV3::bind_v3(run, bind_authority)
    }

    #[allow(dead_code)] // Shared by the private bridge and device-only fixture.
    fn begin_resident_search_sealed_v2(
        self,
        plan: SealedResidentGenerationPlanV1,
        smc_weights: [f64; 11],
        smc_gate_disabled: bool,
        scoring_objective: ResidentScoringObjectiveV2,
        novelty_weight: f64,
    ) -> Result<ResidentSearchRunV2, ResidentSearchV2Error> {
        self.begin_resident_search_sealed_impl_v3(
            plan,
            smc_weights,
            smc_gate_disabled,
            scoring_objective,
            novelty_weight,
            None,
        )
    }

    fn begin_resident_search_sealed_impl_v3(
        mut self,
        plan: SealedResidentGenerationPlanV1,
        smc_weights: [f64; 11],
        smc_gate_disabled: bool,
        scoring_objective: ResidentScoringObjectiveV2,
        novelty_weight: f64,
        slice2_binding: Option<&RawResidentArchiveKnnBindV2>,
    ) -> Result<ResidentSearchRunV2, ResidentSearchV2Error> {
        // No Search work exists yet. Authorize ordinary destruction before any
        // fallible validation/admission step so a clean start failure cannot
        // strand the V3 population session's original leak-only policy.
        self.authorize_resident_session_destroy_v3();
        if smc_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || smc_weights.iter().all(|weight| *weight == 0.0)
        {
            return Err(ResidentSearchV2Error::InvalidPlan(
                "SMC weights must be finite, nonnegative and not all zero",
            ));
        }
        let expected_feature_count = plan.feature_count_v1();
        let expected_population = plan.logical_population_count_v1();
        let expected_max_terms = plan.max_terms_per_gene_v1();
        #[cfg(feature = "cuda-device-fixtures")]
        let expected_survivor_count = plan.survivor_count_v1();
        let retained_evaluation_capacity = plan.retained_evaluation_capacity_v1();
        let expected_plan_identity_sha256 = plan.plan_identity_sha256_v1();
        let expected_generation_semantics_sha256 = plan.generation_semantics_sha256_v1();
        let session_handle = self
            .admit_resident_search_owner_v2(usize::try_from(expected_feature_count).map_err(
                |_| ResidentSearchV2Error::InvalidPlan("feature count does not fit host usize"),
            )?)
            .map_err(ResidentSearchV2Error::InvalidAdmission)?;
        let mut runtime = RawResidentSearchRuntimeFactsV2::default();
        let status = unsafe {
            neoethos_gpu_cuda_population_reserve_resident_search_runtime_v2(
                session_handle,
                &mut runtime,
            )
        };
        if status != STATUS_OK {
            return Err(native_error(
                "neoethos_gpu_cuda_population_reserve_resident_search_runtime_v2",
                status,
            ));
        }
        let scoring_plan =
            seal_resident_scoring_plan_v2(&plan, scoring_objective, novelty_weight, &runtime)
                .map_err(scoring_error)?;
        let mut raw_admission = RawResidentSearchCombinedAdmissionV2::default();
        let status = unsafe {
            match slice2_binding {
                Some(binding) => neoethos_gpu_cuda_population_query_resident_search_slice2_v3(
                    session_handle,
                    plan.raw_plan_v1(),
                    scoring_plan.raw_v2(),
                    &runtime,
                    binding,
                    &mut raw_admission,
                ),
                None => neoethos_gpu_cuda_population_query_resident_search_combined_v2(
                    session_handle,
                    plan.raw_plan_v1(),
                    scoring_plan.raw_v2(),
                    &runtime,
                    &mut raw_admission,
                ),
            }
        };
        if status != STATUS_OK {
            return Err(native_error(
                "neoethos_gpu_cuda_population_query_resident_search_combined_v2",
                status,
            ));
        }
        let admission = seal_combined_search_admission_v2(raw_admission).map_err(scoring_error)?;
        let mut generation = std::ptr::null_mut();
        let mut scoring_native = std::ptr::null_mut();
        // SAFETY: native validates the sealed combined receipt and the exact
        // runtime facts before allocating either device arena.
        let status = unsafe {
            match slice2_binding {
                Some(binding) => neoethos_gpu_cuda_population_create_resident_search_slice2_v3(
                    session_handle,
                    plan.raw_plan_v1(),
                    scoring_plan.raw_v2(),
                    &admission.raw,
                    binding,
                    &mut generation,
                    &mut scoring_native,
                ),
                None => neoethos_gpu_cuda_population_create_resident_search_combined_v2(
                    session_handle,
                    plan.raw_plan_v1(),
                    scoring_plan.raw_v2(),
                    &admission.raw,
                    &mut generation,
                    &mut scoring_native,
                ),
            }
        };
        if status != STATUS_OK {
            return Err(native_error(
                "neoethos_gpu_cuda_population_create_resident_search_combined_v2",
                status,
            ));
        }
        // A successful combined create published both native arenas into the
        // session. Re-arm leak-only ownership before validating returned
        // handles or enqueuing initialization, so every later error retains the
        // complete owner graph until terminal proof.
        self.arm_resident_session_leak_only_v3();
        let generation = NonNull::new(generation).ok_or(ResidentSearchV2Error::Native {
            operation: "neoethos_gpu_cuda_population_create_resident_search_combined_v2",
            status,
        })?;
        let scoring = ResidentScoringRunV2::from_combined_v2(session_handle, scoring_native)
            .map_err(scoring_error)?;
        let mut owner = ResidentSearchRunV2 {
            session: Some(self),
            generation: Some(generation),
            scoring: Some(scoring),
            admission: Some(admission),
            ready: None,
            ready_receipt_address: 0,
            view: RawResidentGenerationGeneViewV2::default(),
            expected_population,
            expected_feature_count,
            expected_max_terms,
            #[cfg(feature = "cuda-device-fixtures")]
            expected_survivor_count,
            retained_evaluation_capacity,
            expected_plan_identity_sha256,
            expected_generation_semantics_sha256,
            state: ResidentSearchStateV2::Active,
            terminal_receipt: None,
            #[cfg(feature = "cuda-device-fixtures")]
            last_population_counters_fixture_v2: None,
        };
        if let Err(error) = validate_allocation_receipt(
            &owner
                .admission
                .as_ref()
                .ok_or(ResidentSearchV2Error::StateViolation)?
                .raw
                .generation,
            &plan,
        ) {
            owner.state = ResidentSearchStateV2::Poisoned;
            return Err(error);
        }

        let mut initialized = Box::new(RawReadyEventV1::default());
        // SAFETY: `initialized` is boxed before FFI. Its address therefore
        // remains stable when the Rust owner and wrapper are subsequently moved.
        let status = unsafe {
            ffi_initialize_resident_generation_population_v1(
                generation.as_ptr(),
                initialized.as_mut(),
            )
        };
        if status != STATUS_OK {
            owner.state = ResidentSearchStateV2::Poisoned;
            return Err(native_error(
                "initialize_resident_generation_population_v1",
                status,
            ));
        }
        owner.ready_receipt_address = std::ptr::from_ref(initialized.as_ref()) as usize;
        owner.ready = Some(initialized);
        if let Err(error) = validate_ready_event(
            owner
                .ready
                .as_deref()
                .ok_or(ResidentSearchV2Error::StateViolation)?,
        ) {
            owner.state = ResidentSearchStateV2::Poisoned;
            return Err(error);
        }

        let mut configured = Box::new(RawReadyEventV1::default());
        let dependency = owner
            .ready
            .as_ref()
            .ok_or(ResidentSearchV2Error::StateViolation)?;
        // SAFETY: dependency and output are distinct stable Boxes. Native copies
        // the 11 immutable SMC weights once on the admitted stream and records a
        // new exact receipt pointer for subsequent gene-view exports.
        let status = unsafe {
            configure_resident_generation_evaluator_v2(
                generation.as_ptr(),
                dependency.as_ref(),
                smc_weights.as_ptr(),
                u32::from(smc_gate_disabled),
                configured.as_mut(),
            )
        };
        if status != STATUS_OK {
            owner.state = ResidentSearchStateV2::Poisoned;
            return Err(native_error(
                "configure_resident_generation_evaluator_v2",
                status,
            ));
        }
        owner.ready_receipt_address = std::ptr::from_ref(configured.as_ref()) as usize;
        owner.ready = Some(configured);
        if let Err(error) = validate_ready_event(
            owner
                .ready
                .as_deref()
                .ok_or(ResidentSearchV2Error::StateViolation)?,
        ) {
            owner.state = ResidentSearchStateV2::Poisoned;
            return Err(error);
        }
        owner.refresh_current_gene_view_v2()?;
        Ok(owner)
    }
}

#[cfg(feature = "cuda-device-fixtures")]
pub struct ResidentSearchFixturePlanV2 {
    sealed: SealedResidentGenerationPlanV1,
}

#[cfg(feature = "cuda-device-fixtures")]
impl ResidentSearchFixturePlanV2 {
    pub fn new(
        logical_population_count: usize,
        feature_count: usize,
    ) -> Result<Self, ResidentSearchV2Error> {
        if logical_population_count == 0
            || logical_population_count > i32::MAX as usize
            || feature_count == 0
            || feature_count > u32::MAX as usize
        {
            return Err(ResidentSearchV2Error::InvalidPlan(
                "population/features are outside the native V2 ABI",
            ));
        }
        Ok(Self {
            sealed: SealedResidentGenerationPlanV1::resident_search_fixture_v2(
                logical_population_count,
                feature_count,
            ),
        })
    }

    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn new_with_scoring_objective_v2(
        logical_population_count: usize,
        feature_count: usize,
        objective: ResidentScoringObjectiveV2,
    ) -> Result<Self, ResidentSearchV2Error> {
        if logical_population_count == 0
            || logical_population_count > i32::MAX as usize
            || feature_count == 0
            || feature_count > u32::MAX as usize
        {
            return Err(ResidentSearchV2Error::InvalidPlan(
                "population/features are outside the native V2 ABI",
            ));
        }
        Ok(Self {
            sealed: SealedResidentGenerationPlanV1::resident_search_scoring_fixture_v2(
                logical_population_count,
                feature_count,
                objective,
            ),
        })
    }

    pub const fn feature_count(&self) -> usize {
        self.sealed.feature_count_v1() as usize
    }

    pub const fn logical_population_count(&self) -> usize {
        self.sealed.logical_population_count_v1() as usize
    }

    pub const fn production_eligible(&self) -> bool {
        false
    }
}

#[allow(dead_code)] // Validation for the crate-private V3 -> Search consumer.
fn validate_ready_event(ready: &RawReadyEventV1) -> Result<(), ResidentSearchV2Error> {
    if ready.abi_version != GENERATION_ABI_V1
        || ready.reserved != 0
        || ready.event_id == 0
        || ready.intermediate_host_wait_count != 0
        || ready.intermediate_readback_count != 0
    {
        return Err(ResidentSearchV2Error::GeneViewIdentityMismatch);
    }
    Ok(())
}

#[allow(dead_code)] // Validation for the crate-private V3 -> Search consumer.
fn validate_allocation_receipt(
    allocation: &RawAllocationReceiptV1,
    plan: &SealedResidentGenerationPlanV1,
) -> Result<(), ResidentSearchV2Error> {
    let charged = [
        allocation.logical_gene_scalar_bytes,
        allocation.logical_gene_index_bytes,
        allocation.logical_gene_weight_bytes,
        allocation.offspring_bytes,
        allocation.metric_row_bytes,
        allocation.rank_key_bytes,
        allocation.selection_bytes,
        allocation.dedup_hash_bytes,
        allocation.cub_scratch_bytes,
        allocation.retained_evaluation_workspace_bytes,
        allocation.terminal_device_receipt_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, u64::checked_add)
    .ok_or(ResidentSearchV2Error::InvalidPlan(
        "generation allocation receipt overflowed",
    ))?;
    let available_after_reserve = allocation
        .same_context_free_bytes
        .checked_sub(allocation.full_discovery_reserve_bytes)
        .ok_or(ResidentSearchV2Error::InvalidPlan(
            "generation reserve exceeds same-context free bytes",
        ))?;
    let expected_generation_chunks = plan.logical_population_count_v1()
        / plan.retained_evaluation_capacity_v1()
        + u64::from(
            plan.logical_population_count_v1() % plan.retained_evaluation_capacity_v1() != 0,
        );
    if allocation.abi_version != GENERATION_ABI_V1
        || allocation.generation_store_allocation_count != 1
        || allocation.logical_population_count != plan.logical_population_count_v1()
        || allocation.retained_evaluation_capacity != plan.retained_evaluation_capacity_v1()
        || allocation.generation_chunk_count != expected_generation_chunks
        || allocation.allocation_plan_sha256 != plan.plan_identity_sha256_v1()
        || allocation.total_device_bytes != charged
        || allocation.total_device_bytes > available_after_reserve
    {
        return Err(ResidentSearchV2Error::InvalidPlan(
            "generation allocation receipt differs from the exact sealed plan",
        ));
    }
    Ok(())
}

fn native_error(operation: &'static str, status: i32) -> ResidentSearchV2Error {
    if matches!(
        status,
        crate::population::STATUS_ASYNC_FREE_OUTCOME_UNKNOWN
            | crate::population::STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN
    ) {
        return ResidentSearchV2Error::Population(CudaPopulationError::native(operation, status));
    }
    ResidentSearchV2Error::Native { operation, status }
}

fn scoring_error(error: ResidentScoringV2Error) -> ResidentSearchV2Error {
    match error {
        ResidentScoringV2Error::InvalidNoveltyWeight => ResidentSearchV2Error::InvalidPlan(
            "novelty weight must have the exact +0.0 bit pattern",
        ),
        ResidentScoringV2Error::InvalidPlan(reason) => ResidentSearchV2Error::InvalidPlan(reason),
        ResidentScoringV2Error::ArithmeticOverflow => {
            ResidentSearchV2Error::InvalidPlan("scoring admission arithmetic overflowed")
        }
        ResidentScoringV2Error::AsyncFreeOutcomeUnknownDeliberateLeak { operation } => {
            ResidentSearchV2Error::Population(
                CudaPopulationError::AsyncFreeOutcomeUnknownDeliberateLeak { operation },
            )
        }
        ResidentScoringV2Error::AsyncAllocationOutcomeUnknownDeliberateLeak { operation } => {
            ResidentSearchV2Error::Population(
                CudaPopulationError::AsyncAllocationOutcomeUnknownDeliberateLeak { operation },
            )
        }
        ResidentScoringV2Error::Native { operation, status } => {
            ResidentSearchV2Error::Native { operation, status }
        }
    }
}
