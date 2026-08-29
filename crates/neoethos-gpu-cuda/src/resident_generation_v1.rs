//! Opaque, stream-ordered native CUDA generation ownership.
//!
//! This module deliberately has no CPU implementation. When CUDA support is
//! absent, acquisition fails before a run or store can be minted. The module is
//! additive until the existing population-session owner can move its private
//! native session into `ResidentGenerationPopulationSessionImportV1`.

use sha2::{Digest, Sha256};
use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;

pub const DISCOVERY_GENERATION_SEMANTICS_V1: &str = concat!(
    "neoethos.discovery-generation.v1;",
    "fixed-stride-normalized-gene;",
    "philox4x32-10-address-v1;",
    "decision-slot-high32-retry-low32;",
    "rank-weighted-parent-and-survivor-only;",
    "fixed-original-rank-integer-weights;",
    "sealed-u64-scoring-novelty-decision-key;",
    "metric-row-identity-only;",
    "stable-cub-radix-u64-key-and-gene-identity-tie;",
    "cuda-cccl-toolkit-native-build-bound;",
    "same-admitted-stream;no-floating-decision-reduction;",
    "resident-global-full-gene-dedup;fnv4-resident-content;",
    "no-candidate-revival;no-host-decision"
);

const ABI_VERSION_V1: u32 = 1;
const PARENT_RANK_WEIGHTED_V1: u32 = 1;
const SURVIVOR_RANK_WEIGHTED_V1: u32 = 1;
const STATUS_OK_V1: i32 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ParentSelectionPolicyV1 {
    RankWeighted = 1,
    Uniform = 2,
    Tournament = 3,
    Softmax = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum SurvivorSelectionPolicyV1 {
    RankWeighted = 1,
    Elitist = 2,
    Tournament = 3,
    Generational = 4,
}

#[derive(Debug)]
pub enum ResidentGenerationDeviceErrorV1 {
    CudaFeatureNotCompiled,
    UnsupportedUniformSelection,
    UnsupportedTournamentSelection,
    UnsupportedSoftmaxSelection,
    UnsupportedElitistSelection,
    UnsupportedGenerationalSelection,
    InvalidPlan(&'static str),
    IdentityMismatch(&'static str),
    ArithmeticOverflow,
    CapacityUnavailable,
    Native {
        operation: &'static str,
        status: i32,
    },
    RunStateViolation,
    EventIdentityMismatch,
    DeviceContentFault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentGenerationRunStateV1 {
    StrictIdle,
    InFlight,
    Sealed,
    PostGaInPlace,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationArtifactClassV1 {
    ResearchOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum GeneticOperatorIdentityV1 {
    InitializeTermCount = 1,
    InitializeIndicator = 2,
    InitializeWeightLevel = 3,
    InitializeWeightSign = 4,
    InitializeThreshold = 5,
    InitializeStopGeometry = 6,
    InitializeSmcFlag = 7,
    ParentA = 8,
    ParentB = 9,
    CrossoverScalar = 10,
    MutationKind = 11,
    MutationValue = 12,
    MutationSmc = 13,
    Survivor = 14,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhiloxDrawAddressV1 {
    counter: [u32; 4],
    key: [u32; 2],
}

impl PhiloxDrawAddressV1 {
    pub const fn counter(&self) -> [u32; 4] {
        self.counter
    }

    pub const fn key(&self) -> [u32; 2] {
        self.key
    }
}

pub fn checked_philox_counter_mapping_v1(
    search_seed: u64,
    run_identity_sha256: &[u8; 32],
    generation_index: usize,
    candidate_identity: u64,
    genetic_operator_identity: GeneticOperatorIdentityV1,
    draw_index: u64,
) -> Result<PhiloxDrawAddressV1, ResidentGenerationDeviceErrorV1> {
    let generation_index = u32::try_from(generation_index)
        .map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?;
    let run_word_0 =
        u32::from_le_bytes(run_identity_sha256[0..4].try_into().map_err(|_| {
            ResidentGenerationDeviceErrorV1::IdentityMismatch("run identity word 0")
        })?);
    let run_word_1 =
        u32::from_le_bytes(run_identity_sha256[4..8].try_into().map_err(|_| {
            ResidentGenerationDeviceErrorV1::IdentityMismatch("run identity word 1")
        })?);
    Ok(PhiloxDrawAddressV1 {
        counter: [
            candidate_identity as u32,
            (candidate_identity >> 32) as u32,
            generation_index,
            draw_index as u32,
        ],
        key: [
            search_seed as u32 ^ run_word_0 ^ genetic_operator_identity as u32,
            (search_seed >> 32) as u32 ^ run_word_1 ^ (draw_index >> 32) as u32,
        ],
    })
}

pub fn checked_philox_rejection_draw_index_v1(decision_slot: u32, rejection_attempt: u32) -> u64 {
    (u64::from(decision_slot) << 32) | u64::from(rejection_attempt)
}

pub fn philox4x32_10_reference_v1(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    for _ in 0..10 {
        let product_0 = (M0 as u64) * (counter[0] as u64);
        let product_1 = (M1 as u64) * (counter[2] as u64);
        counter = [
            (product_1 >> 32) as u32 ^ counter[1] ^ key[0],
            product_1 as u32,
            (product_0 >> 32) as u32 ^ counter[3] ^ key[1],
            product_0 as u32,
        ];
        key[0] = key[0].wrapping_add(W0);
        key[1] = key[1].wrapping_add(W1);
    }
    counter
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawPopulationSessionImportV1 {
    pub(crate) abi_version: u32,
    pub(crate) selected_cuda_ordinal: u32,
    pub(crate) admitted_run_stream: *mut c_void,
    pub(crate) resident_parent_ready_event: *mut c_void,
    pub(crate) generation_ready_event: *mut c_void,
    pub(crate) population_lifetime_owner: *mut c_void,
    pub(crate) full_discovery_reserve_bytes: u64,
    pub(crate) cuda_device_identity_sha256: [u8; 32],
    pub(crate) primary_context_identity_sha256: [u8; 32],
    pub(crate) run_stream_identity_sha256: [u8; 32],
    pub(crate) cuda_build_manifest_sha256: [u8; 32],
    pub(crate) resident_input_content_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RawResidentGenerationMetricRowV1 {
    pub(crate) candidate_id: u64,
    pub(crate) scenario_id: u64,
    pub(crate) values: [f64; 11],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawGenerationPlanV1 {
    abi_version: u32,
    parent_selection_policy: u32,
    survivor_selection_policy: u32,
    max_terms_per_gene: u32,
    minimum_terms_per_gene: u32,
    threshold_level_count: u32,
    smc_flag_count: u32,
    reserved: u32,
    logical_population_count: u64,
    retained_evaluation_capacity: u64,
    feature_count: u64,
    generation_count: u64,
    survivor_count: u64,
    immigrant_count: u64,
    search_seed: u64,
    mutation_intensity_q32: u64,
    threshold_ladder_bits: [u64; 6],
    stop_bounds_bits: [u64; 6],
    smc_probability_q32: [u64; 11],
    generation_semantics_sha256: [u8; 32],
    run_identity_sha256: [u8; 32],
    strategy_gene_schema_sha256: [u8; 32],
    rank_semantics_sha256: [u8; 32],
    metric_semantics_sha256: [u8; 32],
    scoring_semantics_sha256: [u8; 32],
    novelty_semantics_sha256: [u8; 32],
    scenario_order_semantics_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    rng_mapping_sha256: [u8; 32],
    plan_identity_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawAllocationReceiptV1 {
    abi_version: u32,
    generation_store_allocation_count: u32,
    logical_gene_scalar_bytes: u64,
    logical_gene_index_bytes: u64,
    logical_gene_weight_bytes: u64,
    offspring_bytes: u64,
    metric_row_bytes: u64,
    rank_key_bytes: u64,
    selection_bytes: u64,
    dedup_hash_bytes: u64,
    cub_scratch_bytes: u64,
    retained_evaluation_workspace_bytes: u64,
    total_device_bytes: u64,
    same_context_free_bytes: u64,
    full_discovery_reserve_bytes: u64,
    logical_population_count: u64,
    retained_evaluation_capacity: u64,
    generation_chunk_count: u64,
    allocation_plan_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawMetricRowsImportV1 {
    pub(crate) abi_version: u32,
    pub(crate) metric_value_count: u32,
    pub(crate) metric_rows_device: *const RawResidentGenerationMetricRowV1,
    pub(crate) resident_decision_keys_device: *const u64,
    pub(crate) expected_scenario_ids_device: *const u64,
    pub(crate) logical_offset: u64,
    pub(crate) active_scenarios: u64,
    pub(crate) scoring_novelty_ready_event: *mut c_void,
    pub(crate) metric_semantics_sha256: [u8; 32],
    pub(crate) scoring_semantics_sha256: [u8; 32],
    pub(crate) novelty_semantics_sha256: [u8; 32],
    pub(crate) scenario_order_semantics_sha256: [u8; 32],
    pub(crate) rank_semantics_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawReadyEventV1 {
    abi_version: u32,
    reserved: u32,
    event_id: u64,
    generation_index: u64,
    same_stream_enqueue_count: u64,
    intermediate_host_wait_count: u64,
    intermediate_readback_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawContentReceiptV1 {
    abi_version: u32,
    reserved: u32,
    gene_content_identity_handle: u64,
    metric_content_identity_handle: u64,
    generation_receipt_identity_handle: u64,
    ready_event_id: u64,
    final_compact_readback_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawPostGaInPlaceReceiptV1 {
    abi_version: u32,
    reserved: u32,
    ready_event_id: u64,
    current_generation_index: u64,
    same_stream_enqueue_count: u64,
    logical_population_count: u64,
    retained_evaluation_capacity: u64,
    generation_allocation_total_device_bytes: u64,
    additional_allocation_count: u64,
    additional_device_bytes: u64,
    gene_content_identity_handle: u64,
    metric_content_identity_handle: u64,
    generation_receipt_identity_handle: u64,
}

const _: [(); 208] = [(); std::mem::size_of::<RawPopulationSessionImportV1>()];
const _: [(); 104] = [(); std::mem::size_of::<RawResidentGenerationMetricRowV1>()];
const _: [(); 632] = [(); std::mem::size_of::<RawGenerationPlanV1>()];
const _: [(); 168] = [(); std::mem::size_of::<RawAllocationReceiptV1>()];
const _: [(); 216] = [(); std::mem::size_of::<RawMetricRowsImportV1>()];
const _: [(); 48] = [(); std::mem::size_of::<RawReadyEventV1>()];
const _: [(); 48] = [(); std::mem::size_of::<RawContentReceiptV1>()];
const _: [(); 96] = [(); std::mem::size_of::<RawPostGaInPlaceReceiptV1>()];

enum NativeResidentGenerationRunV1 {}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    #[link_name = "query_resident_generation_allocation_v1"]
    fn ffi_query_resident_generation_allocation_v1(
        import: *const RawPopulationSessionImportV1,
        plan: *const RawGenerationPlanV1,
        receipt: *mut RawAllocationReceiptV1,
    ) -> i32;
    #[link_name = "create_resident_generation_run_from_import_v1"]
    fn ffi_create_resident_generation_run_from_import_v1(
        import: *const RawPopulationSessionImportV1,
        plan: *const RawGenerationPlanV1,
        receipt: *const RawAllocationReceiptV1,
        run: *mut *mut NativeResidentGenerationRunV1,
    ) -> i32;
    #[link_name = "initialize_resident_generation_population_v1"]
    fn ffi_initialize_resident_generation_population_v1(
        run: *mut NativeResidentGenerationRunV1,
        ready: *mut RawReadyEventV1,
    ) -> i32;
    #[link_name = "enqueue_exact_generation_chunk_v1"]
    fn ffi_enqueue_exact_generation_chunk_v1(
        run: *mut NativeResidentGenerationRunV1,
        metrics: *const RawMetricRowsImportV1,
        ready: *mut RawReadyEventV1,
    ) -> i32;
    #[link_name = "enqueue_resident_rank_selection_offspring_v1"]
    fn ffi_enqueue_resident_rank_selection_offspring_v1(
        run: *mut NativeResidentGenerationRunV1,
        generation_index: u64,
        ready: *mut RawReadyEventV1,
    ) -> i32;
    #[link_name = "seal_resident_generation_content_v1"]
    fn ffi_seal_resident_generation_content_v1(
        run: *mut NativeResidentGenerationRunV1,
        receipt: *mut RawContentReceiptV1,
        ready: *mut RawReadyEventV1,
    ) -> i32;
    #[link_name = "begin_resident_post_ga_in_place_v1"]
    fn ffi_begin_resident_post_ga_in_place_v1(
        run: *mut NativeResidentGenerationRunV1,
        dependency: *const RawReadyEventV1,
        gene_content_identity_handle: u64,
        metric_content_identity_handle: u64,
        generation_receipt_identity_handle: u64,
        receipt: *mut RawPostGaInPlaceReceiptV1,
    ) -> i32;
    #[link_name = "enqueue_resident_generation_release_v1"]
    fn ffi_enqueue_resident_generation_release_v1(run: *mut NativeResidentGenerationRunV1) -> i32;
}

/// One private, move-only import minted by the existing population owner. The
/// boxed owner keeps the V3 parent, population session, context and stream live.
pub(crate) struct ResidentGenerationPopulationSessionImportV1 {
    raw: RawPopulationSessionImportV1,
    lifetime_owner: Option<Box<dyn Any>>,
}

impl ResidentGenerationPopulationSessionImportV1 {
    /// The only intended caller is the existing gpu-cuda population-session
    /// module after it has validated the exact admitted context and stream.
    ///
    /// # Safety
    ///
    /// Every raw handle and identity must originate from the one live admitted
    /// population session retained by `lifetime_owner`. The owner must keep the
    /// context, stream, parent buffers, parent-ready event and distinct
    /// generation-ready event alive until terminal stream-ordered consumption.
    pub(crate) unsafe fn from_population_session_parts_v1<T: Any>(
        raw: RawPopulationSessionImportV1,
        lifetime_owner: T,
    ) -> Result<Self, ResidentGenerationDeviceErrorV1> {
        if raw.abi_version != ABI_VERSION_V1
            || raw.admitted_run_stream.is_null()
            || raw.resident_parent_ready_event.is_null()
            || raw.generation_ready_event.is_null()
            || raw.generation_ready_event == raw.resident_parent_ready_event
            || raw.population_lifetime_owner.is_null()
            || raw.selected_cuda_ordinal == u32::MAX
            || identity_is_zero_v1(&raw.cuda_device_identity_sha256)
            || identity_is_zero_v1(&raw.primary_context_identity_sha256)
            || identity_is_zero_v1(&raw.run_stream_identity_sha256)
            || identity_is_zero_v1(&raw.cuda_build_manifest_sha256)
            || identity_is_zero_v1(&raw.resident_input_content_sha256)
        {
            return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
                "population import is incomplete",
            ));
        }
        Ok(Self {
            raw,
            lifetime_owner: Some(Box::new(lifetime_owner)),
        })
    }
}

impl Drop for ResidentGenerationPopulationSessionImportV1 {
    fn drop(&mut self) {
        if let Some(owner) = self.lifetime_owner.take() {
            std::mem::forget(owner);
        }
    }
}

pub(crate) struct ResidentGenerationPlanAuthorityInputV1 {
    pub(crate) parent_selection: ParentSelectionPolicyV1,
    pub(crate) survivor_selection: SurvivorSelectionPolicyV1,
    pub(crate) max_terms_per_gene: usize,
    pub(crate) minimum_terms_per_gene: usize,
    pub(crate) logical_population_count: usize,
    pub(crate) retained_evaluation_capacity: usize,
    pub(crate) feature_count: usize,
    pub(crate) generation_count: usize,
    pub(crate) survivor_count: usize,
    pub(crate) immigrant_count: usize,
    pub(crate) search_seed: u64,
    pub(crate) mutation_intensity_q32: u64,
    pub(crate) threshold_ladder_bits: [u64; 6],
    pub(crate) stop_bounds_bits: [u64; 6],
    pub(crate) smc_probability_q32: [u64; 11],
    pub(crate) generation_semantics_sha256: [u8; 32],
    pub(crate) run_identity_sha256: [u8; 32],
    pub(crate) strategy_gene_schema_sha256: [u8; 32],
    pub(crate) rank_semantics_sha256: [u8; 32],
    pub(crate) metric_semantics_sha256: [u8; 32],
    pub(crate) scoring_semantics_sha256: [u8; 32],
    pub(crate) novelty_semantics_sha256: [u8; 32],
    pub(crate) scenario_order_semantics_sha256: [u8; 32],
    pub(crate) cuda_build_manifest_sha256: [u8; 32],
    pub(crate) rng_mapping_sha256: [u8; 32],
}

pub struct SealedResidentGenerationPlanV1 {
    raw: RawGenerationPlanV1,
    plan_identity_sha256: [u8; 32],
}

pub(crate) fn seal_resident_generation_plan_v1(
    input: ResidentGenerationPlanAuthorityInputV1,
) -> Result<SealedResidentGenerationPlanV1, ResidentGenerationDeviceErrorV1> {
    validate_rank_weighted_only_v1(input.parent_selection, input.survivor_selection)?;
    if input.logical_population_count == 0
        || input.retained_evaluation_capacity == 0
        || input.retained_evaluation_capacity > input.logical_population_count
        || input.feature_count == 0
        || input.generation_count == 0
        || input.max_terms_per_gene == 0
        || input.minimum_terms_per_gene == 0
        || input.minimum_terms_per_gene > input.max_terms_per_gene
        || input.max_terms_per_gene > input.feature_count
        || input.logical_population_count > i32::MAX as usize
        || input.retained_evaluation_capacity > i32::MAX as usize
        || input.generation_count > u32::MAX as usize
        || input.survivor_count > input.logical_population_count
        || input.immigrant_count > input.logical_population_count
        || input
            .survivor_count
            .checked_add(input.immigrant_count)
            .is_none_or(|reserved| reserved > input.logical_population_count)
        || input.mutation_intensity_q32 > (1_u64 << 32)
        || identity_is_zero_v1(&input.run_identity_sha256)
        || identity_is_zero_v1(&input.strategy_gene_schema_sha256)
        || identity_is_zero_v1(&input.rank_semantics_sha256)
        || identity_is_zero_v1(&input.metric_semantics_sha256)
        || identity_is_zero_v1(&input.scoring_semantics_sha256)
        || identity_is_zero_v1(&input.novelty_semantics_sha256)
        || identity_is_zero_v1(&input.scenario_order_semantics_sha256)
        || identity_is_zero_v1(&input.cuda_build_manifest_sha256)
        || identity_is_zero_v1(&input.rng_mapping_sha256)
    {
        return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
            "generation extents are invalid",
        ));
    }
    if input
        .smc_probability_q32
        .iter()
        .any(|probability| *probability > (1_u64 << 32))
    {
        return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
            "SMC Q32 probability exceeds one",
        ));
    }
    let expected_semantics = sha256_v1(&[DISCOVERY_GENERATION_SEMANTICS_V1.as_bytes()]);
    if input.generation_semantics_sha256 != expected_semantics {
        return Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
            "generation semantics",
        ));
    }
    validate_f64_plan_bits_v1(&input.threshold_ladder_bits, true)?;
    validate_f64_plan_bits_v1(&input.stop_bounds_bits, false)?;

    let mut raw = RawGenerationPlanV1 {
        abi_version: ABI_VERSION_V1,
        parent_selection_policy: input.parent_selection as u32,
        survivor_selection_policy: input.survivor_selection as u32,
        max_terms_per_gene: checked_u32_v1(input.max_terms_per_gene)?,
        minimum_terms_per_gene: checked_u32_v1(input.minimum_terms_per_gene)?,
        threshold_level_count: 6,
        smc_flag_count: 11,
        reserved: 0,
        logical_population_count: checked_u64_v1(input.logical_population_count)?,
        retained_evaluation_capacity: checked_u64_v1(input.retained_evaluation_capacity)?,
        feature_count: checked_u64_v1(input.feature_count)?,
        generation_count: checked_u64_v1(input.generation_count)?,
        survivor_count: checked_u64_v1(input.survivor_count)?,
        immigrant_count: checked_u64_v1(input.immigrant_count)?,
        search_seed: input.search_seed,
        mutation_intensity_q32: input.mutation_intensity_q32,
        threshold_ladder_bits: input.threshold_ladder_bits,
        stop_bounds_bits: input.stop_bounds_bits,
        smc_probability_q32: input.smc_probability_q32,
        generation_semantics_sha256: input.generation_semantics_sha256,
        run_identity_sha256: input.run_identity_sha256,
        strategy_gene_schema_sha256: input.strategy_gene_schema_sha256,
        rank_semantics_sha256: input.rank_semantics_sha256,
        metric_semantics_sha256: input.metric_semantics_sha256,
        scoring_semantics_sha256: input.scoring_semantics_sha256,
        novelty_semantics_sha256: input.novelty_semantics_sha256,
        scenario_order_semantics_sha256: input.scenario_order_semantics_sha256,
        cuda_build_manifest_sha256: input.cuda_build_manifest_sha256,
        rng_mapping_sha256: input.rng_mapping_sha256,
        plan_identity_sha256: [0; 32],
    };
    let plan_identity_sha256 = hash_raw_plan_v1(&raw);
    raw.plan_identity_sha256 = plan_identity_sha256;
    Ok(SealedResidentGenerationPlanV1 {
        raw,
        plan_identity_sha256,
    })
}

fn validate_rank_weighted_only_v1(
    parent: ParentSelectionPolicyV1,
    survivor: SurvivorSelectionPolicyV1,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    match parent {
        ParentSelectionPolicyV1::RankWeighted => {}
        ParentSelectionPolicyV1::Uniform => {
            return Err(ResidentGenerationDeviceErrorV1::UnsupportedUniformSelection);
        }
        ParentSelectionPolicyV1::Tournament => {
            return Err(ResidentGenerationDeviceErrorV1::UnsupportedTournamentSelection);
        }
        ParentSelectionPolicyV1::Softmax => {
            return Err(ResidentGenerationDeviceErrorV1::UnsupportedSoftmaxSelection);
        }
    }
    match survivor {
        SurvivorSelectionPolicyV1::RankWeighted => Ok(()),
        SurvivorSelectionPolicyV1::Elitist => {
            Err(ResidentGenerationDeviceErrorV1::UnsupportedElitistSelection)
        }
        SurvivorSelectionPolicyV1::Tournament => {
            Err(ResidentGenerationDeviceErrorV1::UnsupportedTournamentSelection)
        }
        SurvivorSelectionPolicyV1::Generational => {
            Err(ResidentGenerationDeviceErrorV1::UnsupportedGenerationalSelection)
        }
    }
}

pub struct ActualResidentGenerationAllocationPlanV1 {
    raw: RawAllocationReceiptV1,
    logical_gene_scalar_bytes: u64,
    logical_gene_index_bytes: u64,
    logical_gene_weight_bytes: u64,
    offspring_bytes: u64,
    metric_row_bytes: u64,
    rank_key_bytes: u64,
    selection_bytes: u64,
    dedup_hash_bytes: u64,
    cub_scratch_bytes: u64,
    retained_evaluation_workspace_bytes: u64,
    same_context_free_bytes: u64,
    full_discovery_reserve_bytes: u64,
    total_device_bytes: u64,
}

impl ActualResidentGenerationAllocationPlanV1 {
    pub const fn retained_batch_capacity(&self) -> usize {
        self.raw.retained_evaluation_capacity as usize
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

    pub const fn generation_store_allocation_count(&self) -> u32 {
        self.raw.generation_store_allocation_count
    }
}

pub(crate) fn query_actual_resident_generation_allocation_plan_v1(
    import: &ResidentGenerationPopulationSessionImportV1,
    plan: &SealedResidentGenerationPlanV1,
) -> Result<ActualResidentGenerationAllocationPlanV1, ResidentGenerationDeviceErrorV1> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (import, plan);
        Err(ResidentGenerationDeviceErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        let mut raw = RawAllocationReceiptV1::default();
        // SAFETY: both sealed inputs and the out-receipt remain live for the
        // call. The native query reads same-context memory facts itself.
        let status = unsafe {
            ffi_query_resident_generation_allocation_v1(&import.raw, &plan.raw, &mut raw)
        };
        require_native_ok_v1("query_resident_generation_allocation_v1", status)?;
        validate_allocation_receipt_v1(import, plan, &raw)
    }
}

#[must_use = "resident generation work must be consumed on the admitted run stream"]
pub struct ResidentGenerationDeviceRunV1 {
    native: NonNull<NativeResidentGenerationRunV1>,
    population_session_import: Option<ResidentGenerationPopulationSessionImportV1>,
    dependency_lifetime_owners: Vec<Box<dyn Any>>,
    state: ResidentGenerationRunStateV1,
    selected_cuda_ordinal: u32,
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    generation_semantics_sha256: [u8; 32],
    run_identity_sha256: [u8; 32],
    plan: SealedResidentGenerationPlanV1,
    allocation: ActualResidentGenerationAllocationPlanV1,
}

pub(crate) fn bind_population_session_import_v1(
    import: ResidentGenerationPopulationSessionImportV1,
    plan: SealedResidentGenerationPlanV1,
    allocation: ActualResidentGenerationAllocationPlanV1,
) -> Result<ResidentGenerationDeviceRunV1, ResidentGenerationDeviceErrorV1> {
    validate_import_plan_allocation_identity_v1(&import, &plan, &allocation)?;
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (import, plan, allocation);
        Err(ResidentGenerationDeviceErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        let mut native = std::ptr::null_mut();
        // SAFETY: the import retains the population owner and admitted stream;
        // the native side copies only metadata and aliases that same stream.
        let status = unsafe {
            ffi_create_resident_generation_run_from_import_v1(
                &import.raw,
                &plan.raw,
                &allocation.raw,
                &mut native,
            )
        };
        require_native_ok_v1("create_resident_generation_run_from_import_v1", status)?;
        let native = NonNull::new(native).ok_or(ResidentGenerationDeviceErrorV1::Native {
            operation: "create_resident_generation_run_from_import_v1",
            status,
        })?;
        Ok(ResidentGenerationDeviceRunV1 {
            native,
            state: ResidentGenerationRunStateV1::StrictIdle,
            selected_cuda_ordinal: import.raw.selected_cuda_ordinal,
            primary_context_identity_sha256: import.raw.primary_context_identity_sha256,
            run_stream_identity_sha256: import.raw.run_stream_identity_sha256,
            cuda_build_manifest_sha256: import.raw.cuda_build_manifest_sha256,
            generation_semantics_sha256: plan.raw.generation_semantics_sha256,
            run_identity_sha256: plan.raw.run_identity_sha256,
            population_session_import: Some(import),
            dependency_lifetime_owners: Vec::new(),
            plan,
            allocation,
        })
    }
}

#[must_use = "the ready event owns its native run until the next resident stage consumes it"]
pub struct ResidentGenerationReadyEventV1 {
    run: Option<ResidentGenerationDeviceRunV1>,
    raw: RawReadyEventV1,
}

pub(crate) fn initialize_resident_generation_population_v1(
    mut run: ResidentGenerationDeviceRunV1,
) -> Result<ResidentGenerationReadyEventV1, ResidentGenerationDeviceErrorV1> {
    require_run_state_v1(&run, ResidentGenerationRunStateV1::StrictIdle)?;
    run.state = ResidentGenerationRunStateV1::InFlight;
    let mut raw = RawReadyEventV1::default();
    #[cfg(not(feature = "cuda"))]
    {
        let _ = raw;
        run.state = ResidentGenerationRunStateV1::Poisoned;
        Err(ResidentGenerationDeviceErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        // SAFETY: the move-only run owns the native handle and admitted stream.
        let status = unsafe {
            ffi_initialize_resident_generation_population_v1(run.native.as_ptr(), &mut raw)
        };
        if status != STATUS_OK_V1 {
            run.state = ResidentGenerationRunStateV1::Poisoned;
            return Err(native_error_v1(
                "initialize_resident_generation_population_v1",
                status,
            ));
        }
        validate_ready_event_v1(&run, &raw)?;
        Ok(ResidentGenerationReadyEventV1 {
            run: Some(run),
            raw,
        })
    }
}

/// One move-only import minted only by the future strict scoring/novelty GPU
/// stage. Its owner retains the exact metric rows, integer decision keys,
/// scenario ordering and ready event until the generation stream consumes them.
pub(crate) struct ResidentScoredDecisionRowsEventImportV1 {
    raw: RawMetricRowsImportV1,
    lifetime_owner: Option<Box<dyn Any>>,
}

impl ResidentScoredDecisionRowsEventImportV1 {
    /// # Safety
    ///
    /// The metric rows, decision keys and expected scenario IDs must be live
    /// device allocations in the admitted primary context. The ready event must
    /// order their final writes, and `lifetime_owner` must retain all of them.
    pub(crate) unsafe fn from_scoring_novelty_parts_v1<T: Any>(
        raw: RawMetricRowsImportV1,
        lifetime_owner: T,
    ) -> Result<Self, ResidentGenerationDeviceErrorV1> {
        if raw.abi_version != ABI_VERSION_V1
            || raw.metric_value_count != 11
            || raw.metric_rows_device.is_null()
            || raw.resident_decision_keys_device.is_null()
            || raw.expected_scenario_ids_device.is_null()
            || raw.scoring_novelty_ready_event.is_null()
            || identity_is_zero_v1(&raw.metric_semantics_sha256)
            || identity_is_zero_v1(&raw.scoring_semantics_sha256)
            || identity_is_zero_v1(&raw.novelty_semantics_sha256)
            || identity_is_zero_v1(&raw.scenario_order_semantics_sha256)
            || identity_is_zero_v1(&raw.rank_semantics_sha256)
        {
            return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
                "scored decision-row event import is incomplete",
            ));
        }
        Ok(Self {
            raw,
            lifetime_owner: Some(Box::new(lifetime_owner)),
        })
    }

    fn into_raw_and_owner_v1(mut self) -> (RawMetricRowsImportV1, Box<dyn Any>) {
        let raw = self.raw;
        let owner = self
            .lifetime_owner
            .take()
            .expect("validated metric import owns its lifetime authority");
        (raw, owner)
    }
}

pub(crate) fn enqueue_exact_generation_chunk_v1(
    ready: ResidentGenerationReadyEventV1,
    metrics: ResidentScoredDecisionRowsEventImportV1,
) -> Result<ResidentGenerationReadyEventV1, ResidentGenerationDeviceErrorV1> {
    let (mut run, dependency) = ready.into_parts_v1()?;
    validate_exact_chunk_v1(&run.plan, &metrics.raw)?;
    let (metrics_raw, metrics_owner) = metrics.into_raw_and_owner_v1();
    let mut next = RawReadyEventV1::default();
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (dependency, metrics_raw, metrics_owner, next);
        run.state = ResidentGenerationRunStateV1::Poisoned;
        Err(ResidentGenerationDeviceErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        require_dependency_identity_v1(&run, &dependency)?;
        // SAFETY: both dependency owners remain live through the same-stream
        // enqueue; native code retains neither Rust address.
        // The owner is attached before the fallible native enqueue. Any
        // ambiguous-after-launch failure therefore takes the run's leak-only
        // Drop path instead of freeing metrics that the stream may still read.
        run.dependency_lifetime_owners.push(metrics_owner);
        let status = unsafe {
            ffi_enqueue_exact_generation_chunk_v1(run.native.as_ptr(), &metrics_raw, &mut next)
        };
        if status != STATUS_OK_V1 {
            run.state = ResidentGenerationRunStateV1::Poisoned;
            return Err(native_error_v1("enqueue_exact_generation_chunk_v1", status));
        }
        validate_ready_event_v1(&run, &next)?;
        Ok(ResidentGenerationReadyEventV1 {
            run: Some(run),
            raw: next,
        })
    }
}

pub(crate) fn enqueue_resident_rank_selection_offspring_v1(
    ready: ResidentGenerationReadyEventV1,
    generation_index: usize,
) -> Result<ResidentGenerationReadyEventV1, ResidentGenerationDeviceErrorV1> {
    let (mut run, dependency) = ready.into_parts_v1()?;
    require_dependency_identity_v1(&run, &dependency)?;
    let generation_index = checked_u64_v1(generation_index)?;
    if generation_index >= run.plan.raw.generation_count {
        run.state = ResidentGenerationRunStateV1::Poisoned;
        return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
            "generation index is outside the sealed plan",
        ));
    }
    let mut next = RawReadyEventV1::default();
    #[cfg(not(feature = "cuda"))]
    {
        let _ = next;
        run.state = ResidentGenerationRunStateV1::Poisoned;
        Err(ResidentGenerationDeviceErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        // SAFETY: native work is appended to the run's one admitted stream.
        let status = unsafe {
            ffi_enqueue_resident_rank_selection_offspring_v1(
                run.native.as_ptr(),
                generation_index,
                &mut next,
            )
        };
        if status != STATUS_OK_V1 {
            run.state = ResidentGenerationRunStateV1::Poisoned;
            return Err(native_error_v1(
                "enqueue_resident_rank_selection_offspring_v1",
                status,
            ));
        }
        validate_ready_event_v1(&run, &next)?;
        Ok(ResidentGenerationReadyEventV1 {
            run: Some(run),
            raw: next,
        })
    }
}

pub struct ResidentGenerationContentIdentityV1 {
    identity_handle: u64,
    run_identity_sha256: [u8; 32],
}

pub struct ResidentGenerationReceiptIdentityV1 {
    identity_handle: u64,
    run_identity_sha256: [u8; 32],
}

pub struct SealedResidentGenerationDeviceOutcomeV1 {
    ready: ResidentGenerationReadyEventV1,
    resident_gene_content: ResidentGenerationContentIdentityV1,
    resident_metric_content: ResidentGenerationContentIdentityV1,
    resident_generation_receipt: ResidentGenerationReceiptIdentityV1,
    artifact_class: GenerationArtifactClassV1,
    promotion_eligibility: GenerationPromotionEligibilityV1,
}

pub(crate) fn seal_content_identities_on_device_v1(
    ready: ResidentGenerationReadyEventV1,
) -> Result<SealedResidentGenerationDeviceOutcomeV1, ResidentGenerationDeviceErrorV1> {
    let (mut run, dependency) = ready.into_parts_v1()?;
    require_dependency_identity_v1(&run, &dependency)?;
    let mut raw_receipt = RawContentReceiptV1::default();
    let mut raw_ready = RawReadyEventV1::default();
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (raw_receipt, raw_ready);
        run.state = ResidentGenerationRunStateV1::Poisoned;
        Err(ResidentGenerationDeviceErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        // SAFETY: content identities and their receipt stay on the owning
        // device; only opaque native handles are returned.
        let status = unsafe {
            ffi_seal_resident_generation_content_v1(
                run.native.as_ptr(),
                &mut raw_receipt,
                &mut raw_ready,
            )
        };
        if status != STATUS_OK_V1 {
            run.state = ResidentGenerationRunStateV1::Poisoned;
            return Err(native_error_v1(
                "seal_resident_generation_content_v1",
                status,
            ));
        }
        if raw_receipt.abi_version != ABI_VERSION_V1
            || raw_receipt.ready_event_id != raw_ready.event_id
            || !(raw_receipt.final_compact_readback_count == 0)
            || raw_receipt.gene_content_identity_handle == 0
            || raw_receipt.metric_content_identity_handle == 0
            || raw_receipt.generation_receipt_identity_handle == 0
        {
            run.state = ResidentGenerationRunStateV1::Poisoned;
            return Err(ResidentGenerationDeviceErrorV1::DeviceContentFault);
        }
        validate_ready_event_v1(&run, &raw_ready)?;
        run.state = ResidentGenerationRunStateV1::Sealed;
        let run_identity_sha256 = run.run_identity_sha256;
        Ok(SealedResidentGenerationDeviceOutcomeV1 {
            ready: ResidentGenerationReadyEventV1 {
                run: Some(run),
                raw: raw_ready,
            },
            resident_gene_content: ResidentGenerationContentIdentityV1 {
                identity_handle: raw_receipt.gene_content_identity_handle,
                run_identity_sha256,
            },
            resident_metric_content: ResidentGenerationContentIdentityV1 {
                identity_handle: raw_receipt.metric_content_identity_handle,
                run_identity_sha256,
            },
            resident_generation_receipt: ResidentGenerationReceiptIdentityV1 {
                identity_handle: raw_receipt.generation_receipt_identity_handle,
                run_identity_sha256,
            },
            artifact_class: GenerationArtifactClassV1::ResearchOnly,
            promotion_eligibility: GenerationPromotionEligibilityV1::NotPromotionEligible,
        })
    }
}

pub(crate) struct ResidentGenerationPostGaInputV1 {
    ready: ResidentGenerationReadyEventV1,
    gene_content: ResidentGenerationContentIdentityV1,
    metric_content: ResidentGenerationContentIdentityV1,
    receipt: ResidentGenerationReceiptIdentityV1,
}

struct ResidentGenerationPostGaContentAuthorityV1 {
    gene_content: ResidentGenerationContentIdentityV1,
    metric_content: ResidentGenerationContentIdentityV1,
    receipt: ResidentGenerationReceiptIdentityV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentGenerationPostGaWorkspaceStateV1 {
    GenerationStoresOnly,
}

#[must_use = "post-GA in-place work owns the generation run until a resident stage consumes it"]
pub(crate) struct ResidentGenerationPostGaInPlaceRunV1 {
    run: Option<ResidentGenerationDeviceRunV1>,
    dependency: RawReadyEventV1,
    content_authority: ResidentGenerationPostGaContentAuthorityV1,
    receipt: RawPostGaInPlaceReceiptV1,
    workspace_state: ResidentGenerationPostGaWorkspaceStateV1,
}

impl SealedResidentGenerationDeviceOutcomeV1 {
    pub(crate) fn consume_into_post_ga_v1(self) -> ResidentGenerationPostGaInputV1 {
        ResidentGenerationPostGaInputV1 {
            ready: self.ready,
            gene_content: self.resident_gene_content,
            metric_content: self.resident_metric_content,
            receipt: self.resident_generation_receipt,
        }
    }
}

pub(crate) fn begin_resident_post_ga_in_place_v1(
    input: ResidentGenerationPostGaInputV1,
) -> Result<ResidentGenerationPostGaInPlaceRunV1, ResidentGenerationDeviceErrorV1> {
    let ResidentGenerationPostGaInputV1 {
        ready,
        gene_content,
        metric_content,
        receipt,
    } = input;
    let (mut run, dependency) = ready.into_parts_v1()?;
    require_run_state_v1(&run, ResidentGenerationRunStateV1::Sealed)?;
    let content_authority = ResidentGenerationPostGaContentAuthorityV1 {
        gene_content,
        metric_content,
        receipt,
    };
    validate_post_ga_content_identity_v1(&run, &content_authority)?;
    let mut raw_receipt = RawPostGaInPlaceReceiptV1::default();

    // Once native validation is attempted, any failure can be ambiguous with
    // respect to an enqueued same-stream dependency. Poison before FFI so every
    // error path retains the run and all evaluator/scoring lifetime owners.
    run.state = ResidentGenerationRunStateV1::Poisoned;
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (&dependency, &content_authority, &mut raw_receipt);
        Err(ResidentGenerationDeviceErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        // SAFETY: the move-only Rust owner retains the sole native generation
        // run, admitted stream/event and every imported dependency owner. The
        // native bridge returns identity/count evidence only and allocates no
        // device or host storage.
        let status = unsafe {
            ffi_begin_resident_post_ga_in_place_v1(
                run.native.as_ptr(),
                &dependency,
                content_authority.gene_content.identity_handle,
                content_authority.metric_content.identity_handle,
                content_authority.receipt.identity_handle,
                &mut raw_receipt,
            )
        };
        if status != STATUS_OK_V1 {
            return Err(native_error_v1(
                "begin_resident_post_ga_in_place_v1",
                status,
            ));
        }
        validate_post_ga_in_place_receipt_v1(&run, &dependency, &content_authority, &raw_receipt)?;
        run.state = ResidentGenerationRunStateV1::PostGaInPlace;
        Ok(ResidentGenerationPostGaInPlaceRunV1 {
            run: Some(run),
            dependency,
            content_authority,
            receipt: raw_receipt,
            workspace_state: ResidentGenerationPostGaWorkspaceStateV1::GenerationStoresOnly,
        })
    }
}

impl ResidentGenerationReadyEventV1 {
    fn into_parts_v1(
        mut self,
    ) -> Result<(ResidentGenerationDeviceRunV1, RawReadyEventV1), ResidentGenerationDeviceErrorV1>
    {
        let run = self
            .run
            .take()
            .ok_or(ResidentGenerationDeviceErrorV1::RunStateViolation)?;
        Ok((run, self.raw))
    }
}

impl Drop for ResidentGenerationDeviceRunV1 {
    fn drop(&mut self) {
        // No terminal completion lease exists in this additive slice. Even an
        // idle-looking run may own queued allocation work on the admitted
        // stream, so ordinary Drop is always leak-only. A later terminal GPU
        // stage must consume the opaque run and return a completion lease before
        // the native async release API can be exposed safely.
        leak_live_native_generation_run_v1(self);
    }
}

fn leak_live_native_generation_run_v1(run: &mut ResidentGenerationDeviceRunV1) {
    for owner in run.dependency_lifetime_owners.drain(..) {
        std::mem::forget(owner);
    }
    if let Some(import) = run.population_session_import.take() {
        std::mem::forget(import);
    }
}

fn checked_generation_chunk_count_v1(
    logical_population_count: usize,
    retained_evaluation_capacity: usize,
) -> Result<usize, ResidentGenerationDeviceErrorV1> {
    if !(retained_evaluation_capacity >= 1) || logical_population_count == 0 {
        return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
            "chunk capacities must be non-zero",
        ));
    }
    logical_population_count
        .checked_add(retained_evaluation_capacity - 1)
        .and_then(|value| value.checked_div(retained_evaluation_capacity))
        .ok_or(ResidentGenerationDeviceErrorV1::ArithmeticOverflow)
}

fn checked_generation_chunk_range_v1(
    logical_population_count: usize,
    retained_evaluation_capacity: usize,
    chunk_index: usize,
) -> Result<std::ops::Range<usize>, ResidentGenerationDeviceErrorV1> {
    let start = chunk_index
        .checked_mul(retained_evaluation_capacity)
        .ok_or(ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?;
    let end = start
        .checked_add(retained_evaluation_capacity)
        .ok_or(ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?
        .min(logical_population_count);
    let active_scenarios = end.saturating_sub(start);
    if !(active_scenarios <= retained_evaluation_capacity) || active_scenarios == 0 {
        return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
            "chunk is empty or exceeds retained capacity",
        ));
    }
    Ok(start..end)
}

fn validate_exact_chunk_v1(
    plan: &SealedResidentGenerationPlanV1,
    metrics: &RawMetricRowsImportV1,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    let logical_population_count = usize::try_from(plan.raw.logical_population_count)
        .map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?;
    let retained_evaluation_capacity = usize::try_from(plan.raw.retained_evaluation_capacity)
        .map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?;
    let chunk_count =
        checked_generation_chunk_count_v1(logical_population_count, retained_evaluation_capacity)?;
    let logical_offset = usize::try_from(metrics.logical_offset)
        .map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?;
    let active_scenarios = usize::try_from(metrics.active_scenarios)
        .map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?;
    let chunk_index = logical_offset
        .checked_div(retained_evaluation_capacity)
        .ok_or(ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?;
    if chunk_index >= chunk_count {
        return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
            "chunk index is outside the exact schedule",
        ));
    }
    let exact = checked_generation_chunk_range_v1(
        logical_population_count,
        retained_evaluation_capacity,
        chunk_index,
    )?;
    let covered_logical_population = checked_generation_chunk_range_v1(
        logical_population_count,
        retained_evaluation_capacity,
        chunk_count - 1,
    )?
    .end;
    if !(covered_logical_population == logical_population_count)
        || exact.start != logical_offset
        || exact.len() != active_scenarios
        || metrics.metric_value_count != 11
        || metrics.metric_rows_device.is_null()
        || metrics.resident_decision_keys_device.is_null()
        || metrics.expected_scenario_ids_device.is_null()
        || metrics.scoring_novelty_ready_event.is_null()
        || metrics.metric_semantics_sha256 != plan.raw.metric_semantics_sha256
        || metrics.scoring_semantics_sha256 != plan.raw.scoring_semantics_sha256
        || metrics.novelty_semantics_sha256 != plan.raw.novelty_semantics_sha256
        || metrics.scenario_order_semantics_sha256 != plan.raw.scenario_order_semantics_sha256
        || metrics.rank_semantics_sha256 != plan.raw.rank_semantics_sha256
    {
        return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
            "metric chunk differs from the exact sealed schedule",
        ));
    }
    Ok(())
}

fn validate_allocation_receipt_v1(
    import: &ResidentGenerationPopulationSessionImportV1,
    plan: &SealedResidentGenerationPlanV1,
    raw: &RawAllocationReceiptV1,
) -> Result<ActualResidentGenerationAllocationPlanV1, ResidentGenerationDeviceErrorV1> {
    let expected_chunks = checked_generation_chunk_count_v1(
        usize::try_from(plan.raw.logical_population_count)
            .map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?,
        usize::try_from(plan.raw.retained_evaluation_capacity)
            .map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?,
    )?;
    if raw.abi_version != ABI_VERSION_V1
        || raw.generation_store_allocation_count != 1
        || raw.logical_population_count != plan.raw.logical_population_count
        || raw.retained_evaluation_capacity != plan.raw.retained_evaluation_capacity
        || raw.generation_chunk_count
            != u64::try_from(expected_chunks)
                .map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?
        || raw.full_discovery_reserve_bytes != import.raw.full_discovery_reserve_bytes
    {
        return Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
            "allocation receipt",
        ));
    }
    let charged = [
        raw.logical_gene_scalar_bytes,
        raw.logical_gene_index_bytes,
        raw.logical_gene_weight_bytes,
        raw.offspring_bytes,
        raw.metric_row_bytes,
        raw.rank_key_bytes,
        raw.selection_bytes,
        raw.dedup_hash_bytes,
        raw.cub_scratch_bytes,
        raw.retained_evaluation_workspace_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| total.checked_add(bytes))
    .ok_or(ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?;
    let reusable = raw
        .same_context_free_bytes
        .checked_sub(raw.full_discovery_reserve_bytes)
        .ok_or(ResidentGenerationDeviceErrorV1::CapacityUnavailable)?;
    if charged != raw.total_device_bytes || raw.total_device_bytes > reusable {
        return Err(ResidentGenerationDeviceErrorV1::CapacityUnavailable);
    }
    Ok(ActualResidentGenerationAllocationPlanV1 {
        raw: *raw,
        logical_gene_scalar_bytes: raw.logical_gene_scalar_bytes,
        logical_gene_index_bytes: raw.logical_gene_index_bytes,
        logical_gene_weight_bytes: raw.logical_gene_weight_bytes,
        offspring_bytes: raw.offspring_bytes,
        metric_row_bytes: raw.metric_row_bytes,
        rank_key_bytes: raw.rank_key_bytes,
        selection_bytes: raw.selection_bytes,
        dedup_hash_bytes: raw.dedup_hash_bytes,
        cub_scratch_bytes: raw.cub_scratch_bytes,
        retained_evaluation_workspace_bytes: raw.retained_evaluation_workspace_bytes,
        same_context_free_bytes: raw.same_context_free_bytes,
        full_discovery_reserve_bytes: raw.full_discovery_reserve_bytes,
        total_device_bytes: raw.total_device_bytes,
    })
}

fn validate_import_plan_allocation_identity_v1(
    import: &ResidentGenerationPopulationSessionImportV1,
    plan: &SealedResidentGenerationPlanV1,
    allocation: &ActualResidentGenerationAllocationPlanV1,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    if import.raw.abi_version != ABI_VERSION_V1
        || plan.raw.abi_version != ABI_VERSION_V1
        || allocation.raw.abi_version != ABI_VERSION_V1
        || import.raw.admitted_run_stream.is_null()
        || import.raw.selected_cuda_ordinal == u32::MAX
        || import.raw.cuda_build_manifest_sha256 != plan.raw.cuda_build_manifest_sha256
        || plan.plan_identity_sha256 != allocation.raw.allocation_plan_sha256
    {
        return Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
            "population import, plan and allocation",
        ));
    }
    Ok(())
}

fn validate_ready_event_v1(
    run: &ResidentGenerationDeviceRunV1,
    ready: &RawReadyEventV1,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    if ready.abi_version != ABI_VERSION_V1
        || ready.event_id == 0
        || ready.intermediate_host_wait_count != 0
        || ready.intermediate_readback_count != 0
        || ready.generation_index > run.plan.raw.generation_count
    {
        return Err(ResidentGenerationDeviceErrorV1::EventIdentityMismatch);
    }
    Ok(())
}

fn require_dependency_identity_v1(
    run: &ResidentGenerationDeviceRunV1,
    dependency: &RawReadyEventV1,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    validate_ready_event_v1(run, dependency)
}

fn require_run_state_v1(
    run: &ResidentGenerationDeviceRunV1,
    required: ResidentGenerationRunStateV1,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    if run.state != required {
        return Err(ResidentGenerationDeviceErrorV1::RunStateViolation);
    }
    Ok(())
}

fn validate_post_ga_content_identity_v1(
    run: &ResidentGenerationDeviceRunV1,
    content: &ResidentGenerationPostGaContentAuthorityV1,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    if content.gene_content.identity_handle == 0
        || content.metric_content.identity_handle == 0
        || content.receipt.identity_handle == 0
        || content.gene_content.run_identity_sha256 != run.run_identity_sha256
        || content.metric_content.run_identity_sha256 != run.run_identity_sha256
        || content.receipt.run_identity_sha256 != run.run_identity_sha256
    {
        return Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
            "post-GA generation content",
        ));
    }
    Ok(())
}

fn validate_post_ga_in_place_receipt_v1(
    run: &ResidentGenerationDeviceRunV1,
    dependency: &RawReadyEventV1,
    content: &ResidentGenerationPostGaContentAuthorityV1,
    receipt: &RawPostGaInPlaceReceiptV1,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    let exact_generation_allocation =
        receipt.generation_allocation_total_device_bytes == run.allocation.total_device_bytes();
    if receipt.abi_version != ABI_VERSION_V1
        || receipt.ready_event_id != dependency.event_id
        || receipt.current_generation_index != dependency.generation_index
        || receipt.same_stream_enqueue_count != dependency.same_stream_enqueue_count
        || receipt.logical_population_count != run.plan.raw.logical_population_count
        || receipt.retained_evaluation_capacity != run.plan.raw.retained_evaluation_capacity
        || !exact_generation_allocation
        || !(receipt.additional_allocation_count == 0)
        || !(receipt.additional_device_bytes == 0)
        || receipt.gene_content_identity_handle != content.gene_content.identity_handle
        || receipt.metric_content_identity_handle != content.metric_content.identity_handle
        || receipt.generation_receipt_identity_handle != content.receipt.identity_handle
    {
        return Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
            "post-GA in-place bridge receipt",
        ));
    }
    Ok(())
}

fn validate_f64_plan_bits_v1(
    bits: &[u64],
    strictly_increasing: bool,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    let mut prior = None;
    for raw in bits {
        let value = f64::from_bits(*raw);
        if !value.is_finite() || value < 0.0 {
            return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
                "non-finite or negative generation geometry",
            ));
        }
        if strictly_increasing && prior.is_some_and(|previous| value <= previous) {
            return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
                "threshold ladder is not strictly increasing",
            ));
        }
        prior = Some(value);
    }
    Ok(())
}

fn hash_raw_plan_v1(plan: &RawGenerationPlanV1) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.resident-generation-plan.v1\0");
    hasher.update(plan.abi_version.to_le_bytes());
    hasher.update(plan.parent_selection_policy.to_le_bytes());
    hasher.update(plan.survivor_selection_policy.to_le_bytes());
    hasher.update(plan.max_terms_per_gene.to_le_bytes());
    hasher.update(plan.minimum_terms_per_gene.to_le_bytes());
    hasher.update(plan.threshold_level_count.to_le_bytes());
    hasher.update(plan.smc_flag_count.to_le_bytes());
    hasher.update(plan.reserved.to_le_bytes());
    hasher.update(plan.logical_population_count.to_le_bytes());
    hasher.update(plan.retained_evaluation_capacity.to_le_bytes());
    hasher.update(plan.feature_count.to_le_bytes());
    hasher.update(plan.generation_count.to_le_bytes());
    hasher.update(plan.survivor_count.to_le_bytes());
    hasher.update(plan.immigrant_count.to_le_bytes());
    hasher.update(plan.search_seed.to_le_bytes());
    hasher.update(plan.mutation_intensity_q32.to_le_bytes());
    for value in plan.threshold_ladder_bits {
        hasher.update(value.to_le_bytes());
    }
    for value in plan.stop_bounds_bits {
        hasher.update(value.to_le_bytes());
    }
    for value in plan.smc_probability_q32 {
        hasher.update(value.to_le_bytes());
    }
    hasher.update(plan.generation_semantics_sha256);
    hasher.update(plan.run_identity_sha256);
    hasher.update(plan.strategy_gene_schema_sha256);
    hasher.update(plan.rank_semantics_sha256);
    hasher.update(plan.metric_semantics_sha256);
    hasher.update(plan.scoring_semantics_sha256);
    hasher.update(plan.novelty_semantics_sha256);
    hasher.update(plan.scenario_order_semantics_sha256);
    hasher.update(plan.cuda_build_manifest_sha256);
    hasher.update(plan.rng_mapping_sha256);
    hasher.finalize().into()
}

fn sha256_v1(fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for field in fields {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hasher.finalize().into()
}

fn identity_is_zero_v1(identity: &[u8; 32]) -> bool {
    identity.iter().all(|byte| *byte == 0)
}

fn checked_u32_v1(value: usize) -> Result<u32, ResidentGenerationDeviceErrorV1> {
    u32::try_from(value).map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)
}

fn checked_u64_v1(value: usize) -> Result<u64, ResidentGenerationDeviceErrorV1> {
    u64::try_from(value).map_err(|_| ResidentGenerationDeviceErrorV1::ArithmeticOverflow)
}

fn require_native_ok_v1(
    operation: &'static str,
    status: i32,
) -> Result<(), ResidentGenerationDeviceErrorV1> {
    if status == STATUS_OK_V1 {
        Ok(())
    } else {
        Err(native_error_v1(operation, status))
    }
}

fn native_error_v1(operation: &'static str, status: i32) -> ResidentGenerationDeviceErrorV1 {
    ResidentGenerationDeviceErrorV1::Native { operation, status }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_zero_vector_matches_random123() {
        assert_eq!(
            philox4x32_10_reference_v1([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn counter_address_changes_for_every_bound_dimension() {
        let run = [7_u8; 32];
        let base = checked_philox_counter_mapping_v1(
            11,
            &run,
            2,
            3,
            GeneticOperatorIdentityV1::ParentA,
            4,
        )
        .expect("valid base address");
        let changed = checked_philox_counter_mapping_v1(
            11,
            &run,
            2,
            3,
            GeneticOperatorIdentityV1::ParentB,
            4,
        )
        .expect("valid changed address");
        assert_ne!(base, changed);
    }

    #[test]
    fn rejection_attempts_stay_inside_their_decision_slot() {
        assert_eq!(checked_philox_rejection_draw_index_v1(0, 0), 0);
        assert_eq!(
            checked_philox_rejection_draw_index_v1(7, 11),
            (7_u64 << 32) | 11
        );
        assert!(
            checked_philox_rejection_draw_index_v1(7, u32::MAX)
                < checked_philox_rejection_draw_index_v1(8, 0)
        );
    }
}
