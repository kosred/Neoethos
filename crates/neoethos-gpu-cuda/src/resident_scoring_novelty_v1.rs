//! Opaque, same-stream CUDA scoring and novelty authority.
//!
//! V1 ports the current Search scoring formulas without changing their branch
//! order or constants. Its output remains `ResearchOnly`: CPU/GPU golden parity is required before strict full-discovery authority, especially for `sqrt` and
//! `log` in the build-bound Risky objective.

use sha2::{Digest, Sha256};
use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;

pub const RESIDENT_SCORING_NOVELTY_SEMANTICS_V1: &str = concat!(
    "neoethos.resident-scoring-novelty.v1;",
    "scoring-version-5;propfirm-ga-fitness-v4-or-risky-growth-v5;",
    "candidate-ordered-jaccard-set-over-fixed-stride-indices;",
    "finite-min-max-normalization;stable-u64-score-key-and-gene-id-tie;",
    "same-admitted-stream;preowned-events;device-fault-invalidates-seal;",
    "cuda-cccl-toolkit-native-device-build-and-math-flags-bound;",
    "research-only;not-promotion-eligible"
);

const PROP_FIRM_SCORING_SEMANTICS_V1: &str = concat!(
    "neoethos.ga-fitness.propfirm.v4;",
    "metrics-0-1-3-4-5-7-8-9-10;zero-trades-minus-100;",
    "activity-30;monthly-hit-0.45;net-20000x0.15;",
    "sharpe-0.10;consistency-0.10;pf-piecewise;wr-0.10;",
    "drawdown-x15-cap5;daily-drawdown-x10"
);

const RISKY_GROWTH_SCORING_SEMANTICS_V1: &str = concat!(
    "neoethos.ga-fitness-growth.risky.v5;",
    "metrics-0-1-4-5-8;zero-trades-minus-100;",
    "p-cap-0.99;pf-cap-10;half-kelly-cap-0.25;",
    "ordered-log-growth-x10;below-edge-gradient"
);

const NOVELTY_SEMANTICS_V1: &str = concat!(
    "neoethos.novelty.mean-jaccard.v1;",
    "set-of-indicator-indices;candidate-order-sum;",
    "finite-fitness-min-max;max-novelty;floor-1e-9;weighted-blend"
);

const RANK_SEMANTICS_V1: &str = concat!(
    "neoethos.rank-key.v1;finite-f64-monotone-u64;",
    "descending-key;ascending-gene-identity-tie;zero-reserved-invalid"
);

pub const CUDA_MATH_FLAGS_V1: [&str; 4] = [
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

const ABI_VERSION_V1: u32 = 1;
const SCORING_VERSION_V1: u32 = 5;
const STATUS_OK_V1: i32 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ResidentScoringObjectiveV1 {
    CanonicalPropFirmGaFitnessV4 = 1,
    CanonicalRiskyGaFitnessGrowthV5 = 2,
}

#[derive(Debug)]
pub enum ResidentScoringNoveltyErrorV1 {
    CudaFeatureNotCompiled,
    InvalidPlan(&'static str),
    IdentityMismatch(&'static str),
    ArithmeticOverflow,
    CapacityUnavailable,
    RunStateViolation,
    EventIdentityMismatch,
    Native {
        operation: &'static str,
        status: i32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentScoringNoveltyRunStateV1 {
    StrictIdle,
    InFlight,
    Sealed,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScoringNoveltyArtifactClassV1 {
    ResearchOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScoringNoveltyPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RawResidentScoringNoveltyMetricRowV1 {
    pub(crate) candidate_id: u64,
    pub(crate) scenario_id: u64,
    pub(crate) values: [f64; 11],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RawResidentScoringNoveltyGeneScalarV1 {
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

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawResidentScoringNoveltyPopulationImportV1 {
    pub(crate) abi_version: u32,
    pub(crate) selected_cuda_ordinal: u32,
    pub(crate) admitted_run_stream: *mut c_void,
    pub(crate) metrics_ready_event: *mut c_void,
    pub(crate) scoring_novelty_ready_event: *mut c_void,
    pub(crate) population_lifetime_owner: *mut c_void,
    pub(crate) metric_rows_device: *const RawResidentScoringNoveltyMetricRowV1,
    pub(crate) gene_scalars_device: *const RawResidentScoringNoveltyGeneScalarV1,
    pub(crate) gene_indices_device: *const u64,
    pub(crate) expected_scenario_ids_device: *const u64,
    pub(crate) logical_population_count: u64,
    pub(crate) feature_count: u64,
    pub(crate) max_terms_per_gene: u32,
    pub(crate) reserved: u32,
    pub(crate) full_discovery_reserve_bytes: u64,
    pub(crate) cuda_device_identity_sha256: [u8; 32],
    pub(crate) primary_context_identity_sha256: [u8; 32],
    pub(crate) run_stream_identity_sha256: [u8; 32],
    pub(crate) metric_semantics_sha256: [u8; 32],
    pub(crate) gene_schema_sha256: [u8; 32],
    pub(crate) scenario_order_semantics_sha256: [u8; 32],
    pub(crate) cuda_build_manifest_sha256: [u8; 32],
    pub(crate) cuda_math_flags_sha256: [u8; 32],
    pub(crate) resident_input_content_sha256: [u8; 32],
    pub(crate) gene_content_sha256: [u8; 32],
    pub(crate) metric_content_sha256: [u8; 32],
    pub(crate) scenario_order_content_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawResidentScoringNoveltyPlanV1 {
    abi_version: u32,
    scoring_objective: u32,
    scoring_version: u32,
    reserved: u32,
    logical_population_count: u64,
    feature_count: u64,
    max_terms_per_gene: u32,
    reserved_extents: u32,
    novelty_weight_bits: u64,
    metric_semantics_sha256: [u8; 32],
    scoring_semantics_sha256: [u8; 32],
    novelty_semantics_sha256: [u8; 32],
    scenario_order_semantics_sha256: [u8; 32],
    gene_schema_sha256: [u8; 32],
    rank_semantics_sha256: [u8; 32],
    cuda_device_identity_sha256: [u8; 32],
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    cuda_math_flags_sha256: [u8; 32],
    plan_identity_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawResidentScoringNoveltyAllocationReceiptV1 {
    abi_version: u32,
    scoring_store_allocation_count: u32,
    set_bitmap_bytes: u64,
    fitness_score_bytes: u64,
    novelty_score_bytes: u64,
    decision_key_bytes: u64,
    cub_scratch_bytes: u64,
    device_control_bytes: u64,
    total_device_bytes: u64,
    same_context_free_bytes: u64,
    full_discovery_reserve_bytes: u64,
    logical_population_count: u64,
    feature_word_count: u64,
    allocation_plan_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawResidentScoringNoveltyReadyEventV1 {
    abi_version: u32,
    reserved: u32,
    event_id: u64,
    same_stream_enqueue_count: u64,
    intermediate_host_wait_count: u64,
    intermediate_readback_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RawDeviceSealV1 {
    pub(crate) abi_version: u32,
    pub(crate) valid: u32,
    pub(crate) device_fault_word: u32,
    pub(crate) reserved: u32,
    pub(crate) content_lanes: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawScoredDecisionRowsV1 {
    abi_version: u32,
    reserved: u32,
    metric_rows_device: *const RawResidentScoringNoveltyMetricRowV1,
    resident_decision_keys_device: *const u64,
    expected_scenario_ids_device: *const u64,
    device_seal: *const RawDeviceSealV1,
    scoring_novelty_ready_event: *mut c_void,
    logical_population_count: u64,
    event_id: u64,
    same_stream_enqueue_count: u64,
    intermediate_host_wait_count: u64,
    intermediate_readback_count: u64,
    metric_semantics_sha256: [u8; 32],
    scoring_semantics_sha256: [u8; 32],
    novelty_semantics_sha256: [u8; 32],
    scenario_order_semantics_sha256: [u8; 32],
    rank_semantics_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    cuda_math_flags_sha256: [u8; 32],
}

impl Default for RawScoredDecisionRowsV1 {
    fn default() -> Self {
        Self {
            abi_version: 0,
            reserved: 0,
            metric_rows_device: std::ptr::null(),
            resident_decision_keys_device: std::ptr::null(),
            expected_scenario_ids_device: std::ptr::null(),
            device_seal: std::ptr::null(),
            scoring_novelty_ready_event: std::ptr::null_mut(),
            logical_population_count: 0,
            event_id: 0,
            same_stream_enqueue_count: 0,
            intermediate_host_wait_count: 0,
            intermediate_readback_count: 0,
            metric_semantics_sha256: [0; 32],
            scoring_semantics_sha256: [0; 32],
            novelty_semantics_sha256: [0; 32],
            scenario_order_semantics_sha256: [0; 32],
            rank_semantics_sha256: [0; 32],
            cuda_build_manifest_sha256: [0; 32],
            cuda_math_flags_sha256: [0; 32],
        }
    }
}

const _: [(); 104] = [(); std::mem::size_of::<RawResidentScoringNoveltyMetricRowV1>()];
const _: [(); 72] = [(); std::mem::size_of::<RawResidentScoringNoveltyGeneScalarV1>()];
const _: [(); 488] = [(); std::mem::size_of::<RawResidentScoringNoveltyPopulationImportV1>()];
const _: [(); 432] = [(); std::mem::size_of::<RawResidentScoringNoveltyPlanV1>()];
const _: [(); 128] = [(); std::mem::size_of::<RawResidentScoringNoveltyAllocationReceiptV1>()];
const _: [(); 40] = [(); std::mem::size_of::<RawResidentScoringNoveltyReadyEventV1>()];
const _: [(); 48] = [(); std::mem::size_of::<RawDeviceSealV1>()];
const _: [(); 312] = [(); std::mem::size_of::<RawScoredDecisionRowsV1>()];

enum NativeResidentScoringNoveltyRunV1 {}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    #[link_name = "query_resident_scoring_novelty_allocation_v1"]
    fn ffi_query_resident_scoring_novelty_allocation_v1(
        import: *const RawResidentScoringNoveltyPopulationImportV1,
        plan: *const RawResidentScoringNoveltyPlanV1,
        receipt: *mut RawResidentScoringNoveltyAllocationReceiptV1,
    ) -> i32;
    #[link_name = "create_resident_scoring_novelty_run_v1"]
    fn ffi_create_resident_scoring_novelty_run_v1(
        import: *const RawResidentScoringNoveltyPopulationImportV1,
        plan: *const RawResidentScoringNoveltyPlanV1,
        receipt: *const RawResidentScoringNoveltyAllocationReceiptV1,
        run: *mut *mut NativeResidentScoringNoveltyRunV1,
    ) -> i32;
    #[link_name = "enqueue_and_seal_resident_scoring_novelty_v1"]
    fn ffi_enqueue_and_seal_resident_scoring_novelty_v1(
        run: *mut NativeResidentScoringNoveltyRunV1,
        output: *mut RawScoredDecisionRowsV1,
        ready: *mut RawResidentScoringNoveltyReadyEventV1,
    ) -> i32;
    #[link_name = "enqueue_resident_scoring_novelty_release_v1"]
    fn ffi_enqueue_resident_scoring_novelty_release_v1(
        run: *mut NativeResidentScoringNoveltyRunV1,
    ) -> i32;
}

/// One private import minted only by the existing admitted population owner.
pub(crate) struct ResidentScoringNoveltyPopulationImportV1 {
    raw: RawResidentScoringNoveltyPopulationImportV1,
    lifetime_owner: Option<Box<dyn Any>>,
}

impl ResidentScoringNoveltyPopulationImportV1 {
    /// # Safety
    ///
    /// Every pointer, event and identity must belong to the retained population
    /// session in the selected primary context. The two events must remain live
    /// and distinct through downstream same-stream consumption.
    pub(crate) unsafe fn from_population_session_parts_v1<T: Any>(
        raw: RawResidentScoringNoveltyPopulationImportV1,
        lifetime_owner: T,
    ) -> Result<Self, ResidentScoringNoveltyErrorV1> {
        if raw.abi_version != ABI_VERSION_V1
            || raw.selected_cuda_ordinal == u32::MAX
            || raw.admitted_run_stream.is_null()
            || raw.metrics_ready_event.is_null()
            || raw.scoring_novelty_ready_event.is_null()
            || raw.metrics_ready_event == raw.scoring_novelty_ready_event
            || raw.population_lifetime_owner.is_null()
            || raw.metric_rows_device.is_null()
            || raw.gene_scalars_device.is_null()
            || raw.gene_indices_device.is_null()
            || raw.expected_scenario_ids_device.is_null()
            || raw.logical_population_count == 0
            || raw.feature_count == 0
            || raw.max_terms_per_gene == 0
            || raw.full_discovery_reserve_bytes == 0
            || import_identity_missing_v1(&raw)
        {
            return Err(ResidentScoringNoveltyErrorV1::InvalidPlan(
                "resident scoring import is incomplete",
            ));
        }
        Ok(Self {
            raw,
            lifetime_owner: Some(Box::new(lifetime_owner)),
        })
    }
}

impl Drop for ResidentScoringNoveltyPopulationImportV1 {
    fn drop(&mut self) {
        if let Some(owner) = self.lifetime_owner.take() {
            std::mem::forget(owner);
        }
    }
}

pub(crate) struct ResidentScoringNoveltyPlanAuthorityInputV1 {
    pub(crate) scoring_objective: ResidentScoringObjectiveV1,
    pub(crate) logical_population_count: usize,
    pub(crate) feature_count: usize,
    pub(crate) max_terms_per_gene: usize,
    pub(crate) novelty_weight_bits: u64,
    pub(crate) metric_semantics_sha256: [u8; 32],
    pub(crate) scoring_semantics_sha256: [u8; 32],
    pub(crate) novelty_semantics_sha256: [u8; 32],
    pub(crate) scenario_order_semantics_sha256: [u8; 32],
    pub(crate) gene_schema_sha256: [u8; 32],
    pub(crate) rank_semantics_sha256: [u8; 32],
    pub(crate) cuda_device_identity_sha256: [u8; 32],
    pub(crate) primary_context_identity_sha256: [u8; 32],
    pub(crate) run_stream_identity_sha256: [u8; 32],
    pub(crate) cuda_build_manifest_sha256: [u8; 32],
    pub(crate) cuda_math_flags_sha256: [u8; 32],
}

pub struct SealedResidentScoringNoveltyPlanV1 {
    raw: RawResidentScoringNoveltyPlanV1,
    plan_identity_sha256: [u8; 32],
}

pub(crate) fn seal_resident_scoring_novelty_plan_v1(
    input: ResidentScoringNoveltyPlanAuthorityInputV1,
) -> Result<SealedResidentScoringNoveltyPlanV1, ResidentScoringNoveltyErrorV1> {
    let novelty_weight = f64::from_bits(input.novelty_weight_bits);
    if input.logical_population_count == 0
        || input.logical_population_count > i32::MAX as usize
        || input.feature_count == 0
        || input.max_terms_per_gene == 0
        || input.max_terms_per_gene > input.feature_count
        || !novelty_weight.is_finite()
        || !(0.0..=1.0).contains(&novelty_weight)
        || identity_is_zero_v1(&input.metric_semantics_sha256)
        || identity_is_zero_v1(&input.scoring_semantics_sha256)
        || identity_is_zero_v1(&input.novelty_semantics_sha256)
        || identity_is_zero_v1(&input.scenario_order_semantics_sha256)
        || identity_is_zero_v1(&input.gene_schema_sha256)
        || identity_is_zero_v1(&input.rank_semantics_sha256)
        || identity_is_zero_v1(&input.cuda_device_identity_sha256)
        || identity_is_zero_v1(&input.primary_context_identity_sha256)
        || identity_is_zero_v1(&input.run_stream_identity_sha256)
        || identity_is_zero_v1(&input.cuda_build_manifest_sha256)
        || identity_is_zero_v1(&input.cuda_math_flags_sha256)
    {
        return Err(ResidentScoringNoveltyErrorV1::InvalidPlan(
            "resident scoring plan is incomplete",
        ));
    }
    let expected_scoring = scoring_semantics_sha256_v1(input.scoring_objective);
    let expected_novelty = sha256_v1(&[NOVELTY_SEMANTICS_V1.as_bytes()]);
    let expected_rank = sha256_v1(&[RANK_SEMANTICS_V1.as_bytes()]);
    let expected_math = cuda_math_flags_sha256_v1();
    if input.scoring_semantics_sha256 != expected_scoring
        || input.novelty_semantics_sha256 != expected_novelty
        || input.rank_semantics_sha256 != expected_rank
        || input.cuda_math_flags_sha256 != expected_math
    {
        return Err(ResidentScoringNoveltyErrorV1::IdentityMismatch(
            "scoring, novelty, rank or CUDA math semantics",
        ));
    }
    let mut raw = RawResidentScoringNoveltyPlanV1 {
        abi_version: ABI_VERSION_V1,
        scoring_objective: input.scoring_objective as u32,
        scoring_version: SCORING_VERSION_V1,
        reserved: 0,
        logical_population_count: checked_u64_v1(input.logical_population_count)?,
        feature_count: checked_u64_v1(input.feature_count)?,
        max_terms_per_gene: checked_u32_v1(input.max_terms_per_gene)?,
        reserved_extents: 0,
        novelty_weight_bits: input.novelty_weight_bits,
        metric_semantics_sha256: input.metric_semantics_sha256,
        scoring_semantics_sha256: input.scoring_semantics_sha256,
        novelty_semantics_sha256: input.novelty_semantics_sha256,
        scenario_order_semantics_sha256: input.scenario_order_semantics_sha256,
        gene_schema_sha256: input.gene_schema_sha256,
        rank_semantics_sha256: input.rank_semantics_sha256,
        cuda_device_identity_sha256: input.cuda_device_identity_sha256,
        primary_context_identity_sha256: input.primary_context_identity_sha256,
        run_stream_identity_sha256: input.run_stream_identity_sha256,
        cuda_build_manifest_sha256: input.cuda_build_manifest_sha256,
        cuda_math_flags_sha256: input.cuda_math_flags_sha256,
        plan_identity_sha256: [0; 32],
    };
    let plan_identity_sha256 = hash_raw_plan_v1(&raw);
    raw.plan_identity_sha256 = plan_identity_sha256;
    Ok(SealedResidentScoringNoveltyPlanV1 {
        raw,
        plan_identity_sha256,
    })
}

pub struct ActualResidentScoringNoveltyAllocationPlanV1 {
    raw: RawResidentScoringNoveltyAllocationReceiptV1,
}

pub(crate) fn query_actual_resident_scoring_novelty_allocation_plan_v1(
    import: &ResidentScoringNoveltyPopulationImportV1,
    plan: &SealedResidentScoringNoveltyPlanV1,
) -> Result<ActualResidentScoringNoveltyAllocationPlanV1, ResidentScoringNoveltyErrorV1> {
    validate_import_plan_identity_v1(import, plan)?;
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (import, plan);
        Err(ResidentScoringNoveltyErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        let mut raw = RawResidentScoringNoveltyAllocationReceiptV1::default();
        // SAFETY: both sealed inputs and the out-receipt remain live for the call.
        let status = unsafe {
            ffi_query_resident_scoring_novelty_allocation_v1(&import.raw, &plan.raw, &mut raw)
        };
        require_native_ok_v1("query_resident_scoring_novelty_allocation_v1", status)?;
        validate_allocation_receipt_v1(import, plan, &raw)?;
        Ok(ActualResidentScoringNoveltyAllocationPlanV1 { raw })
    }
}

#[must_use = "resident scoring/novelty work must be consumed by the next device stage"]
pub struct ResidentScoringNoveltyDeviceRunV1 {
    native: NonNull<NativeResidentScoringNoveltyRunV1>,
    population_import: Option<ResidentScoringNoveltyPopulationImportV1>,
    state: ResidentScoringNoveltyRunStateV1,
    selected_cuda_ordinal: u32,
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    cuda_math_flags_sha256: [u8; 32],
    plan: SealedResidentScoringNoveltyPlanV1,
    allocation: ActualResidentScoringNoveltyAllocationPlanV1,
}

pub(crate) fn bind_resident_scoring_novelty_run_v1(
    import: ResidentScoringNoveltyPopulationImportV1,
    plan: SealedResidentScoringNoveltyPlanV1,
    allocation: ActualResidentScoringNoveltyAllocationPlanV1,
) -> Result<ResidentScoringNoveltyDeviceRunV1, ResidentScoringNoveltyErrorV1> {
    validate_import_plan_allocation_identity_v1(&import, &plan, &allocation)?;
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (import, plan, allocation);
        Err(ResidentScoringNoveltyErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        let mut native = std::ptr::null_mut();
        // SAFETY: the move-only import retains every borrowed device allocation,
        // event, context and stream through the native run lifetime.
        let status = unsafe {
            ffi_create_resident_scoring_novelty_run_v1(
                &import.raw,
                &plan.raw,
                &allocation.raw,
                &mut native,
            )
        };
        require_native_ok_v1("create_resident_scoring_novelty_run_v1", status)?;
        let native = NonNull::new(native).ok_or(ResidentScoringNoveltyErrorV1::Native {
            operation: "create_resident_scoring_novelty_run_v1",
            status,
        })?;
        Ok(ResidentScoringNoveltyDeviceRunV1 {
            native,
            state: ResidentScoringNoveltyRunStateV1::StrictIdle,
            selected_cuda_ordinal: import.raw.selected_cuda_ordinal,
            primary_context_identity_sha256: import.raw.primary_context_identity_sha256,
            run_stream_identity_sha256: import.raw.run_stream_identity_sha256,
            cuda_build_manifest_sha256: import.raw.cuda_build_manifest_sha256,
            cuda_math_flags_sha256: import.raw.cuda_math_flags_sha256,
            population_import: Some(import),
            plan,
            allocation,
        })
    }
}

pub struct SealedResidentScoringNoveltyDecisionRowsV1 {
    run: Option<ResidentScoringNoveltyDeviceRunV1>,
    raw: RawScoredDecisionRowsV1,
    artifact_class: ScoringNoveltyArtifactClassV1,
    promotion_eligibility: ScoringNoveltyPromotionEligibilityV1,
}

pub(crate) fn enqueue_and_seal_resident_scoring_novelty_v1(
    mut run: ResidentScoringNoveltyDeviceRunV1,
) -> Result<SealedResidentScoringNoveltyDecisionRowsV1, ResidentScoringNoveltyErrorV1> {
    if run.state != ResidentScoringNoveltyRunStateV1::StrictIdle {
        return Err(ResidentScoringNoveltyErrorV1::RunStateViolation);
    }
    run.state = ResidentScoringNoveltyRunStateV1::InFlight;
    let mut raw = RawScoredDecisionRowsV1::default();
    let mut ready = RawResidentScoringNoveltyReadyEventV1::default();
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (raw, ready);
        run.state = ResidentScoringNoveltyRunStateV1::Poisoned;
        Err(ResidentScoringNoveltyErrorV1::CudaFeatureNotCompiled)
    }
    #[cfg(feature = "cuda")]
    {
        // SAFETY: state becomes InFlight before the first fallible launch. Any
        // ambiguity therefore reaches the leak-only Drop boundary.
        let status = unsafe {
            ffi_enqueue_and_seal_resident_scoring_novelty_v1(
                run.native.as_ptr(),
                &mut raw,
                &mut ready,
            )
        };
        if status != STATUS_OK_V1 {
            run.state = ResidentScoringNoveltyRunStateV1::Poisoned;
            return Err(native_error_v1(
                "enqueue_and_seal_resident_scoring_novelty_v1",
                status,
            ));
        }
        validate_ready_and_output_v1(&run, &ready, &raw)?;
        let final_compact_readback_count = 0_u64;
        debug_assert!(final_compact_readback_count == 0);
        run.state = ResidentScoringNoveltyRunStateV1::Sealed;
        Ok(SealedResidentScoringNoveltyDecisionRowsV1 {
            run: Some(run),
            raw,
            artifact_class: ScoringNoveltyArtifactClassV1::ResearchOnly,
            promotion_eligibility: ScoringNoveltyPromotionEligibilityV1::NotPromotionEligible,
        })
    }
}

impl Drop for ResidentScoringNoveltyDeviceRunV1 {
    fn drop(&mut self) {
        leak_live_native_scoring_novelty_run_v1(self);
    }
}

fn leak_live_native_scoring_novelty_run_v1(run: &mut ResidentScoringNoveltyDeviceRunV1) {
    let _ = run.native;
    if let Some(import) = run.population_import.take() {
        std::mem::forget(import);
    }
}

fn validate_import_plan_identity_v1(
    import: &ResidentScoringNoveltyPopulationImportV1,
    plan: &SealedResidentScoringNoveltyPlanV1,
) -> Result<(), ResidentScoringNoveltyErrorV1> {
    if import.raw.logical_population_count != plan.raw.logical_population_count
        || import.raw.feature_count != plan.raw.feature_count
        || import.raw.max_terms_per_gene != plan.raw.max_terms_per_gene
        || import.raw.metric_semantics_sha256 != plan.raw.metric_semantics_sha256
        || import.raw.gene_schema_sha256 != plan.raw.gene_schema_sha256
        || import.raw.scenario_order_semantics_sha256 != plan.raw.scenario_order_semantics_sha256
        || import.raw.cuda_device_identity_sha256 != plan.raw.cuda_device_identity_sha256
        || import.raw.primary_context_identity_sha256 != plan.raw.primary_context_identity_sha256
        || import.raw.run_stream_identity_sha256 != plan.raw.run_stream_identity_sha256
        || import.raw.cuda_build_manifest_sha256 != plan.raw.cuda_build_manifest_sha256
        || import.raw.cuda_math_flags_sha256 != plan.raw.cuda_math_flags_sha256
    {
        return Err(ResidentScoringNoveltyErrorV1::IdentityMismatch(
            "population import and scoring plan",
        ));
    }
    Ok(())
}

fn validate_allocation_receipt_v1(
    import: &ResidentScoringNoveltyPopulationImportV1,
    plan: &SealedResidentScoringNoveltyPlanV1,
    raw: &RawResidentScoringNoveltyAllocationReceiptV1,
) -> Result<(), ResidentScoringNoveltyErrorV1> {
    if raw.abi_version != ABI_VERSION_V1
        || raw.scoring_store_allocation_count != 1
        || raw.logical_population_count != plan.raw.logical_population_count
        || raw.feature_word_count == 0
        || raw.full_discovery_reserve_bytes != import.raw.full_discovery_reserve_bytes
        || raw.allocation_plan_sha256 != plan.plan_identity_sha256
    {
        return Err(ResidentScoringNoveltyErrorV1::IdentityMismatch(
            "scoring allocation receipt",
        ));
    }
    let charged = [
        raw.set_bitmap_bytes,
        raw.fitness_score_bytes,
        raw.novelty_score_bytes,
        raw.decision_key_bytes,
        raw.cub_scratch_bytes,
        raw.device_control_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, u64::checked_add)
    .ok_or(ResidentScoringNoveltyErrorV1::ArithmeticOverflow)?;
    let reusable = raw
        .same_context_free_bytes
        .checked_sub(raw.full_discovery_reserve_bytes)
        .ok_or(ResidentScoringNoveltyErrorV1::CapacityUnavailable)?;
    if charged != raw.total_device_bytes || raw.total_device_bytes > reusable {
        return Err(ResidentScoringNoveltyErrorV1::CapacityUnavailable);
    }
    Ok(())
}

fn validate_import_plan_allocation_identity_v1(
    import: &ResidentScoringNoveltyPopulationImportV1,
    plan: &SealedResidentScoringNoveltyPlanV1,
    allocation: &ActualResidentScoringNoveltyAllocationPlanV1,
) -> Result<(), ResidentScoringNoveltyErrorV1> {
    validate_import_plan_identity_v1(import, plan)?;
    if allocation.raw.allocation_plan_sha256 != plan.plan_identity_sha256 {
        return Err(ResidentScoringNoveltyErrorV1::IdentityMismatch(
            "scoring plan and allocation",
        ));
    }
    Ok(())
}

fn validate_ready_and_output_v1(
    run: &ResidentScoringNoveltyDeviceRunV1,
    ready: &RawResidentScoringNoveltyReadyEventV1,
    raw: &RawScoredDecisionRowsV1,
) -> Result<(), ResidentScoringNoveltyErrorV1> {
    if ready.abi_version != ABI_VERSION_V1
        || ready.event_id == 0
        || ready.intermediate_host_wait_count != 0
        || ready.intermediate_readback_count != 0
        || raw.abi_version != ABI_VERSION_V1
        || raw.event_id != ready.event_id
        || raw.metric_rows_device.is_null()
        || raw.resident_decision_keys_device.is_null()
        || raw.expected_scenario_ids_device.is_null()
        || raw.device_seal.is_null()
        || raw.scoring_novelty_ready_event.is_null()
        || raw.logical_population_count != run.plan.raw.logical_population_count
        || raw.intermediate_host_wait_count != 0
        || raw.intermediate_readback_count != 0
        || raw.metric_semantics_sha256 != run.plan.raw.metric_semantics_sha256
        || raw.scoring_semantics_sha256 != run.plan.raw.scoring_semantics_sha256
        || raw.novelty_semantics_sha256 != run.plan.raw.novelty_semantics_sha256
        || raw.scenario_order_semantics_sha256 != run.plan.raw.scenario_order_semantics_sha256
        || raw.rank_semantics_sha256 != run.plan.raw.rank_semantics_sha256
        || raw.cuda_build_manifest_sha256 != run.cuda_build_manifest_sha256
        || raw.cuda_math_flags_sha256 != run.cuda_math_flags_sha256
    {
        return Err(ResidentScoringNoveltyErrorV1::EventIdentityMismatch);
    }
    Ok(())
}

fn import_identity_missing_v1(raw: &RawResidentScoringNoveltyPopulationImportV1) -> bool {
    [
        &raw.cuda_device_identity_sha256,
        &raw.primary_context_identity_sha256,
        &raw.run_stream_identity_sha256,
        &raw.metric_semantics_sha256,
        &raw.gene_schema_sha256,
        &raw.scenario_order_semantics_sha256,
        &raw.cuda_build_manifest_sha256,
        &raw.cuda_math_flags_sha256,
        &raw.resident_input_content_sha256,
        &raw.gene_content_sha256,
        &raw.metric_content_sha256,
        &raw.scenario_order_content_sha256,
    ]
    .into_iter()
    .any(identity_is_zero_v1)
}

fn scoring_semantics_sha256_v1(objective: ResidentScoringObjectiveV1) -> [u8; 32] {
    match objective {
        ResidentScoringObjectiveV1::CanonicalPropFirmGaFitnessV4 => {
            sha256_v1(&[PROP_FIRM_SCORING_SEMANTICS_V1.as_bytes()])
        }
        ResidentScoringObjectiveV1::CanonicalRiskyGaFitnessGrowthV5 => {
            sha256_v1(&[RISKY_GROWTH_SCORING_SEMANTICS_V1.as_bytes()])
        }
    }
}

fn cuda_math_flags_sha256_v1() -> [u8; 32] {
    let fields: Vec<&[u8]> = CUDA_MATH_FLAGS_V1
        .iter()
        .map(|flag| flag.as_bytes())
        .collect();
    sha256_v1(&fields)
}

fn hash_raw_plan_v1(plan: &RawResidentScoringNoveltyPlanV1) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.resident-scoring-novelty-plan.v1\0");
    hasher.update(plan.abi_version.to_le_bytes());
    hasher.update(plan.scoring_objective.to_le_bytes());
    hasher.update(plan.scoring_version.to_le_bytes());
    hasher.update(plan.reserved.to_le_bytes());
    hasher.update(plan.logical_population_count.to_le_bytes());
    hasher.update(plan.feature_count.to_le_bytes());
    hasher.update(plan.max_terms_per_gene.to_le_bytes());
    hasher.update(plan.reserved_extents.to_le_bytes());
    hasher.update(plan.novelty_weight_bits.to_le_bytes());
    hasher.update(plan.metric_semantics_sha256);
    hasher.update(plan.scoring_semantics_sha256);
    hasher.update(plan.novelty_semantics_sha256);
    hasher.update(plan.scenario_order_semantics_sha256);
    hasher.update(plan.gene_schema_sha256);
    hasher.update(plan.rank_semantics_sha256);
    hasher.update(plan.cuda_device_identity_sha256);
    hasher.update(plan.primary_context_identity_sha256);
    hasher.update(plan.run_stream_identity_sha256);
    hasher.update(plan.cuda_build_manifest_sha256);
    hasher.update(plan.cuda_math_flags_sha256);
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

fn checked_u32_v1(value: usize) -> Result<u32, ResidentScoringNoveltyErrorV1> {
    u32::try_from(value).map_err(|_| ResidentScoringNoveltyErrorV1::ArithmeticOverflow)
}

fn checked_u64_v1(value: usize) -> Result<u64, ResidentScoringNoveltyErrorV1> {
    u64::try_from(value).map_err(|_| ResidentScoringNoveltyErrorV1::ArithmeticOverflow)
}

fn require_native_ok_v1(
    operation: &'static str,
    status: i32,
) -> Result<(), ResidentScoringNoveltyErrorV1> {
    if status == STATUS_OK_V1 {
        Ok(())
    } else {
        Err(native_error_v1(operation, status))
    }
}

fn native_error_v1(operation: &'static str, status: i32) -> ResidentScoringNoveltyErrorV1 {
    ResidentScoringNoveltyErrorV1::Native { operation, status }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_f64_reference_is_monotone_and_reserves_zero_for_invalid() {
        fn ordered_f64_decision_key_reference_v1(value: f64) -> u64 {
            if !value.is_finite() {
                return 0;
            }
            let value = if value == 0.0 { 0.0 } else { value };
            let bits = value.to_bits();
            let key = if bits >> 63 == 0 {
                bits ^ (1_u64 << 63)
            } else {
                !bits
            };
            key.max(1)
        }

        let values = [-100.0, -1.0, -0.0, 0.0, 1.0, 100.0];
        let keys: Vec<u64> = values
            .into_iter()
            .map(ordered_f64_decision_key_reference_v1)
            .collect();
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(keys[2], keys[3], "signed zero is one canonical tie");
        assert_eq!(ordered_f64_decision_key_reference_v1(f64::NAN), 0);
    }

    #[test]
    fn canonical_semantic_hashes_are_distinct() {
        assert_ne!(
            scoring_semantics_sha256_v1(ResidentScoringObjectiveV1::CanonicalPropFirmGaFitnessV4),
            scoring_semantics_sha256_v1(
                ResidentScoringObjectiveV1::CanonicalRiskyGaFitnessGrowthV5
            )
        );
        assert!(!identity_is_zero_v1(&cuda_math_flags_sha256_v1()));
    }
}
