//! Move-only CUDA scoring admission for one bounded resident Search generation.

use crate::resident_generation_v1::{RawAllocationReceiptV1, SealedResidentGenerationPlanV1};
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::ptr::NonNull;
use thiserror::Error;

pub(crate) const RESIDENT_SCORING_SEMANTICS_PROPFIRM_V2: &str = concat!(
    "neoethos.resident-scoring.v2;objective=prop-firm-v4;",
    "all-eleven-metrics-finite;finite-objective;raw-objective;",
    "positive-zero-canonical;cuda-build-and-math-bound;no-host-decision"
);
pub(crate) const RESIDENT_SCORING_SEMANTICS_RISKY_V2: &str = concat!(
    "neoethos.resident-scoring.v2;objective=risky-growth-v5;",
    "all-eleven-metrics-finite;finite-objective;raw-objective;",
    "positive-zero-canonical;cuda-build-and-math-bound;no-host-decision"
);
pub(crate) const RESIDENT_NOVELTY_DISABLED_SEMANTICS_V2: &str = concat!(
    "neoethos.resident-novelty-disabled.v2;novelty-weight-bits=positive-zero;",
    "no-current-only-mean-jaccard;no-knn-without-explicit-k;no-archive"
);
pub(crate) const RESIDENT_RANK_SEMANTICS_V2: &str = concat!(
    "neoethos.resident-rank.v2;stable-lsd;score-desc;gene-identity-asc;",
    "population-ordinal-asc;ordered-f64;positive-zero-canonical;",
    "defined-sentinel-cub-inputs;fault-gated-semantic-commit"
);
pub(crate) const RESIDENT_CUDA_MATH_SEMANTICS_V2: &str =
    "neoethos.cuda-math.v2;fmad=false;ftz=false;prec-div=true;prec-sqrt=true";

const SCORING_ABI_V1: u32 = 1;
const SCORING_VERSION_V1: u32 = 5;
const STATUS_OK: i32 = 0;
const STATUS_ASYNC_FREE_OUTCOME_UNKNOWN: i32 = -48;
const STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN: i32 = -49;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum ResidentScoringObjectiveV2 {
    PropFirmV4 = 1,
    #[allow(dead_code)] // Constructed by the real-device dual-objective oracle.
    RiskyGrowthV5 = 2,
}

#[derive(Debug, Error)]
pub(crate) enum ResidentScoringV2Error {
    #[error("resident scoring V2 novelty weight must have the exact +0.0 bit pattern")]
    InvalidNoveltyWeight,
    #[error("invalid resident scoring V2 plan: {0}")]
    InvalidPlan(&'static str),
    #[error("resident scoring V2 allocation arithmetic overflowed")]
    ArithmeticOverflow,
    #[error(
        "resident scoring V2 native operation {operation} reported an unknown stream-ordered free outcome; the pointer identity is retired and a possible allocation leak is deliberate"
    )]
    AsyncFreeOutcomeUnknownDeliberateLeak { operation: &'static str },
    #[error(
        "resident scoring V2 native operation {operation} reported an unknown stream-ordered allocation outcome; no device identity is available for reuse or cleanup"
    )]
    AsyncAllocationOutcomeUnknownDeliberateLeak { operation: &'static str },
    #[error("resident scoring V2 native operation {operation} failed with status {status}")]
    Native {
        operation: &'static str,
        status: i32,
    },
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawResidentScoringPlanV2 {
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
pub(crate) struct RawResidentScoringAllocationReceiptV2 {
    abi_version: u32,
    scoring_store_allocation_count: u32,
    set_bitmap_bytes: u64,
    fitness_score_bytes: u64,
    novelty_score_bytes: u64,
    decision_key_bytes: u64,
    cub_scratch_bytes: u64,
    device_control_bytes: u64,
    pub(crate) total_device_bytes: u64,
    pub(crate) same_context_free_bytes: u64,
    pub(crate) full_discovery_reserve_bytes: u64,
    logical_population_count: u64,
    feature_word_count: u64,
    pub(crate) allocation_plan_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawResidentSearchRuntimeFactsV2 {
    pub(crate) abi_version: u32,
    pub(crate) selected_cuda_ordinal: u32,
    pub(crate) run_admission_ordinal: u64,
    pub(crate) device_uuid: [u8; 16],
    pub(crate) compute_capability_major: u32,
    pub(crate) compute_capability_minor: u32,
    pub(crate) primary_context_id: u64,
    pub(crate) run_stream_id: u64,
    pub(crate) admitted_primary_context: *mut c_void,
    pub(crate) admitted_run_stream: *mut c_void,
    pub(crate) admitted_memory_pool: *mut c_void,
    pub(crate) pool_location_type: u32,
    pub(crate) pool_location_id: i32,
    pub(crate) pool_allocation_type: u32,
    pub(crate) pool_handle_types: u32,
    pub(crate) active_pool_is_default: u32,
    pub(crate) reserved: u32,
    pub(crate) pool_reserved_current_bytes: u64,
    pub(crate) pool_used_current_bytes: u64,
    pub(crate) allocator_context_reserve_bytes: u64,
    pub(crate) run_stream_process_token: [u8; 32],
}

impl Default for RawResidentSearchRuntimeFactsV2 {
    fn default() -> Self {
        Self {
            abi_version: 0,
            selected_cuda_ordinal: 0,
            run_admission_ordinal: 0,
            device_uuid: [0; 16],
            compute_capability_major: 0,
            compute_capability_minor: 0,
            primary_context_id: 0,
            run_stream_id: 0,
            admitted_primary_context: std::ptr::null_mut(),
            admitted_run_stream: std::ptr::null_mut(),
            admitted_memory_pool: std::ptr::null_mut(),
            pool_location_type: 0,
            pool_location_id: 0,
            pool_allocation_type: 0,
            pool_handle_types: 0,
            active_pool_is_default: 0,
            reserved: 0,
            pool_reserved_current_bytes: 0,
            pool_used_current_bytes: 0,
            allocator_context_reserve_bytes: 0,
            run_stream_process_token: [0; 32],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RawResidentSearchCombinedAdmissionV2 {
    pub(crate) abi_version: u32,
    pub(crate) flags: u32,
    pub(crate) free_memory_snapshot_count: u32,
    pub(crate) generation_allocation_count: u32,
    pub(crate) scoring_allocation_count: u32,
    pub(crate) terminal_host_allocation_count: u32,
    pub(crate) terminal_host_receipt_bytes: u64,
    pub(crate) same_context_free_bytes: u64,
    pub(crate) same_context_total_bytes: u64,
    pub(crate) full_discovery_reserve_bytes: u64,
    pub(crate) generation_device_bytes: u64,
    pub(crate) scoring_device_bytes: u64,
    pub(crate) total_device_bytes: u64,
    pub(crate) pool_reserved_current_bytes: u64,
    pub(crate) pool_used_current_bytes: u64,
    pub(crate) runtime: RawResidentSearchRuntimeFactsV2,
    pub(crate) generation: RawAllocationReceiptV1,
    pub(crate) scoring: RawResidentScoringAllocationReceiptV2,
    pub(crate) receipt_identity_sha256: [u8; 32],
}

pub(crate) enum NativeResidentScoringRunV2 {}

const _: [(); 432] = [(); std::mem::size_of::<RawResidentScoringPlanV2>()];
const _: [(); 128] = [(); std::mem::size_of::<RawResidentScoringAllocationReceiptV2>()];
const _: [(); 160] = [(); std::mem::size_of::<RawResidentSearchRuntimeFactsV2>()];
const _: [(); 592] = [(); std::mem::size_of::<RawResidentSearchCombinedAdmissionV2>()];

unsafe extern "C" {
    #[allow(dead_code)] // Roots the normal-CUDA core ABI; the session wrapper owns admission.
    fn query_resident_scoring_admission_v2(
        admission: *const c_void,
        plan: *const RawResidentScoringPlanV2,
        receipt: *mut RawResidentScoringAllocationReceiptV2,
    ) -> i32;
    #[allow(dead_code)] // Roots the normal-CUDA core ABI; the session wrapper owns admission.
    fn create_unbound_resident_scoring_run_v2(
        admission: *const c_void,
        plan: *const RawResidentScoringPlanV2,
        receipt: *const RawResidentScoringAllocationReceiptV2,
        run: *mut *mut NativeResidentScoringRunV2,
    ) -> i32;
    #[allow(dead_code)] // Called by the native composite so Rust never exposes raw device input.
    fn bind_and_seal_resident_scoring_v2(
        run: *mut NativeResidentScoringRunV2,
        population: *const c_void,
        output: *mut c_void,
        ready: *mut c_void,
    ) -> i32;
    #[allow(dead_code)] // Roots the core release ABI used by the session-owned native wrapper.
    fn enqueue_resident_scoring_release_v2(run: *mut NativeResidentScoringRunV2) -> i32;
    fn neoethos_gpu_cuda_population_release_resident_scoring_run_v2(
        session: *mut c_void,
        run: *mut NativeResidentScoringRunV2,
    ) -> i32;
}

pub(crate) struct SealedResidentScoringPlanV2 {
    raw: RawResidentScoringPlanV2,
}

impl SealedResidentScoringPlanV2 {
    pub(crate) const fn raw_v2(&self) -> &RawResidentScoringPlanV2 {
        &self.raw
    }
}

#[allow(dead_code)] // Retained as the immutable proof for the bounded Search owner.
pub(crate) struct SealedResidentSearchAdmissionV2 {
    pub(crate) generation_device_bytes: u64,
    pub(crate) scoring_device_bytes: u64,
    pub(crate) total_device_bytes: u64,
    pub(crate) same_context_free_bytes: u64,
    pub(crate) full_discovery_reserve_bytes: u64,
    pub(crate) generation_allocation_plan_sha256: [u8; 32],
    pub(crate) scoring_allocation_plan_sha256: [u8; 32],
    pub(crate) receipt_identity_sha256: [u8; 32],
    pub(crate) raw: RawResidentSearchCombinedAdmissionV2,
}

pub(crate) struct ResidentScoringRunV2 {
    session: NonNull<c_void>,
    native: Option<NonNull<NativeResidentScoringRunV2>>,
    state: ResidentScoringStateV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentScoringStateV2 {
    Unbound,
    Bound,
    Poisoned,
    Released,
}

impl ResidentScoringRunV2 {
    pub(crate) fn from_combined_v2(
        session: *mut c_void,
        native: *mut NativeResidentScoringRunV2,
    ) -> Result<Self, ResidentScoringV2Error> {
        Ok(Self {
            session: NonNull::new(session).ok_or(ResidentScoringV2Error::InvalidPlan(
                "population session handle is null",
            ))?,
            native: Some(
                NonNull::new(native).ok_or(ResidentScoringV2Error::InvalidPlan(
                    "combined admission returned a null scoring owner",
                ))?,
            ),
            state: ResidentScoringStateV2::Unbound,
        })
    }

    pub(crate) const fn native_v2(&self) -> Option<NonNull<NativeResidentScoringRunV2>> {
        self.native
    }

    pub(crate) fn mark_bound_v2(&mut self) {
        self.state = ResidentScoringStateV2::Bound;
    }

    pub(crate) fn poison_v2(&mut self) {
        self.state = ResidentScoringStateV2::Poisoned;
    }

    pub(crate) fn release_v2(&mut self) -> Result<(), ResidentScoringV2Error> {
        let Some(native) = self.native else {
            return Ok(());
        };
        // SAFETY: the pointer is owned exactly once and native release is
        // ordered after all prior work on its admitted stream.
        let status = unsafe {
            neoethos_gpu_cuda_population_release_resident_scoring_run_v2(
                self.session.as_ptr(),
                native.as_ptr(),
            )
        };
        if status != STATUS_OK {
            self.state = ResidentScoringStateV2::Poisoned;
            return Err(native_error("enqueue_resident_scoring_release_v2", status));
        }
        self.native = None;
        self.state = ResidentScoringStateV2::Released;
        Ok(())
    }
}

impl Drop for ResidentScoringRunV2 {
    fn drop(&mut self) {
        if self.state != ResidentScoringStateV2::Poisoned
            && self.native.is_some()
            && self.release_v2().is_err()
        {
            self.state = ResidentScoringStateV2::Poisoned;
        }
    }
}

pub(crate) fn seal_resident_scoring_plan_v2(
    generation: &SealedResidentGenerationPlanV1,
    objective: ResidentScoringObjectiveV2,
    novelty_weight: f64,
    runtime: &RawResidentSearchRuntimeFactsV2,
) -> Result<SealedResidentScoringPlanV2, ResidentScoringV2Error> {
    if novelty_weight.to_bits() != 0_u64 {
        return Err(ResidentScoringV2Error::InvalidNoveltyWeight);
    }
    let scoring_semantics_sha256 = scoring_semantics_sha256_v2(objective);
    let novelty_semantics_sha256 = novelty_disabled_semantics_sha256_v2();
    let rank_semantics_sha256 = rank_semantics_sha256_v2();
    if generation.retained_evaluation_capacity_v1() != generation.logical_population_count_v1()
        || generation.scoring_semantics_sha256_v1() != scoring_semantics_sha256
        || generation.novelty_semantics_sha256_v1() != novelty_semantics_sha256
        || generation.rank_semantics_sha256_v1() != rank_semantics_sha256
    {
        return Err(ResidentScoringV2Error::InvalidPlan(
            "generation/scoring semantics or full-population capacity differ",
        ));
    }
    if runtime.abi_version != 2
        || runtime.run_admission_ordinal == 0
        || runtime.device_uuid == [0; 16]
        || runtime.primary_context_id == 0
        || runtime.run_stream_id == 0
        || runtime.admitted_primary_context.is_null()
        || runtime.admitted_run_stream.is_null()
        || runtime.admitted_memory_pool.is_null()
        || runtime.active_pool_is_default != 1
        || runtime.reserved != 0
        || runtime.pool_used_current_bytes > runtime.pool_reserved_current_bytes
        || runtime.allocator_context_reserve_bytes == 0
        || runtime.run_stream_process_token == [0; 32]
    {
        return Err(ResidentScoringV2Error::InvalidPlan(
            "CUDA runtime facts are incomplete or inconsistent",
        ));
    }
    let (cuda_device_identity_sha256, primary_context_identity_sha256, run_stream_identity_sha256) =
        runtime_identity_hashes_v2(runtime);
    let mut raw = RawResidentScoringPlanV2 {
        abi_version: SCORING_ABI_V1,
        scoring_objective: objective as u32,
        scoring_version: SCORING_VERSION_V1,
        reserved: 0,
        logical_population_count: generation.logical_population_count_v1(),
        feature_count: generation.feature_count_v1(),
        max_terms_per_gene: generation.max_terms_per_gene_v1(),
        reserved_extents: 0,
        novelty_weight_bits: novelty_weight.to_bits(),
        metric_semantics_sha256: generation.metric_semantics_sha256_v1(),
        scoring_semantics_sha256,
        novelty_semantics_sha256,
        scenario_order_semantics_sha256: generation.scenario_order_semantics_sha256_v1(),
        gene_schema_sha256: generation.strategy_gene_schema_sha256_v1(),
        rank_semantics_sha256,
        cuda_device_identity_sha256,
        primary_context_identity_sha256,
        run_stream_identity_sha256,
        cuda_build_manifest_sha256: generation.cuda_build_manifest_sha256_v1(),
        cuda_math_flags_sha256: cuda_math_flags_sha256_v2(),
        plan_identity_sha256: [0; 32],
    };
    raw.plan_identity_sha256 = hash_scoring_plan_v2(&raw);
    Ok(SealedResidentScoringPlanV2 { raw })
}

pub(crate) fn seal_combined_search_admission_v2(
    mut raw: RawResidentSearchCombinedAdmissionV2,
) -> Result<SealedResidentSearchAdmissionV2, ResidentScoringV2Error> {
    let generation = &raw.generation;
    let scoring = &raw.scoring;
    let total_device_bytes = generation
        .total_device_bytes
        .checked_add(scoring.total_device_bytes)
        .ok_or(ResidentScoringV2Error::ArithmeticOverflow)?;
    let full_discovery_reserve_bytes = raw.full_discovery_reserve_bytes;
    let available = generation
        .same_context_free_bytes
        .checked_sub(full_discovery_reserve_bytes)
        .ok_or(ResidentScoringV2Error::ArithmeticOverflow)?;
    if raw.abi_version != 2
        || raw.flags != 0
        || raw.free_memory_snapshot_count != 1
        || raw.generation_allocation_count != 1
        || raw.scoring_allocation_count != 1
        || raw.terminal_host_allocation_count != 1
        || raw.terminal_host_receipt_bytes == 0
        || raw.generation_device_bytes != generation.total_device_bytes
        || raw.scoring_device_bytes != scoring.total_device_bytes
        || raw.total_device_bytes != total_device_bytes
        || generation.same_context_free_bytes != raw.same_context_free_bytes
        || scoring.same_context_free_bytes != raw.same_context_free_bytes
        || generation.full_discovery_reserve_bytes != full_discovery_reserve_bytes
        || scoring.full_discovery_reserve_bytes != full_discovery_reserve_bytes
        || raw.pool_used_current_bytes > raw.pool_reserved_current_bytes
        || total_device_bytes > available
    {
        return Err(ResidentScoringV2Error::InvalidPlan(
            "combined generation/scoring allocation exceeds admitted reserve",
        ));
    }
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.resident-search.combined-admission.v2");
    hasher.update(generation.total_device_bytes.to_le_bytes());
    hasher.update(scoring.total_device_bytes.to_le_bytes());
    hasher.update(total_device_bytes.to_le_bytes());
    hasher.update(generation.same_context_free_bytes.to_le_bytes());
    hasher.update(full_discovery_reserve_bytes.to_le_bytes());
    hasher.update(generation.allocation_plan_sha256);
    hasher.update(scoring.allocation_plan_sha256);
    hasher.update(raw.runtime.device_uuid);
    hasher.update(raw.runtime.run_admission_ordinal.to_le_bytes());
    hasher.update(raw.runtime.primary_context_id.to_le_bytes());
    hasher.update(raw.runtime.run_stream_id.to_le_bytes());
    hasher.update(raw.runtime.run_stream_process_token);
    hasher.update(raw.pool_reserved_current_bytes.to_le_bytes());
    hasher.update(raw.pool_used_current_bytes.to_le_bytes());
    hasher.update(raw.terminal_host_receipt_bytes.to_le_bytes());
    raw.receipt_identity_sha256 = hasher.finalize().into();
    Ok(SealedResidentSearchAdmissionV2 {
        generation_device_bytes: generation.total_device_bytes,
        scoring_device_bytes: scoring.total_device_bytes,
        total_device_bytes,
        same_context_free_bytes: generation.same_context_free_bytes,
        full_discovery_reserve_bytes,
        generation_allocation_plan_sha256: generation.allocation_plan_sha256,
        scoring_allocation_plan_sha256: scoring.allocation_plan_sha256,
        receipt_identity_sha256: raw.receipt_identity_sha256,
        raw,
    })
}

fn runtime_identity_hashes_v2(
    runtime: &RawResidentSearchRuntimeFactsV2,
) -> ([u8; 32], [u8; 32], [u8; 32]) {
    let mut device = Sha256::new();
    device.update(b"neoethos.cuda-device-runtime.v2");
    device.update(runtime.selected_cuda_ordinal.to_le_bytes());
    device.update(runtime.device_uuid);
    device.update(runtime.compute_capability_major.to_le_bytes());
    device.update(runtime.compute_capability_minor.to_le_bytes());
    let mut context = Sha256::new();
    context.update(b"neoethos.cuda-primary-context-runtime.v2");
    context.update(runtime.device_uuid);
    context.update(runtime.run_admission_ordinal.to_le_bytes());
    context.update(runtime.primary_context_id.to_le_bytes());
    let mut stream = Sha256::new();
    stream.update(b"neoethos.cuda-stream-pool-runtime.v2");
    stream.update(runtime.device_uuid);
    stream.update(runtime.run_admission_ordinal.to_le_bytes());
    stream.update(runtime.primary_context_id.to_le_bytes());
    stream.update(runtime.run_stream_id.to_le_bytes());
    stream.update(runtime.pool_location_type.to_le_bytes());
    stream.update(runtime.pool_location_id.to_le_bytes());
    stream.update(runtime.pool_allocation_type.to_le_bytes());
    stream.update(runtime.pool_handle_types.to_le_bytes());
    stream.update(runtime.run_stream_process_token);
    (
        device.finalize().into(),
        context.finalize().into(),
        stream.finalize().into(),
    )
}

pub(crate) fn scoring_semantics_sha256_v2(objective: ResidentScoringObjectiveV2) -> [u8; 32] {
    sha256_v2(match objective {
        ResidentScoringObjectiveV2::PropFirmV4 => RESIDENT_SCORING_SEMANTICS_PROPFIRM_V2,
        ResidentScoringObjectiveV2::RiskyGrowthV5 => RESIDENT_SCORING_SEMANTICS_RISKY_V2,
    })
}

pub(crate) fn novelty_disabled_semantics_sha256_v2() -> [u8; 32] {
    sha256_v2(RESIDENT_NOVELTY_DISABLED_SEMANTICS_V2)
}

pub(crate) fn rank_semantics_sha256_v2() -> [u8; 32] {
    sha256_v2(RESIDENT_RANK_SEMANTICS_V2)
}

fn cuda_math_flags_sha256_v2() -> [u8; 32] {
    sha256_v2(RESIDENT_CUDA_MATH_SEMANTICS_V2)
}

fn sha256_v2(value: &str) -> [u8; 32] {
    Sha256::digest(value.as_bytes()).into()
}

fn hash_scoring_plan_v2(raw: &RawResidentScoringPlanV2) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.resident-scoring-plan.v2");
    hasher.update(raw.abi_version.to_le_bytes());
    hasher.update(raw.scoring_objective.to_le_bytes());
    hasher.update(raw.scoring_version.to_le_bytes());
    hasher.update(raw.logical_population_count.to_le_bytes());
    hasher.update(raw.feature_count.to_le_bytes());
    hasher.update(raw.max_terms_per_gene.to_le_bytes());
    hasher.update(raw.novelty_weight_bits.to_le_bytes());
    hasher.update(raw.metric_semantics_sha256);
    hasher.update(raw.scoring_semantics_sha256);
    hasher.update(raw.novelty_semantics_sha256);
    hasher.update(raw.scenario_order_semantics_sha256);
    hasher.update(raw.gene_schema_sha256);
    hasher.update(raw.rank_semantics_sha256);
    hasher.update(raw.cuda_device_identity_sha256);
    hasher.update(raw.primary_context_identity_sha256);
    hasher.update(raw.run_stream_identity_sha256);
    hasher.update(raw.cuda_build_manifest_sha256);
    hasher.update(raw.cuda_math_flags_sha256);
    hasher.finalize().into()
}

fn native_error(operation: &'static str, status: i32) -> ResidentScoringV2Error {
    match status {
        STATUS_ASYNC_FREE_OUTCOME_UNKNOWN => {
            ResidentScoringV2Error::AsyncFreeOutcomeUnknownDeliberateLeak { operation }
        }
        STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN => {
            ResidentScoringV2Error::AsyncAllocationOutcomeUnknownDeliberateLeak { operation }
        }
        _ => ResidentScoringV2Error::Native { operation, status },
    }
}
