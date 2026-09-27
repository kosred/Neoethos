//! Move-only native scoring admission with distinct CUDA and HIP runtime identities.

use crate::resident_generation_v1::{RawAllocationReceiptV1, SealedResidentGenerationPlanV1};
use neoethos_gpu_contracts::resident_search_scoring_v2::RiskyGrowthGoal;
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::ptr::NonNull;
use thiserror::Error;

pub(crate) const RESIDENT_SCORING_SEMANTICS_PROPFIRM_V2: &str = concat!(
    "neoethos.resident-scoring.v2;objective=prop-firm-v4;",
    "checked-eleven-metrics;finite-or-authenticated-economic-objective;raw-objective;",
    "positive-zero-canonical;cuda-build-and-math-bound;no-host-decision"
);
pub(crate) const RESIDENT_SCORING_SEMANTICS_RISKY_V2: &str = concat!(
    "neoethos.resident-scoring.v2;objective=risky-growth-v5;",
    "checked-eleven-metrics;finite-or-authenticated-economic-objective;raw-objective;",
    "positive-zero-canonical;cuda-build-and-math-bound;no-host-decision"
);
pub(crate) const RESIDENT_SCORING_SEMANTICS_GOAL_V6: &str = concat!(
    "neoethos.resident-scoring.v2;objective=risky-growth-goal-v6;",
    "ln1p-realized-net/actual-equity;observed-calendar-days;goal-and-context-bits-bound;",
    "one-minus-squared-relative-log-shortfall;pace-proxy-not-goal-proof;",
    "checked-eleven-metrics;finite-or-authenticated-economic-objective;raw-objective;",
    "positive-zero-canonical;cuda-build-and-math-bound;no-host-decision"
);
pub(crate) const RESIDENT_NOVELTY_DISABLED_SEMANTICS_V2: &str = concat!(
    "neoethos.resident-novelty-disabled.v2;novelty-weight-bits=positive-zero;",
    "no-current-only-mean-jaccard;no-knn-without-explicit-k;no-archive"
);
pub(crate) const RESIDENT_RANK_SEMANTICS_V2: &str = concat!(
    "neoethos.resident-rank.v2;stable-lsd;score-desc;gene-identity-asc;",
    "population-ordinal-asc;ordered-f64;positive-zero-canonical;economic-reject-key=1;",
    "defined-sentinel-cub-inputs;fault-gated-semantic-commit"
);
#[cfg(not(feature = "hip-native-kernels"))]
pub(crate) const RESIDENT_CUDA_MATH_SEMANTICS_V2: &str =
    "neoethos.cuda-math.v2;fmad=false;ftz=false;prec-div=true;prec-sqrt=true";

#[cfg(not(feature = "hip-native-kernels"))]
const SCORING_PLAN_ABI_V2: u32 = 2;
#[cfg(feature = "hip-native-kernels")]
const SCORING_PLAN_ABI_V2: u32 = 0x0001_0002;
#[cfg(feature = "hip-native-kernels")]
const RESIDENT_HIP_MATH_SEMANTICS_V1: &str = "neoethos.hip-math.v1;fast-math=false;fp-contract=off;denormal-fp-math=ieee;denormal-fp-math-f32=ieee;gpu-flush-denormals-to-zero=false;fp32-correctly-rounded-divide-sqrt=true;unsafe-fp-atomics=false;explicit-fma=preserved";
const SCORING_VERSION_V1: u32 = 5;
const STATUS_OK: i32 = 0;
const STATUS_ASYNC_FREE_OUTCOME_UNKNOWN: i32 = -48;
const STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN: i32 = -49;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResidentScoringObjectiveV2 {
    PropFirmV4 = 1,
    #[allow(dead_code)] // Constructed by the real-device dual-objective oracle.
    RiskyGrowthV5 = 2,
    RiskyGrowthGoalV6 = 3,
}

/// Exact run-owned scoring inputs, not a replacement for the simulation's
/// capital or a claim that the goal has been reached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResidentScoringGoalContextV2 {
    pub(crate) initial_equity: f64,
    pub(crate) span_days: f64,
    pub(crate) goal: RiskyGrowthGoal,
}

fn scoring_goal_bits_v2(
    objective: ResidentScoringObjectiveV2,
    context: Option<ResidentScoringGoalContextV2>,
) -> Result<[u64; 5], ResidentScoringV2Error> {
    match (objective, context) {
        (ResidentScoringObjectiveV2::RiskyGrowthGoalV6, Some(context)) => {
            context
                .goal
                .validate()
                .map_err(ResidentScoringV2Error::InvalidPlan)?;
            if !context.initial_equity.is_finite()
                || context.initial_equity <= 0.0
                || !context.span_days.is_finite()
                || context.span_days <= 0.0
            {
                return Err(ResidentScoringV2Error::InvalidPlan(
                    "goal scoring requires positive finite actual equity and observed calendar span",
                ));
            }
            Ok([
                context.initial_equity.to_bits(),
                context.span_days.to_bits(),
                context.goal.start_balance.to_bits(),
                context.goal.target_balance.to_bits(),
                context.goal.horizon_days.to_bits(),
            ])
        }
        (
            ResidentScoringObjectiveV2::PropFirmV4 | ResidentScoringObjectiveV2::RiskyGrowthV5,
            None,
        ) => Ok([0; 5]),
        _ => Err(ResidentScoringV2Error::InvalidPlan(
            "goal context must be present exactly for the V6 goal objective",
        )),
    }
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
    initial_equity_bits: u64,
    span_days_bits: u64,
    goal_start_balance_bits: u64,
    goal_target_balance_bits: u64,
    goal_horizon_days_bits: u64,
    metric_semantics_sha256: [u8; 32],
    scoring_semantics_sha256: [u8; 32],
    novelty_semantics_sha256: [u8; 32],
    scenario_order_semantics_sha256: [u8; 32],
    gene_schema_sha256: [u8; 32],
    rank_semantics_sha256: [u8; 32],
    #[cfg(not(feature = "hip-native-kernels"))]
    cuda_device_identity_sha256: [u8; 32],
    #[cfg(feature = "hip-native-kernels")]
    hip_device_identity_sha256: [u8; 32],
    #[cfg(not(feature = "hip-native-kernels"))]
    primary_context_identity_sha256: [u8; 32],
    #[cfg(feature = "hip-native-kernels")]
    hip_lease_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    #[cfg(not(feature = "hip-native-kernels"))]
    cuda_build_manifest_sha256: [u8; 32],
    #[cfg(feature = "hip-native-kernels")]
    hip_build_manifest_sha256: [u8; 32],
    #[cfg(not(feature = "hip-native-kernels"))]
    cuda_math_flags_sha256: [u8; 32],
    #[cfg(feature = "hip-native-kernels")]
    hip_math_flags_sha256: [u8; 32],
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

impl RawResidentScoringAllocationReceiptV2 {
    pub(crate) const fn cub_scratch_bytes_v2(&self) -> u64 {
        self.cub_scratch_bytes
    }
}

#[cfg(not(feature = "hip-native-kernels"))]
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

#[cfg(not(feature = "hip-native-kernels"))]
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

#[cfg(not(feature = "hip-native-kernels"))]
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

#[cfg(feature = "hip-native-kernels")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawResidentSearchRuntimeFactsHipV1 {
    pub(crate) abi_version: u32,
    pub(crate) backend_kind: u32,
    pub(crate) run_admission_ordinal: u64,
    pub(crate) owner: crate::hip_runtime_v1::RawHipRuntimeFactsV1,
    pub(crate) allocator_context_reserve_bytes: u64,
    pub(crate) run_stream_process_token: [u8; 32],
}
#[cfg(feature = "hip-native-kernels")]
impl Default for RawResidentSearchRuntimeFactsHipV1 {
    fn default() -> Self {
        Self {
            abi_version: 0,
            backend_kind: 0,
            run_admission_ordinal: 0,
            owner: crate::hip_runtime_v1::RawHipRuntimeFactsV1::empty(),
            allocator_context_reserve_bytes: 0,
            run_stream_process_token: [0; 32],
        }
    }
}
#[cfg(feature = "hip-native-kernels")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RawResidentSearchCombinedAdmissionHipV1 {
    pub(crate) abi_version: u32,
    pub(crate) flags: u32,
    pub(crate) free_memory_snapshot_count: u32,
    pub(crate) generation_allocation_count: u32,
    pub(crate) scoring_allocation_count: u32,
    pub(crate) terminal_host_allocation_count: u32,
    pub(crate) terminal_host_receipt_bytes: u64,
    pub(crate) same_lease_free_bytes: u64,
    pub(crate) same_lease_total_bytes: u64,
    pub(crate) full_discovery_reserve_bytes: u64,
    pub(crate) generation_device_bytes: u64,
    pub(crate) scoring_device_bytes: u64,
    pub(crate) total_device_bytes: u64,
    pub(crate) pool_reserved_current_bytes: u64,
    pub(crate) pool_used_current_bytes: u64,
    pub(crate) runtime: RawResidentSearchRuntimeFactsHipV1,
    pub(crate) generation: RawAllocationReceiptV1,
    pub(crate) scoring: RawResidentScoringAllocationReceiptV2,
    pub(crate) receipt_identity_sha256: [u8; 32],
}
#[cfg(not(feature = "hip-native-kernels"))]
pub(crate) type SelectedResidentSearchRuntimeFactsV3 = RawResidentSearchRuntimeFactsV2;
#[cfg(feature = "hip-native-kernels")]
pub(crate) type SelectedResidentSearchRuntimeFactsV3 = RawResidentSearchRuntimeFactsHipV1;
#[cfg(not(feature = "hip-native-kernels"))]
pub(crate) type SelectedResidentSearchCombinedAdmissionV3 = RawResidentSearchCombinedAdmissionV2;
#[cfg(feature = "hip-native-kernels")]
pub(crate) type SelectedResidentSearchCombinedAdmissionV3 = RawResidentSearchCombinedAdmissionHipV1;

/// Checked interpretation of the selected native DTO, not caller admission.
pub(crate) struct ResidentSearchRuntimeIdentityV3 {
    pub(crate) ordinal: u32,
    pub(crate) device_uuid: [u8; 16],
    pub(crate) owner_identity: u64,
    pub(crate) stream_identity: u64,
    pub(crate) pool_identity: u64,
}
impl SelectedResidentSearchRuntimeFactsV3 {
    pub(crate) fn identity_v3(
        &self,
    ) -> Result<ResidentSearchRuntimeIdentityV3, ResidentScoringV2Error> {
        if self.run_admission_ordinal == 0
            || self.allocator_context_reserve_bytes == 0
            || self.run_stream_process_token == [0; 32]
        {
            return Err(ResidentScoringV2Error::InvalidPlan(
                "native runtime run authority is incomplete",
            ));
        }
        #[cfg(not(feature = "hip-native-kernels"))]
        {
            if self.abi_version != 2
                || self.device_uuid == [0; 16]
                || self.primary_context_id == 0
                || self.run_stream_id == 0
                || self.admitted_primary_context.is_null()
                || self.admitted_run_stream.is_null()
                || self.admitted_memory_pool.is_null()
                || self.active_pool_is_default != 1
                || self.reserved != 0
            {
                return Err(ResidentScoringV2Error::InvalidPlan(
                    "CUDA runtime facts are incomplete or inconsistent",
                ));
            }
            Ok(ResidentSearchRuntimeIdentityV3 {
                ordinal: self.selected_cuda_ordinal,
                device_uuid: self.device_uuid,
                owner_identity: self.primary_context_id,
                stream_identity: self.run_stream_id,
                pool_identity: self.admitted_memory_pool as usize as u64,
            })
        }
        #[cfg(feature = "hip-native-kernels")]
        {
            if self.abi_version != 1 || self.backend_kind != 2 {
                return Err(ResidentScoringV2Error::InvalidPlan(
                    "HIP Search runtime ABI differs",
                ));
            }
            let lease = std::num::NonZeroU64::new(self.owner.lease_id)
                .ok_or(ResidentScoringV2Error::InvalidPlan("HIP lease is zero"))?;
            let ordinal = u32::try_from(self.owner.device_ordinal)
                .map_err(|_| ResidentScoringV2Error::InvalidPlan("HIP ordinal is negative"))?;
            crate::hip_runtime_v1::validate_facts_v1(&self.owner, lease, ordinal).map_err(
                |_| {
                    ResidentScoringV2Error::InvalidPlan(
                        "HIP runtime facts are incomplete or inconsistent",
                    )
                },
            )?;
            Ok(ResidentSearchRuntimeIdentityV3 {
                ordinal,
                device_uuid: self.owner.uuid,
                owner_identity: self.owner.lease_id,
                stream_identity: self.owner.stream_id,
                pool_identity: self.owner.current_pool_handle,
            })
        }
    }
    pub(crate) fn pool_reserved_bytes_v3(&self) -> u64 {
        #[cfg(not(feature = "hip-native-kernels"))]
        {
            self.pool_reserved_current_bytes
        }
        #[cfg(feature = "hip-native-kernels")]
        {
            self.owner.pool_reserved_bytes
        }
    }
    pub(crate) fn pool_used_bytes_v3(&self) -> u64 {
        #[cfg(not(feature = "hip-native-kernels"))]
        {
            self.pool_used_current_bytes
        }
        #[cfg(feature = "hip-native-kernels")]
        {
            self.owner.pool_used_bytes
        }
    }
}
impl SelectedResidentSearchCombinedAdmissionV3 {
    pub(crate) fn free_bytes_v3(&self) -> u64 {
        #[cfg(not(feature = "hip-native-kernels"))]
        {
            self.same_context_free_bytes
        }
        #[cfg(feature = "hip-native-kernels")]
        {
            self.same_lease_free_bytes
        }
    }
    pub(crate) fn total_bytes_v3(&self) -> u64 {
        #[cfg(not(feature = "hip-native-kernels"))]
        {
            self.same_context_total_bytes
        }
        #[cfg(feature = "hip-native-kernels")]
        {
            self.same_lease_total_bytes
        }
    }
}
pub(crate) const fn selected_combined_abi_v3() -> u32 {
    #[cfg(not(feature = "hip-native-kernels"))]
    {
        2
    }
    #[cfg(feature = "hip-native-kernels")]
    {
        1
    }
}

pub(crate) enum NativeResidentScoringRunV2 {}

const _: [(); 472] = [(); std::mem::size_of::<RawResidentScoringPlanV2>()];
const _: [(); 128] = [(); std::mem::size_of::<RawResidentScoringAllocationReceiptV2>()];
#[cfg(not(feature = "hip-native-kernels"))]
const _: [(); 160] = [(); std::mem::size_of::<RawResidentSearchRuntimeFactsV2>()];
#[cfg(not(feature = "hip-native-kernels"))]
const _: [(); 592] = [(); std::mem::size_of::<RawResidentSearchCombinedAdmissionV2>()];
#[cfg(feature = "hip-native-kernels")]
const _: [(); 424] = [(); std::mem::size_of::<RawResidentSearchRuntimeFactsHipV1>()];
#[cfg(feature = "hip-native-kernels")]
const _: [(); 856] = [(); std::mem::size_of::<RawResidentSearchCombinedAdmissionHipV1>()];

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
    pub(crate) raw: SelectedResidentSearchCombinedAdmissionV3,
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

fn valid_retained_capacity_v3(population: u64, capacity: u64) -> bool {
    population != 0 && capacity != 0 && capacity <= population
}

pub(crate) fn seal_resident_scoring_plan_v2(
    generation: &SealedResidentGenerationPlanV1,
    objective: ResidentScoringObjectiveV2,
    novelty_weight: f64,
    goal_context: Option<ResidentScoringGoalContextV2>,
    runtime: &SelectedResidentSearchRuntimeFactsV3,
) -> Result<SealedResidentScoringPlanV2, ResidentScoringV2Error> {
    if novelty_weight.to_bits() != 0_u64 {
        return Err(ResidentScoringV2Error::InvalidNoveltyWeight);
    }
    let goal_bits = scoring_goal_bits_v2(objective, goal_context)?;
    let scoring_semantics_sha256 = scoring_semantics_sha256_v2(objective);
    let novelty_semantics_sha256 = novelty_disabled_semantics_sha256_v2();
    let rank_semantics_sha256 = rank_semantics_sha256_v2();
    if !valid_retained_capacity_v3(
        generation.logical_population_count_v1(),
        generation.retained_evaluation_capacity_v1(),
    ) || generation.scoring_semantics_sha256_v1() != scoring_semantics_sha256
        || generation.novelty_semantics_sha256_v1() != novelty_semantics_sha256
        || generation.rank_semantics_sha256_v1() != rank_semantics_sha256
    {
        return Err(ResidentScoringV2Error::InvalidPlan(
            "generation/scoring semantics or bounded evaluation capacity differ",
        ));
    }
    runtime.identity_v3()?;
    let (cuda_device_identity_sha256, primary_context_identity_sha256, run_stream_identity_sha256) =
        runtime_identity_hashes_v2(runtime);
    let mut raw = RawResidentScoringPlanV2 {
        abi_version: SCORING_PLAN_ABI_V2,
        scoring_objective: objective as u32,
        scoring_version: if objective == ResidentScoringObjectiveV2::RiskyGrowthGoalV6 {
            6
        } else {
            SCORING_VERSION_V1
        },
        reserved: 0,
        logical_population_count: generation.logical_population_count_v1(),
        feature_count: generation.feature_count_v1(),
        max_terms_per_gene: generation.max_terms_per_gene_v1(),
        reserved_extents: 0,
        novelty_weight_bits: novelty_weight.to_bits(),
        initial_equity_bits: goal_bits[0],
        span_days_bits: goal_bits[1],
        goal_start_balance_bits: goal_bits[2],
        goal_target_balance_bits: goal_bits[3],
        goal_horizon_days_bits: goal_bits[4],
        metric_semantics_sha256: generation.metric_semantics_sha256_v1(),
        scoring_semantics_sha256,
        novelty_semantics_sha256,
        scenario_order_semantics_sha256: generation.scenario_order_semantics_sha256_v1(),
        gene_schema_sha256: generation.strategy_gene_schema_sha256_v1(),
        rank_semantics_sha256,
        #[cfg(not(feature = "hip-native-kernels"))]
        cuda_device_identity_sha256,
        #[cfg(feature = "hip-native-kernels")]
        hip_device_identity_sha256: cuda_device_identity_sha256,
        #[cfg(not(feature = "hip-native-kernels"))]
        primary_context_identity_sha256,
        #[cfg(feature = "hip-native-kernels")]
        hip_lease_identity_sha256: primary_context_identity_sha256,
        run_stream_identity_sha256,
        #[cfg(not(feature = "hip-native-kernels"))]
        cuda_build_manifest_sha256: generation.native_build_manifest_sha256_v1(),
        #[cfg(feature = "hip-native-kernels")]
        hip_build_manifest_sha256: generation.native_build_manifest_sha256_v1(),
        #[cfg(not(feature = "hip-native-kernels"))]
        cuda_math_flags_sha256: native_math_flags_sha256_v3(),
        #[cfg(feature = "hip-native-kernels")]
        hip_math_flags_sha256: native_math_flags_sha256_v3(),
        plan_identity_sha256: [0; 32],
    };
    raw.plan_identity_sha256 = hash_scoring_plan_v2(&raw);
    Ok(SealedResidentScoringPlanV2 { raw })
}

pub(crate) fn seal_combined_search_admission_v2(
    mut raw: SelectedResidentSearchCombinedAdmissionV3,
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
    if raw.abi_version != selected_combined_abi_v3()
        || raw.flags != 0
        || raw.free_memory_snapshot_count != 1
        || raw.generation_allocation_count != 1
        || raw.scoring_allocation_count != 1
        || raw.terminal_host_allocation_count != 1
        || raw.terminal_host_receipt_bytes == 0
        || raw.generation_device_bytes != generation.total_device_bytes
        || raw.scoring_device_bytes != scoring.total_device_bytes
        || raw.total_device_bytes != total_device_bytes
        || generation.same_context_free_bytes != raw.free_bytes_v3()
        || scoring.same_context_free_bytes != raw.free_bytes_v3()
        || generation.full_discovery_reserve_bytes != full_discovery_reserve_bytes
        || scoring.full_discovery_reserve_bytes != full_discovery_reserve_bytes
        || total_device_bytes > available
    {
        return Err(ResidentScoringV2Error::InvalidPlan(
            "combined generation/scoring allocation exceeds admitted reserve",
        ));
    }
    let identity = raw.runtime.identity_v3()?;
    #[cfg(feature = "hip-native-kernels")]
    if raw.runtime.owner.free_memory_bytes != raw.free_bytes_v3()
        || raw.runtime.owner.total_memory_bytes != raw.total_bytes_v3()
        || raw.generation.abi_version != crate::resident_generation_v1::selected_generation_abi_v1()
        || raw.scoring.abi_version != crate::resident_generation_v1::selected_generation_abi_v1()
        || raw.runtime.pool_reserved_bytes_v3() != raw.pool_reserved_current_bytes
        || raw.runtime.pool_used_bytes_v3() != raw.pool_used_current_bytes
        || raw.runtime.allocator_context_reserve_bytes != full_discovery_reserve_bytes
    {
        return Err(ResidentScoringV2Error::InvalidPlan(
            "HIP combined receipt lost its single captured runtime snapshot",
        ));
    }
    let mut hasher = Sha256::new();
    #[cfg(not(feature = "hip-native-kernels"))]
    hasher.update(b"neoethos.resident-search.combined-admission.v2");
    #[cfg(feature = "hip-native-kernels")]
    hasher.update(b"neoethos.hip-resident-search.combined-admission.v1");
    hasher.update(generation.total_device_bytes.to_le_bytes());
    hasher.update(scoring.total_device_bytes.to_le_bytes());
    hasher.update(total_device_bytes.to_le_bytes());
    hasher.update(generation.same_context_free_bytes.to_le_bytes());
    hasher.update(full_discovery_reserve_bytes.to_le_bytes());
    hasher.update(generation.allocation_plan_sha256);
    hasher.update(scoring.allocation_plan_sha256);
    hasher.update(identity.device_uuid);
    hasher.update(raw.runtime.run_admission_ordinal.to_le_bytes());
    hasher.update(identity.owner_identity.to_le_bytes());
    hasher.update(identity.stream_identity.to_le_bytes());
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

#[cfg(not(feature = "hip-native-kernels"))]
fn runtime_identity_hashes_v2(
    runtime: &SelectedResidentSearchRuntimeFactsV3,
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

#[cfg(feature = "hip-native-kernels")]
fn runtime_identity_hashes_v2(
    runtime: &SelectedResidentSearchRuntimeFactsV3,
) -> ([u8; 32], [u8; 32], [u8; 32]) {
    let owner = &runtime.owner;
    let mut device = Sha256::new();
    device.update(b"neoethos.hip-device-runtime.v1");
    device.update(owner.backend_kind.to_le_bytes());
    device.update(owner.device_ordinal.to_le_bytes());
    device.update(owner.uuid);
    device.update(owner.architecture);
    device.update(owner.runtime_version.to_le_bytes());
    device.update(owner.driver_version.to_le_bytes());
    device.update(owner.warp_size.to_le_bytes());
    device.update(owner.total_memory_bytes.to_le_bytes());
    let mut lease = Sha256::new();
    lease.update(b"neoethos.hip-owned-lease-runtime.v1");
    lease.update(owner.uuid);
    lease.update(owner.lease_id.to_le_bytes());
    lease.update(runtime.run_admission_ordinal.to_le_bytes());
    let mut stream = Sha256::new();
    stream.update(b"neoethos.hip-owned-stream-pool-runtime.v1");
    stream.update(owner.uuid);
    stream.update(owner.lease_id.to_le_bytes());
    stream.update(owner.stream_id.to_le_bytes());
    stream.update(owner.stream_handle.to_le_bytes());
    stream.update(owner.current_pool_handle.to_le_bytes());
    stream.update(owner.default_pool_handle.to_le_bytes());
    stream.update(runtime.run_admission_ordinal.to_le_bytes());
    stream.update(runtime.run_stream_process_token);
    (
        device.finalize().into(),
        lease.finalize().into(),
        stream.finalize().into(),
    )
}

pub fn scoring_semantics_sha256_v2(objective: ResidentScoringObjectiveV2) -> [u8; 32] {
    let objective_semantics = match objective {
        ResidentScoringObjectiveV2::PropFirmV4 => RESIDENT_SCORING_SEMANTICS_PROPFIRM_V2,
        ResidentScoringObjectiveV2::RiskyGrowthV5 => RESIDENT_SCORING_SEMANTICS_RISKY_V2,
        ResidentScoringObjectiveV2::RiskyGrowthGoalV6 => RESIDENT_SCORING_SEMANTICS_GOAL_V6,
    };
    let mut hash = Sha256::new();
    #[cfg(not(feature = "hip-native-kernels"))]
    hash.update(objective_semantics.as_bytes());
    #[cfg(feature = "hip-native-kernels")]
    hash.update(
        objective_semantics
            .replace("cuda-build-and-math-bound", "hip-build-and-math-bound")
            .as_bytes(),
    );
    hash.update(
        neoethos_gpu_contracts::resident_search_scoring_v2::RESIDENT_ECONOMIC_REJECTION_V2_SEMANTICS
            .as_bytes(),
    );
    hash.finalize().into()
}

pub fn novelty_disabled_semantics_sha256_v2() -> [u8; 32] {
    sha256_v2(RESIDENT_NOVELTY_DISABLED_SEMANTICS_V2)
}

pub fn rank_semantics_sha256_v2() -> [u8; 32] {
    #[cfg(not(feature = "hip-native-kernels"))]
    {
        sha256_v2(RESIDENT_RANK_SEMANTICS_V2)
    }
    #[cfg(feature = "hip-native-kernels")]
    {
        sha256_v2(&RESIDENT_RANK_SEMANTICS_V2.replace(
            "defined-sentinel-cub-inputs",
            "defined-sentinel-hipcub-rocprim-inputs",
        ))
    }
}

#[cfg(not(feature = "hip-native-kernels"))]
pub(crate) fn cuda_math_flags_sha256_v2() -> [u8; 32] {
    sha256_v2(RESIDENT_CUDA_MATH_SEMANTICS_V2)
}

pub(crate) fn native_math_flags_sha256_v3() -> [u8; 32] {
    #[cfg(not(feature = "hip-native-kernels"))]
    {
        cuda_math_flags_sha256_v2()
    }
    #[cfg(feature = "hip-native-kernels")]
    {
        sha256_v2(RESIDENT_HIP_MATH_SEMANTICS_V1)
    }
}

fn sha256_v2(value: &str) -> [u8; 32] {
    Sha256::digest(value.as_bytes()).into()
}

fn hash_scoring_plan_v2(raw: &RawResidentScoringPlanV2) -> [u8; 32] {
    let mut hasher = Sha256::new();
    #[cfg(not(feature = "hip-native-kernels"))]
    hasher.update(b"neoethos.resident-scoring-plan.v2");
    #[cfg(feature = "hip-native-kernels")]
    hasher.update(b"neoethos.hip-resident-scoring-plan.v1");
    hasher.update(raw.abi_version.to_le_bytes());
    hasher.update(raw.scoring_objective.to_le_bytes());
    hasher.update(raw.scoring_version.to_le_bytes());
    hasher.update(raw.logical_population_count.to_le_bytes());
    hasher.update(raw.feature_count.to_le_bytes());
    hasher.update(raw.max_terms_per_gene.to_le_bytes());
    hasher.update(raw.novelty_weight_bits.to_le_bytes());
    hasher.update(raw.initial_equity_bits.to_le_bytes());
    hasher.update(raw.span_days_bits.to_le_bytes());
    hasher.update(raw.goal_start_balance_bits.to_le_bytes());
    hasher.update(raw.goal_target_balance_bits.to_le_bytes());
    hasher.update(raw.goal_horizon_days_bits.to_le_bytes());
    hasher.update(raw.metric_semantics_sha256);
    hasher.update(raw.scoring_semantics_sha256);
    hasher.update(raw.novelty_semantics_sha256);
    hasher.update(raw.scenario_order_semantics_sha256);
    hasher.update(raw.gene_schema_sha256);
    hasher.update(raw.rank_semantics_sha256);
    #[cfg(not(feature = "hip-native-kernels"))]
    hasher.update(raw.cuda_device_identity_sha256);
    #[cfg(feature = "hip-native-kernels")]
    hasher.update(raw.hip_device_identity_sha256);
    #[cfg(not(feature = "hip-native-kernels"))]
    hasher.update(raw.primary_context_identity_sha256);
    #[cfg(feature = "hip-native-kernels")]
    hasher.update(raw.hip_lease_identity_sha256);
    hasher.update(raw.run_stream_identity_sha256);
    #[cfg(not(feature = "hip-native-kernels"))]
    hasher.update(raw.cuda_build_manifest_sha256);
    #[cfg(feature = "hip-native-kernels")]
    hasher.update(raw.hip_build_manifest_sha256);
    #[cfg(not(feature = "hip-native-kernels"))]
    hasher.update(raw.cuda_math_flags_sha256);
    #[cfg(feature = "hip-native-kernels")]
    hasher.update(raw.hip_math_flags_sha256);
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

#[cfg(test)]
mod goal_scoring_tests {
    use super::*;

    #[test]
    fn retained_capacity_covers_every_candidate_without_requiring_all_live_slots() {
        for (population, capacity) in [(1, 1), (12, 5), (12, 12), (u64::MAX, 1)] {
            assert!(valid_retained_capacity_v3(population, capacity));
        }
        for (population, capacity) in [(0, 0), (0, 1), (12, 0), (12, 13)] {
            assert!(!valid_retained_capacity_v3(population, capacity));
        }
    }

    #[cfg(feature = "hip-native-kernels")]
    fn hip_runtime() -> RawResidentSearchRuntimeFactsHipV1 {
        let mut runtime = RawResidentSearchRuntimeFactsHipV1::default();
        runtime.abi_version = 1;
        runtime.backend_kind = 2;
        runtime.run_admission_ordinal = 19;
        runtime.allocator_context_reserve_bytes = 128;
        runtime.run_stream_process_token = [13; 32];
        let owner = &mut runtime.owner;
        owner.abi_version = 1;
        owner.backend_kind = 2;
        owner.lease_id = 7;
        owner.device_ordinal = 1;
        owner.runtime_version = 70_205_323;
        owner.driver_version = 70_205_323;
        owner.warp_size = 64;
        owner.uuid = [11; 16];
        owner.stream_handle = 0x1000;
        owner.stream_id = 33;
        owner.free_memory_bytes = 1024;
        owner.total_memory_bytes = 2048;
        owner.current_pool_handle = 0x2000;
        owner.default_pool_handle = 0x2000;
        // Separately queried counters may cross; they are not an atomic pair.
        owner.pool_reserved_bytes = 30;
        owner.pool_used_bytes = 31;
        owner.architecture[..6].copy_from_slice(b"gfx942");
        runtime
    }

    #[cfg(feature = "hip-native-kernels")]
    #[test]
    fn hip_runtime_abi_and_identity_are_not_cuda_or_an_atomic_pool_snapshot() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<RawResidentSearchRuntimeFactsHipV1>(), 424);
        assert_eq!(size_of::<RawResidentSearchCombinedAdmissionHipV1>(), 856);
        assert_eq!(
            offset_of!(RawResidentSearchCombinedAdmissionHipV1, runtime),
            96
        );
        assert_eq!(
            offset_of!(RawResidentSearchCombinedAdmissionHipV1, generation),
            520
        );
        assert_eq!(
            offset_of!(
                RawResidentSearchCombinedAdmissionHipV1,
                receipt_identity_sha256
            ),
            824
        );
        let baseline = hip_runtime();
        assert!(baseline.identity_v3().is_ok());
        let hash = runtime_identity_hashes_v2(&baseline);
        let mut changed = baseline;
        changed.owner.lease_id += 1;
        assert_ne!(runtime_identity_hashes_v2(&changed), hash);
        changed = baseline;
        changed.owner.stream_id += 1;
        assert_ne!(runtime_identity_hashes_v2(&changed), hash);
        changed = baseline;
        changed.owner.architecture[5] = b'0';
        assert_ne!(runtime_identity_hashes_v2(&changed), hash);
        for case in 0..5 {
            changed = baseline;
            match case {
                0 => changed.backend_kind = 0,
                1 => changed.owner.lease_id = 0,
                2 => changed.owner.stream_handle = 2,
                3 => changed.owner.current_pool_handle += 1,
                _ => changed.run_stream_process_token = [0; 32],
            }
            assert!(changed.identity_v3().is_err(), "case {case}");
        }
        assert_eq!(
            native_math_flags_sha256_v3(),
            [
                0x71, 0x04, 0xe9, 0x27, 0xbf, 0xc4, 0x17, 0x92, 0x71, 0x8e, 0x78, 0x3d, 0x19, 0x6e,
                0x02, 0x72, 0xe9, 0xee, 0xda, 0xe4, 0x67, 0xe6, 0x9d, 0xed, 0x85, 0x5d, 0x83, 0xae,
                0x2d, 0xf2, 0x57, 0x55
            ]
        );
    }

    fn context() -> ResidentScoringGoalContextV2 {
        ResidentScoringGoalContextV2 {
            initial_equity: 10_000.0,
            span_days: 396.0,
            goal: RiskyGrowthGoal {
                start_balance: 100.0,
                target_balance: 50_000.0,
                horizon_days: 180.0,
            },
        }
    }

    #[test]
    fn goal_context_requires_exact_objective_and_positive_measured_inputs() {
        use ResidentScoringObjectiveV2::*;
        assert!(scoring_goal_bits_v2(RiskyGrowthGoalV6, Some(context())).is_ok());
        assert!(scoring_goal_bits_v2(RiskyGrowthGoalV6, None).is_err());
        for legacy in [PropFirmV4, RiskyGrowthV5] {
            assert_eq!(scoring_goal_bits_v2(legacy, None).unwrap(), [0; 5]);
            assert!(scoring_goal_bits_v2(legacy, Some(context())).is_err());
        }
        for value in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let invalid = [
                ResidentScoringGoalContextV2 {
                    initial_equity: value,
                    ..context()
                },
                ResidentScoringGoalContextV2 {
                    span_days: value,
                    ..context()
                },
                ResidentScoringGoalContextV2 {
                    goal: RiskyGrowthGoal {
                        start_balance: value,
                        ..context().goal
                    },
                    ..context()
                },
                ResidentScoringGoalContextV2 {
                    goal: RiskyGrowthGoal {
                        target_balance: value,
                        ..context().goal
                    },
                    ..context()
                },
                ResidentScoringGoalContextV2 {
                    goal: RiskyGrowthGoal {
                        horizon_days: value,
                        ..context().goal
                    },
                    ..context()
                },
            ];
            for context in invalid {
                assert!(scoring_goal_bits_v2(RiskyGrowthGoalV6, Some(context)).is_err());
            }
        }
        let mut invalid = context();
        invalid.goal.target_balance = invalid.goal.start_balance;
        assert!(scoring_goal_bits_v2(RiskyGrowthGoalV6, Some(invalid)).is_err());
    }

    fn plan(bits: [u64; 5]) -> RawResidentScoringPlanV2 {
        RawResidentScoringPlanV2 {
            abi_version: SCORING_PLAN_ABI_V2,
            scoring_objective: ResidentScoringObjectiveV2::RiskyGrowthGoalV6 as u32,
            scoring_version: 6,
            reserved: 0,
            logical_population_count: 8,
            feature_count: 8,
            max_terms_per_gene: 3,
            reserved_extents: 0,
            novelty_weight_bits: 0,
            initial_equity_bits: bits[0],
            span_days_bits: bits[1],
            goal_start_balance_bits: bits[2],
            goal_target_balance_bits: bits[3],
            goal_horizon_days_bits: bits[4],
            metric_semantics_sha256: [1; 32],
            scoring_semantics_sha256: scoring_semantics_sha256_v2(
                ResidentScoringObjectiveV2::RiskyGrowthGoalV6,
            ),
            novelty_semantics_sha256: novelty_disabled_semantics_sha256_v2(),
            scenario_order_semantics_sha256: [2; 32],
            gene_schema_sha256: [3; 32],
            rank_semantics_sha256: rank_semantics_sha256_v2(),
            #[cfg(not(feature = "hip-native-kernels"))]
            cuda_device_identity_sha256: [4; 32],
            #[cfg(feature = "hip-native-kernels")]
            hip_device_identity_sha256: [4; 32],
            #[cfg(not(feature = "hip-native-kernels"))]
            primary_context_identity_sha256: [5; 32],
            #[cfg(feature = "hip-native-kernels")]
            hip_lease_identity_sha256: [5; 32],
            run_stream_identity_sha256: [6; 32],
            #[cfg(not(feature = "hip-native-kernels"))]
            cuda_build_manifest_sha256: [7; 32],
            #[cfg(feature = "hip-native-kernels")]
            hip_build_manifest_sha256: [7; 32],
            #[cfg(not(feature = "hip-native-kernels"))]
            cuda_math_flags_sha256: cuda_math_flags_sha256_v2(),
            #[cfg(feature = "hip-native-kernels")]
            hip_math_flags_sha256: native_math_flags_sha256_v3(),
            plan_identity_sha256: [0; 32],
        }
    }

    #[test]
    fn goal_plan_hash_binds_every_context_bit_and_objective() {
        let bits = scoring_goal_bits_v2(
            ResidentScoringObjectiveV2::RiskyGrowthGoalV6,
            Some(context()),
        )
        .unwrap();
        let baseline = hash_scoring_plan_v2(&plan(bits));
        for field in 0..5 {
            let mut changed = bits;
            changed[field] ^= 1;
            assert_ne!(
                hash_scoring_plan_v2(&plan(changed)),
                baseline,
                "field {field}"
            );
        }
        for objective in [
            ResidentScoringObjectiveV2::PropFirmV4,
            ResidentScoringObjectiveV2::RiskyGrowthV5,
        ] {
            assert_ne!(
                scoring_semantics_sha256_v2(objective),
                scoring_semantics_sha256_v2(ResidentScoringObjectiveV2::RiskyGrowthGoalV6)
            );
            let mut legacy = plan([0; 5]);
            legacy.scoring_objective = objective as u32;
            legacy.scoring_version = 5;
            legacy.scoring_semantics_sha256 = scoring_semantics_sha256_v2(objective);
            assert_ne!(hash_scoring_plan_v2(&legacy), baseline);
        }
    }

    #[test]
    fn goal_plan_abi_extends_only_plan_and_retains_allocation_layout() {
        assert_eq!(std::mem::size_of::<RawResidentScoringPlanV2>(), 472);
        assert_eq!(
            std::mem::offset_of!(RawResidentScoringPlanV2, initial_equity_bits),
            48
        );
        assert_eq!(
            std::mem::offset_of!(RawResidentScoringPlanV2, goal_horizon_days_bits),
            80
        );
        assert_eq!(
            std::mem::offset_of!(RawResidentScoringPlanV2, metric_semantics_sha256),
            88
        );
        assert_eq!(
            std::mem::offset_of!(RawResidentScoringPlanV2, plan_identity_sha256),
            440
        );
        assert_eq!(
            std::mem::size_of::<RawResidentScoringAllocationReceiptV2>(),
            128
        );
        let receipt = RawResidentScoringAllocationReceiptV2 {
            cub_scratch_bytes: 4096,
            ..Default::default()
        };
        assert_eq!(receipt.cub_scratch_bytes_v2(), 4096);
    }
}
