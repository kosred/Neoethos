//! Shared ABI types for the resident CUDA or AMD HIP generation engine.
//!
//! The composite Search V2/V3 owner is the only Rust lifecycle authority. The
//! superseded standalone generation owner and its post-GA bridge were removed;
//! this module contains its ABI data and checked fixed generation geometry.
//! Sealing geometry is not an attestation of CPU adaptive-policy equivalence or
//! device execution. Philox oracles and synthetic constructors remain fixtures.

use crate::resident_archive_output_v3::RawResidentArchiveGeneScalarV3;
use sha2::{Digest, Sha256};

// One algorithm description; only the actual primitive/backend binding differs.
macro_rules! generation_semantics_v1 {
    ($primitives:literal, $build:literal) => {
        concat!(
            "neoethos.discovery-generation.v1;",
            "fixed-stride-normalized-gene;",
            "philox4x32-10-address-v1;",
            "decision-slot-high32-retry-low32;",
            "rank-weighted-parent-and-survivor-only;",
            "fixed-original-rank-integer-weights;",
            "sealed-u64-scoring-novelty-decision-key;",
            "metric-row-identity-only;",
            $primitives,
            $build,
            "same-admitted-stream;no-floating-decision-reduction;",
            "resident-global-full-gene-dedup;fnv4-resident-content;",
            "no-candidate-revival;no-host-decision"
        )
    };
}

#[cfg(any(feature = "cuda", test))]
pub const DISCOVERY_GENERATION_SEMANTICS_V1: &str = generation_semantics_v1!(
    "stable-cub-radix-u64-key-and-gene-identity-tie;",
    "cuda-cccl-toolkit-native-build-bound;"
);

#[cfg(any(feature = "hip-native-kernels", test))]
const DISCOVERY_GENERATION_HIP_SEMANTICS_V1: &str = generation_semantics_v1!(
    "stable-hipcub-rocprim-radix-u64-key-and-gene-identity-tie;",
    "amd-hip-native-build-bound;generation-abi=65537;"
);

const RESIDENT_METRIC_ROW_PROTOCOL_V2: &str = concat!(
    "neoethos.resident-metric-row.v2;repr-c;bytes=104;",
    "candidate-id=u64-resident-gene-identity;scenario-id=u64-sealed-scenario-identity;",
    "values=f64[11];",
    "slots=net,sharpe,peak,max-dd,win-rate,pf,expectancy,monthly-hit,trades,consistency,max-daily-dd"
);

#[cfg(feature = "cuda")]
const ABI_VERSION_V1: u32 = 1;
#[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
const ABI_VERSION_V1: u32 = 0x0001_0001;

/// Mirrors the selected protocol in the original native generation header.
/// This is a wire selector, not device or run authority.
pub(crate) const fn selected_generation_abi_v1() -> u32 {
    ABI_VERSION_V1
}
#[cfg(feature = "cuda-device-fixtures")]
const PARENT_RANK_WEIGHTED_V1: u32 = 1;
#[cfg(feature = "cuda-device-fixtures")]
const SURVIVOR_RANK_WEIGHTED_V1: u32 = 1;

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
    UnsupportedUniformSelection,
    UnsupportedTournamentSelection,
    UnsupportedSoftmaxSelection,
    UnsupportedElitistSelection,
    UnsupportedGenerationalSelection,
    InvalidPlan(&'static str),
    IdentityMismatch(&'static str),
    ArithmeticOverflow,
}

// CPU Philox oracles are used by unit and real-device unit tests only. The
// production resident algorithm executes its own native device implementation.
#[cfg(test)]
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhiloxDrawAddressV1 {
    counter: [u32; 4],
    key: [u32; 2],
}

#[cfg(test)]
impl PhiloxDrawAddressV1 {
    pub const fn counter(&self) -> [u32; 4] {
        self.counter
    }

    pub const fn key(&self) -> [u32; 2] {
        self.key
    }
}

#[cfg(test)]
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

#[cfg(test)]
pub fn checked_philox_rejection_draw_index_v1(decision_slot: u32, rejection_attempt: u32) -> u64 {
    (u64::from(decision_slot) << 32) | u64::from(rejection_attempt)
}

#[cfg(test)]
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
pub(crate) struct RawGenerationPlanV1 {
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
    #[cfg(feature = "cuda")]
    cuda_build_manifest_sha256: [u8; 32],
    #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
    hip_build_manifest_sha256: [u8; 32],
    rng_mapping_sha256: [u8; 32],
    plan_identity_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RawAllocationReceiptV1 {
    pub(crate) abi_version: u32,
    pub(crate) generation_store_allocation_count: u32,
    pub(crate) logical_gene_scalar_bytes: u64,
    pub(crate) logical_gene_index_bytes: u64,
    pub(crate) logical_gene_weight_bytes: u64,
    pub(crate) offspring_bytes: u64,
    pub(crate) metric_row_bytes: u64,
    pub(crate) rank_key_bytes: u64,
    pub(crate) selection_bytes: u64,
    pub(crate) dedup_hash_bytes: u64,
    pub(crate) cub_scratch_bytes: u64,
    pub(crate) retained_evaluation_workspace_bytes: u64,
    pub(crate) terminal_device_receipt_bytes: u64,
    pub(crate) total_device_bytes: u64,
    pub(crate) same_context_free_bytes: u64,
    pub(crate) full_discovery_reserve_bytes: u64,
    pub(crate) logical_population_count: u64,
    pub(crate) retained_evaluation_capacity: u64,
    pub(crate) generation_chunk_count: u64,
    pub(crate) allocation_plan_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct RawReadyEventV1 {
    pub(crate) abi_version: u32,
    pub(crate) reserved: u32,
    pub(crate) event_id: u64,
    pub(crate) generation_index: u64,
    pub(crate) same_stream_enqueue_count: u64,
    pub(crate) intermediate_host_wait_count: u64,
    pub(crate) intermediate_readback_count: u64,
}

const _: [(); 632] = [(); std::mem::size_of::<RawGenerationPlanV1>()];
const _: [(); 176] = [(); std::mem::size_of::<RawAllocationReceiptV1>()];
const _: [(); 48] = [(); std::mem::size_of::<RawReadyEventV1>()];

pub(crate) enum NativeResidentGenerationRunV1 {}

unsafe extern "C" {
    #[link_name = "initialize_resident_generation_population_v1"]
    pub(crate) fn ffi_initialize_resident_generation_population_v1(
        run: *mut NativeResidentGenerationRunV1,
        ready: *mut RawReadyEventV1,
    ) -> i32;
}

/// Caller-supplied fixed geometry and immutable identity bindings. No defaults
/// are supplied here: orchestration must resolve real run inputs and separately
/// reject policies not represented by this ABI (for example dynamic rescue,
/// convergence, survivor or immigrant schedules). The sealer validates geometry,
/// not the provenance of arbitrary identity bytes or CPU/native policy parity.
#[derive(Clone, Debug)]
pub struct ResidentGenerationPlanAuthorityInputV1 {
    pub parent_selection: ParentSelectionPolicyV1,
    pub survivor_selection: SurvivorSelectionPolicyV1,
    pub max_terms_per_gene: usize,
    pub minimum_terms_per_gene: usize,
    pub logical_population_count: usize,
    pub retained_evaluation_capacity: usize,
    pub feature_count: usize,
    pub generation_count: usize,
    pub survivor_count: usize,
    pub immigrant_count: usize,
    pub search_seed: u64,
    pub mutation_intensity_q32: u64,
    pub threshold_ladder_bits: [u64; 6],
    pub stop_bounds_bits: [u64; 6],
    pub smc_probability_q32: [u64; 11],
    pub generation_semantics_sha256: [u8; 32],
    pub run_identity_sha256: [u8; 32],
    pub strategy_gene_schema_sha256: [u8; 32],
    pub rank_semantics_sha256: [u8; 32],
    pub metric_semantics_sha256: [u8; 32],
    pub scoring_semantics_sha256: [u8; 32],
    pub novelty_semantics_sha256: [u8; 32],
    pub scenario_order_semantics_sha256: [u8; 32],
    #[cfg(feature = "cuda")]
    pub cuda_build_manifest_sha256: [u8; 32],
    #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
    pub hip_native_build_manifest_sha256: [u8; 32],
    pub rng_mapping_sha256: [u8; 32],
}

impl ResidentGenerationPlanAuthorityInputV1 {
    fn native_build_manifest_sha256_v1(&self) -> [u8; 32] {
        #[cfg(feature = "cuda")]
        {
            self.cuda_build_manifest_sha256
        }
        #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
        {
            self.hip_native_build_manifest_sha256
        }
    }
}

pub struct SealedResidentGenerationPlanV1 {
    raw: RawGenerationPlanV1,
    plan_identity_sha256: [u8; 32],
    adaptive: Option<AdaptiveGenerationControlsV3>,
}

impl SealedResidentGenerationPlanV1 {
    pub(crate) fn raw_adaptive_policy_v3(&self) -> Option<&RawResidentAdaptivePolicyV3> {
        self.adaptive.as_ref().map(|controls| &controls.policy)
    }

    pub(crate) fn adaptive_policy_identity_sha256_v3(&self) -> Option<[u8; 32]> {
        self.raw_adaptive_policy_v3()
            .map(|policy| policy.policy_identity_sha256)
    }

    pub(crate) fn adaptive_controls_v3(&self) -> Option<&AdaptiveGenerationControlsV3> {
        self.adaptive.as_ref()
    }

    pub(crate) const fn raw_plan_v1(&self) -> &RawGenerationPlanV1 {
        &self.raw
    }

    pub(crate) const fn logical_population_count_v1(&self) -> u64 {
        self.raw.logical_population_count
    }

    pub(crate) const fn feature_count_v1(&self) -> u64 {
        self.raw.feature_count
    }

    pub(crate) const fn generation_count_v1(&self) -> u64 {
        self.raw.generation_count
    }

    pub(crate) const fn retained_evaluation_capacity_v1(&self) -> u64 {
        self.raw.retained_evaluation_capacity
    }

    pub(crate) const fn max_terms_per_gene_v1(&self) -> u32 {
        self.raw.max_terms_per_gene
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub(crate) const fn survivor_count_v1(&self) -> u64 {
        self.raw.survivor_count
    }

    pub(crate) const fn generation_semantics_sha256_v1(&self) -> [u8; 32] {
        self.raw.generation_semantics_sha256
    }

    pub(crate) const fn plan_identity_sha256_v1(&self) -> [u8; 32] {
        self.plan_identity_sha256
    }

    pub(crate) const fn run_identity_sha256_v1(&self) -> [u8; 32] {
        self.raw.run_identity_sha256
    }

    pub(crate) const fn strategy_gene_schema_sha256_v1(&self) -> [u8; 32] {
        self.raw.strategy_gene_schema_sha256
    }

    pub(crate) const fn rank_semantics_sha256_v1(&self) -> [u8; 32] {
        self.raw.rank_semantics_sha256
    }

    pub(crate) const fn metric_semantics_sha256_v1(&self) -> [u8; 32] {
        self.raw.metric_semantics_sha256
    }

    pub(crate) const fn scoring_semantics_sha256_v1(&self) -> [u8; 32] {
        self.raw.scoring_semantics_sha256
    }

    pub(crate) const fn novelty_semantics_sha256_v1(&self) -> [u8; 32] {
        self.raw.novelty_semantics_sha256
    }

    pub(crate) const fn scenario_order_semantics_sha256_v1(&self) -> [u8; 32] {
        self.raw.scenario_order_semantics_sha256
    }

    #[cfg(feature = "cuda")]
    pub(crate) const fn cuda_build_manifest_sha256_v1(&self) -> [u8; 32] {
        self.raw.cuda_build_manifest_sha256
    }

    pub(crate) const fn native_build_manifest_sha256_v1(&self) -> [u8; 32] {
        #[cfg(feature = "cuda")]
        {
            self.cuda_build_manifest_sha256_v1()
        }
        #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
        {
            self.raw.hip_build_manifest_sha256
        }
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub(crate) fn resident_search_fixture_v2(
        logical_population_count: usize,
        feature_count: usize,
    ) -> Self {
        Self::resident_search_scoring_fixture_v2(
            logical_population_count,
            feature_count,
            crate::resident_scoring_v2::ResidentScoringObjectiveV2::PropFirmV4,
        )
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub(crate) fn resident_search_scoring_fixture_v2(
        logical_population_count: usize,
        feature_count: usize,
        objective: crate::resident_scoring_v2::ResidentScoringObjectiveV2,
    ) -> Self {
        let max_terms_per_gene = feature_count.min(3) as u32;
        let threshold_ladder_bits =
            std::array::from_fn(|index| (0.05 * (index as f64 + 1.0)).to_bits());
        let stop_bounds_bits = std::array::from_fn(|index| (index as f64 + 1.0).to_bits());
        let plan_identity_sha256 = [0x6b; 32];
        Self {
            raw: RawGenerationPlanV1 {
                abi_version: ABI_VERSION_V1,
                parent_selection_policy: PARENT_RANK_WEIGHTED_V1,
                survivor_selection_policy: SURVIVOR_RANK_WEIGHTED_V1,
                max_terms_per_gene,
                minimum_terms_per_gene: 1,
                threshold_level_count: 6,
                smc_flag_count: 11,
                reserved: 0,
                logical_population_count: logical_population_count as u64,
                retained_evaluation_capacity: logical_population_count as u64,
                feature_count: feature_count as u64,
                generation_count: 2,
                survivor_count: 1,
                immigrant_count: 0,
                search_seed: 0x1234_5678_9abc_def0,
                mutation_intensity_q32: 1_u64 << 32,
                threshold_ladder_bits,
                stop_bounds_bits,
                smc_probability_q32: [0; 11],
                generation_semantics_sha256: [0x61; 32],
                run_identity_sha256: [0x62; 32],
                strategy_gene_schema_sha256: [0x63; 32],
                rank_semantics_sha256: crate::resident_scoring_v2::rank_semantics_sha256_v2(),
                metric_semantics_sha256: resident_metric_semantics_sha256_v2(),
                scoring_semantics_sha256: crate::resident_scoring_v2::scoring_semantics_sha256_v2(
                    objective,
                ),
                novelty_semantics_sha256:
                    crate::resident_scoring_v2::novelty_disabled_semantics_sha256_v2(),
                scenario_order_semantics_sha256: [0x68; 32],
                cuda_build_manifest_sha256: [0x69; 32],
                rng_mapping_sha256: [0x6a; 32],
                plan_identity_sha256,
            },
            plan_identity_sha256,
            adaptive: None,
        }
    }
}

/// Seal checked fixed geometry without granting device admission or claiming
/// unsupported CPU adaptive policies have been implemented by the native plan.
pub fn seal_resident_generation_plan_v1(
    input: ResidentGenerationPlanAuthorityInputV1,
) -> Result<SealedResidentGenerationPlanV1, ResidentGenerationDeviceErrorV1> {
    validate_rank_weighted_only_v1(input.parent_selection, input.survivor_selection)?;
    seal_generation_geometry(input, discovery_generation_semantics_sha256_v1(), true)
}

fn seal_generation_geometry(
    input: ResidentGenerationPlanAuthorityInputV1,
    expected_semantics: [u8; 32],
    thresholds_strictly_increasing: bool,
) -> Result<SealedResidentGenerationPlanV1, ResidentGenerationDeviceErrorV1> {
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
        || identity_is_zero_v1(&input.native_build_manifest_sha256_v1())
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
    if input.generation_semantics_sha256 != expected_semantics {
        return Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
            "generation semantics",
        ));
    }
    if input.metric_semantics_sha256 != resident_metric_semantics_sha256_v2() {
        return Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
            "resident metric semantics",
        ));
    }
    validate_f64_plan_bits_v1(&input.threshold_ladder_bits, thresholds_strictly_increasing)?;
    // Adaptive percentiles may tie. Preserve their exact values rather than
    // sorting, deduplicating, or perturbing the caller's resolved ladder.
    if !thresholds_strictly_increasing
        && (input
            .threshold_ladder_bits
            .iter()
            .any(|bits| f64::from_bits(*bits) <= 0.0)
            || input
                .threshold_ladder_bits
                .windows(2)
                .any(|pair| f64::from_bits(pair[1]) < f64::from_bits(pair[0])))
    {
        return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
            "adaptive threshold ladder must be positive and nondecreasing",
        ));
    }
    validate_f64_plan_bits_v1(&input.stop_bounds_bits, false)?;

    let mut raw = RawGenerationPlanV1 {
        abi_version: selected_generation_abi_v1(),
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
        #[cfg(feature = "cuda")]
        cuda_build_manifest_sha256: input.cuda_build_manifest_sha256,
        #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
        hip_build_manifest_sha256: input.hip_native_build_manifest_sha256,
        rng_mapping_sha256: input.rng_mapping_sha256,
        plan_identity_sha256: [0; 32],
    };
    let plan_identity_sha256 = hash_raw_plan_v1(&raw);
    raw.plan_identity_sha256 = plan_identity_sha256;
    Ok(SealedResidentGenerationPlanV1 {
        raw,
        plan_identity_sha256,
        adaptive: None,
    })
}

const DISCOVERY_ADAPTIVE_GENERATION_ALGORITHM_V3: &str = concat!(
    "neoethos.discovery-adaptive-generation.v3;algorithm=1;",
    "resident-philox;rank-uniform-softmax-tournament-parent;",
    "rank-elitist-tournament-generational-survivor;",
    "configured-survivor-immigrant-stagnation-rescue-mutation;",
    "signed-threshold-ladder;tp-sl-vol-distinct-bounds;",
    "smc-gate-progress-stagnation;ordered-templates;bounded-seen-retries;",
    "full-resident-population;checkpoint-control-only;terminal-last-evaluated"
);

#[cfg(any(feature = "hip-native-kernels", test))]
fn adaptive_hip_generation_semantics_sha256_v3() -> [u8; 32] {
    sha256_v1(&[
        b"neoethos.discovery-adaptive-generation.amd-hip.v1;",
        DISCOVERY_ADAPTIVE_GENERATION_ALGORITHM_V3.as_bytes(),
        b"hipcub-rocprim;hip-native-build-bound;generation-abi=65537;",
    ])
}

/// Explicitly versioned GPU algorithm and selected backend semantics. CUDA
/// retains its original identity. HIP binds the same algorithm to AMD HIP,
/// hipCUB/rocPRIM and the distinct native protocol. The actual build digest is
/// separately bound by the plan; neither hash attests to device execution.
pub fn discovery_adaptive_generation_semantics_sha256_v3() -> [u8; 32] {
    #[cfg(feature = "cuda")]
    {
        sha256_v1(&[DISCOVERY_ADAPTIVE_GENERATION_ALGORITHM_V3.as_bytes()])
    }
    #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
    {
        adaptive_hip_generation_semantics_sha256_v3()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RawResidentAdaptivePolicyV3 {
    pub(crate) abi_version: u32,
    pub(crate) algorithm_version: u32,
    pub(crate) parent_policy: u32,
    pub(crate) survivor_policy: u32,
    pub(crate) tournament_size: u32,
    pub(crate) min_structural_smc_flags: u32,
    pub(crate) adaptive_stops_enabled: u32,
    pub(crate) reserved: u32,
    pub(crate) seen_capacity: u64,
    pub(crate) seen_retry_attempts: u64,
    pub(crate) seen_initial_count: u64,
    pub(crate) seed_template_count: u64,
    pub(crate) template_count: u64,
    pub(crate) soft_stagnation_patience: u64,
    pub(crate) survivor_fraction: f64,
    pub(crate) immigrant_fraction: f64,
    pub(crate) selection_temperature: f64,
    pub(crate) minimum_improvement: f64,
    pub(crate) gate_start: f64,
    pub(crate) gate_end: f64,
    pub(crate) gate_curve: f64,
    pub(crate) gate_stagnation_step: f64,
    pub(crate) smc_force_ratio: f64,
    pub(crate) run_identity_sha256: [u8; 32],
    pub(crate) policy_identity_sha256: [u8; 32],
}

const _: [(); 216] = [(); std::mem::size_of::<RawResidentAdaptivePolicyV3>()];
const _: [(); 8] = [(); std::mem::align_of::<RawResidentAdaptivePolicyV3>()];
const _: [(); 80] = [(); std::mem::offset_of!(RawResidentAdaptivePolicyV3, survivor_fraction)];
const _: [(); 184] =
    [(); std::mem::offset_of!(RawResidentAdaptivePolicyV3, policy_identity_sha256)];

/// Ordinary owned template content. No pointer, receipt or execution authority
/// can be supplied here. Values are validated and preserved, never normalized.
#[derive(Clone, Debug)]
pub struct ResidentGenerationTemplateV3 {
    pub feature_indices: Vec<u64>,
    pub weights: Vec<f64>,
    pub smc_flags: u32,
    pub long_threshold: f64,
    pub short_threshold: f64,
    pub target_pips: f64,
    pub stop_pips: f64,
    pub stop_vol_multiplier: f64,
}

/// Resolved run configuration and bounded immutable seed controls. Runtime
/// cancellation and wall-clock stopping remain with the orchestration owner.
#[derive(Clone, Debug)]
pub struct ResidentAdaptiveGenerationInputsV3 {
    pub tournament_size: usize,
    pub min_structural_smc_flags: u32,
    pub adaptive_stops_enabled: bool,
    pub seen_capacity: usize,
    pub seen_retry_attempts: usize,
    pub seed_template_count: usize,
    pub soft_stagnation_patience: usize,
    pub survivor_fraction: f64,
    pub immigrant_fraction: f64,
    pub selection_temperature: f64,
    pub minimum_improvement: f64,
    pub gate_start: f64,
    pub gate_end: f64,
    pub gate_curve: f64,
    pub gate_stagnation_step: f64,
    pub smc_force_ratio: f64,
    pub templates: Vec<ResidentGenerationTemplateV3>,
    pub initial_seen_hashes: Vec<u64>,
}

pub(crate) struct AdaptiveGenerationControlsV3 {
    pub(crate) policy: RawResidentAdaptivePolicyV3,
    pub(crate) template_scalars: Vec<RawResidentArchiveGeneScalarV3>,
    pub(crate) template_indices: Vec<u64>,
    pub(crate) template_weights: Vec<f64>,
    pub(crate) initial_seen_hashes: Vec<u64>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RawResidentAdaptiveCheckpointV3 {
    pub(crate) abi_version: u32,
    pub(crate) algorithm_version: u32,
    pub(crate) run_identity: u64,
    pub(crate) evaluated_generation: u64,
    pub(crate) evaluated_generations: u64,
    pub(crate) evaluation_slots: u64,
    pub(crate) evaluated_gate_bits: u64,
    pub(crate) stagnant_generations: u64,
    pub(crate) best_score_bits: u64,
    pub(crate) survivor_count: u64,
    pub(crate) immigrant_count: u64,
    pub(crate) rescue_count: u64,
    pub(crate) mutation_count: u32,
    pub(crate) reserved: u32,
    pub(crate) mutation_intensity: f64,
    pub(crate) control_copy_count: u64,
    pub(crate) control_copy_bytes: u64,
    pub(crate) initial_upload_count: u64,
    pub(crate) initial_upload_bytes: u64,
}
const _: [(); 136] = [(); std::mem::size_of::<RawResidentAdaptiveCheckpointV3>()];

/// Small device-produced control state. Population genomes, feature matrices
/// and metric rows stay resident until terminal export. Only the native owner
/// can seal this value after checking its actual stream and generation.
#[derive(Clone, Copy, Debug)]
pub struct ResidentAdaptiveCheckpointV3 {
    raw: RawResidentAdaptiveCheckpointV3,
}

impl ResidentAdaptiveCheckpointV3 {
    pub(crate) fn seal_v3(
        raw: RawResidentAdaptiveCheckpointV3,
        run_identity: u64,
        completed_generations: u64,
        population: u64,
    ) -> Result<Self, ResidentGenerationDeviceErrorV1> {
        let best = f64::from_bits(raw.best_score_bits);
        if raw.abi_version != 3
            || raw.algorithm_version != 1
            || raw.reserved != 0
            || raw.run_identity != run_identity
            || completed_generations == 0
            || population == 0
            || raw.evaluated_generations != completed_generations
            || raw.evaluated_generation.checked_add(1) != Some(completed_generations)
            || population.checked_mul(completed_generations) != Some(raw.evaluation_slots)
            || !f64::from_bits(raw.evaluated_gate_bits).is_finite()
            || (!best.is_finite() && best != f64::NEG_INFINITY)
            || raw.stagnant_generations > completed_generations
            || raw
                .survivor_count
                .checked_add(raw.immigrant_count)
                .and_then(|count| count.checked_add(raw.rescue_count))
                .is_none_or(|count| count > population)
            || !(1..=3).contains(&raw.mutation_count)
            || !raw.mutation_intensity.is_finite()
            || raw.mutation_intensity <= 0.0
            || raw.control_copy_count == 0
            || raw.control_copy_count % 2 != 0
            || raw.control_copy_count.checked_mul(92) != Some(raw.control_copy_bytes)
            || raw.initial_upload_count > 4
            || ((raw.initial_upload_count == 0) != (raw.initial_upload_bytes == 0))
        {
            return Err(ResidentGenerationDeviceErrorV1::InvalidPlan(
                "invalid native adaptive checkpoint",
            ));
        }
        Ok(Self { raw })
    }
    pub const fn run_identity(&self) -> u64 {
        self.raw.run_identity
    }
    pub const fn evaluated_generation(&self) -> u64 {
        self.raw.evaluated_generation
    }
    pub const fn evaluated_generations(&self) -> u64 {
        self.raw.evaluated_generations
    }
    pub const fn evaluation_slots(&self) -> u64 {
        self.raw.evaluation_slots
    }
    pub fn evaluated_gate(&self) -> f64 {
        f64::from_bits(self.raw.evaluated_gate_bits)
    }
    pub const fn stagnant_generations(&self) -> u64 {
        self.raw.stagnant_generations
    }
    pub fn best_score(&self) -> f64 {
        f64::from_bits(self.raw.best_score_bits)
    }
    pub const fn survivor_count(&self) -> u64 {
        self.raw.survivor_count
    }
    pub const fn immigrant_count(&self) -> u64 {
        self.raw.immigrant_count
    }
    pub const fn rescue_count(&self) -> u64 {
        self.raw.rescue_count
    }
    pub const fn mutation_count(&self) -> u32 {
        self.raw.mutation_count
    }
    pub const fn mutation_intensity(&self) -> f64 {
        self.raw.mutation_intensity
    }
    pub const fn control_copy_count(&self) -> u64 {
        self.raw.control_copy_count
    }
    pub const fn control_copy_bytes(&self) -> u64 {
        self.raw.control_copy_bytes
    }
    pub const fn initial_upload_count(&self) -> u64 {
        self.raw.initial_upload_count
    }
    pub const fn initial_upload_bytes(&self) -> u64 {
        self.raw.initial_upload_bytes
    }
}

pub fn seal_adaptive_resident_generation_plan_v3(
    input: ResidentGenerationPlanAuthorityInputV1,
    controls: ResidentAdaptiveGenerationInputsV3,
) -> Result<SealedResidentGenerationPlanV1, ResidentGenerationDeviceErrorV1> {
    let mut plan = seal_generation_geometry(
        input,
        discovery_adaptive_generation_semantics_sha256_v3(),
        false,
    )?;
    let invalid = ResidentGenerationDeviceErrorV1::InvalidPlan;
    let unit = |value: f64| value.is_finite() && (0.0..=1.0).contains(&value);
    if controls.tournament_size < 2
        || controls.tournament_size > u32::MAX as usize
        || controls.min_structural_smc_flags > 10
        || controls.seen_retry_attempts == 0
        || controls.seen_retry_attempts > (u32::MAX / 256) as usize
        || controls.initial_seen_hashes.len() > controls.seen_capacity
        || controls.seed_template_count > controls.templates.len()
        || controls.templates.len() > 50
        || plan.raw.max_terms_per_gene > 16
        || controls.seed_template_count as u64 > plan.raw.logical_population_count / 10
        || !unit(controls.survivor_fraction)
        || !unit(controls.immigrant_fraction)
        || controls.survivor_fraction > 0.95
        || controls.immigrant_fraction > 0.95
        || !controls.gate_start.is_finite()
        || !controls.gate_end.is_finite()
        || !unit(controls.smc_force_ratio)
        || !controls.selection_temperature.is_finite()
        || controls.selection_temperature <= 0.0
        || !controls.minimum_improvement.is_finite()
        || controls.minimum_improvement < 0.0
        || !controls.gate_curve.is_finite()
        || controls.gate_curve <= 0.0
        || !controls.gate_stagnation_step.is_finite()
        || controls.gate_stagnation_step < 0.0
    {
        return Err(invalid("invalid adaptive generation controls"));
    }
    if plan.raw.stop_bounds_bits.chunks_exact(2).any(|pair| {
        let lo = f64::from_bits(pair[0]);
        let hi = f64::from_bits(pair[1]);
        lo <= 0.0 || hi < lo
    }) {
        return Err(invalid("invalid adaptive initialization bounds"));
    }
    let stride = plan.raw.max_terms_per_gene as usize;
    let term_extent = controls
        .templates
        .len()
        .checked_mul(stride)
        .ok_or(ResidentGenerationDeviceErrorV1::ArithmeticOverflow)?;
    let mut scalars = Vec::new();
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    scalars
        .try_reserve_exact(controls.templates.len())
        .map_err(|_| invalid("template allocation failed"))?;
    indices
        .try_reserve_exact(term_extent)
        .map_err(|_| invalid("template allocation failed"))?;
    weights
        .try_reserve_exact(term_extent)
        .map_err(|_| invalid("template allocation failed"))?;
    for (ordinal, template) in controls.templates.iter().enumerate() {
        let count = template.feature_indices.len();
        if count == 0
            || count > stride
            || template.weights.len() != count
            || template.smc_flags & !0x7ff != 0
            || !template.long_threshold.is_finite()
            || !template.short_threshold.is_finite()
            || template.long_threshold <= template.short_threshold
            || !template.target_pips.is_finite()
            || template.target_pips <= 0.0
            || !template.stop_pips.is_finite()
            || template.stop_pips <= 0.0
            || !template.stop_vol_multiplier.is_finite()
            || template.stop_vol_multiplier < 0.0
            || template
                .feature_indices
                .iter()
                .any(|index| *index >= plan.raw.feature_count)
            || template.weights.iter().any(|weight| !weight.is_finite())
            || template
                .feature_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid("invalid adaptive template genome"));
        }
        scalars.push(RawResidentArchiveGeneScalarV3 {
            gene_identity: ordinal as u64,
            content_hash: 0,
            term_count: count as u32,
            smc_flags: template.smc_flags,
            long_threshold: template.long_threshold,
            short_threshold: template.short_threshold,
            target_pips: template.target_pips,
            stop_pips: template.stop_pips,
            stop_vol_multiplier: template.stop_vol_multiplier,
            generation: 0,
            reserved: 0,
        });
        indices.extend_from_slice(&template.feature_indices);
        weights.extend_from_slice(&template.weights);
        indices.resize(indices.len() + stride - count, 0);
        weights.resize(weights.len() + stride - count, 0.0);
    }
    let mut policy = RawResidentAdaptivePolicyV3 {
        abi_version: 3,
        algorithm_version: 1,
        parent_policy: match plan.raw.parent_selection_policy {
            1 => 1,
            2 => 2,
            3 => 4,
            4 => 3,
            _ => unreachable!(),
        },
        survivor_policy: plan.raw.survivor_selection_policy,
        tournament_size: checked_u32_v1(controls.tournament_size)?,
        min_structural_smc_flags: controls.min_structural_smc_flags,
        adaptive_stops_enabled: u32::from(controls.adaptive_stops_enabled),
        reserved: 0,
        seen_capacity: checked_u64_v1(controls.seen_capacity)?,
        seen_retry_attempts: checked_u64_v1(controls.seen_retry_attempts)?,
        seen_initial_count: checked_u64_v1(controls.initial_seen_hashes.len())?,
        seed_template_count: checked_u64_v1(controls.seed_template_count)?,
        template_count: checked_u64_v1(controls.templates.len())?,
        soft_stagnation_patience: checked_u64_v1(controls.soft_stagnation_patience)?,
        survivor_fraction: controls.survivor_fraction,
        immigrant_fraction: controls.immigrant_fraction,
        selection_temperature: controls.selection_temperature,
        minimum_improvement: controls.minimum_improvement,
        gate_start: controls.gate_start,
        gate_end: controls.gate_end,
        gate_curve: controls.gate_curve,
        gate_stagnation_step: controls.gate_stagnation_step,
        smc_force_ratio: controls.smc_force_ratio,
        run_identity_sha256: plan.raw.run_identity_sha256,
        policy_identity_sha256: [0; 32],
    };
    let mut hash = Sha256::new();
    hash.update(b"neoethos.resident-adaptive-policy.v3\0");
    hash.update(plan.plan_identity_sha256);
    for value in [
        policy.abi_version,
        policy.algorithm_version,
        policy.parent_policy,
        policy.survivor_policy,
        policy.tournament_size,
        policy.min_structural_smc_flags,
        policy.adaptive_stops_enabled,
        policy.reserved,
    ] {
        hash.update(value.to_le_bytes());
    }
    for value in [
        policy.seen_capacity,
        policy.seen_retry_attempts,
        policy.seen_initial_count,
        policy.seed_template_count,
        policy.template_count,
        policy.soft_stagnation_patience,
    ] {
        hash.update(value.to_le_bytes());
    }
    for value in [
        policy.survivor_fraction,
        policy.immigrant_fraction,
        policy.selection_temperature,
        policy.minimum_improvement,
        policy.gate_start,
        policy.gate_end,
        policy.gate_curve,
        policy.gate_stagnation_step,
        policy.smc_force_ratio,
    ] {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.update(policy.run_identity_sha256);
    for scalar in &scalars {
        hash.update(scalar.gene_identity.to_le_bytes());
        hash.update(scalar.content_hash.to_le_bytes());
        hash.update(scalar.term_count.to_le_bytes());
        hash.update(scalar.smc_flags.to_le_bytes());
        for value in [
            scalar.long_threshold,
            scalar.short_threshold,
            scalar.target_pips,
            scalar.stop_pips,
            scalar.stop_vol_multiplier,
        ] {
            hash.update(value.to_bits().to_le_bytes());
        }
        hash.update(scalar.generation.to_le_bytes());
        hash.update(scalar.reserved.to_le_bytes());
    }
    for value in &indices {
        hash.update(value.to_le_bytes());
    }
    for value in &weights {
        hash.update(value.to_bits().to_le_bytes());
    }
    for value in &controls.initial_seen_hashes {
        hash.update(value.to_le_bytes());
    }
    policy.policy_identity_sha256 = hash.finalize().into();
    plan.adaptive = Some(AdaptiveGenerationControlsV3 {
        policy,
        template_scalars: scalars,
        template_indices: indices,
        template_weights: weights,
        initial_seen_hashes: controls.initial_seen_hashes,
    });
    Ok(plan)
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
    #[cfg(feature = "cuda")]
    hasher.update(plan.cuda_build_manifest_sha256);
    #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
    hasher.update(plan.hip_build_manifest_sha256);
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

pub fn discovery_generation_semantics_sha256_v1() -> [u8; 32] {
    #[cfg(feature = "cuda")]
    {
        sha256_v1(&[DISCOVERY_GENERATION_SEMANTICS_V1.as_bytes()])
    }
    #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
    {
        sha256_v1(&[DISCOVERY_GENERATION_HIP_SEMANTICS_V1.as_bytes()])
    }
}

/// Current resident row protocol and producer/scorer rejection interpretation.
/// This identity does not authenticate caller-supplied rows or grant device
/// admission: the actual sealed producer and run/scenario bindings remain required.
pub fn resident_metric_semantics_sha256_v2() -> [u8; 32] {
    sha256_v1(&[
        RESIDENT_METRIC_ROW_PROTOCOL_V2.as_bytes(),
        neoethos_gpu_contracts::resident_search_scoring_v2::RESIDENT_ECONOMIC_REJECTION_V2_SEMANTICS
            .as_bytes(),
    ])
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

#[cfg(test)]
mod tests {
    use super::*;

    fn checked_geometry_fixture_v1() -> ResidentGenerationPlanAuthorityInputV1 {
        ResidentGenerationPlanAuthorityInputV1 {
            parent_selection: ParentSelectionPolicyV1::RankWeighted,
            survivor_selection: SurvivorSelectionPolicyV1::RankWeighted,
            max_terms_per_gene: 4,
            minimum_terms_per_gene: 2,
            logical_population_count: 37,
            retained_evaluation_capacity: 37,
            feature_count: 19,
            generation_count: 23,
            survivor_count: 7,
            immigrant_count: 11,
            search_seed: 981_234,
            mutation_intensity_q32: 1_u64 << 30,
            threshold_ladder_bits: [0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6].map(f64::to_bits),
            stop_bounds_bits: [0.25_f64, 50.0, 0.5, 100.0, 1.0, 4.0].map(f64::to_bits),
            smc_probability_q32: std::array::from_fn(|index| (index as u64) << 27),
            generation_semantics_sha256: discovery_generation_semantics_sha256_v1(),
            run_identity_sha256: [1; 32],
            strategy_gene_schema_sha256: [2; 32],
            rank_semantics_sha256: [3; 32],
            metric_semantics_sha256: resident_metric_semantics_sha256_v2(),
            scoring_semantics_sha256: [5; 32],
            novelty_semantics_sha256: [6; 32],
            scenario_order_semantics_sha256: [7; 32],
            #[cfg(feature = "cuda")]
            cuda_build_manifest_sha256: [8; 32],
            #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
            hip_native_build_manifest_sha256: [8; 32],
            rng_mapping_sha256: [9; 32],
        }
    }

    fn adaptive_controls_fixture_v3() -> ResidentAdaptiveGenerationInputsV3 {
        ResidentAdaptiveGenerationInputsV3 {
            tournament_size: 3,
            min_structural_smc_flags: 2,
            adaptive_stops_enabled: true,
            seen_capacity: 100,
            seen_retry_attempts: 3,
            seed_template_count: 1,
            soft_stagnation_patience: 7,
            survivor_fraction: 0.8,
            immigrant_fraction: 0.7,
            selection_temperature: 0.75,
            minimum_improvement: 1e-8,
            gate_start: 1.25,
            gate_end: -0.1,
            gate_curve: 1.5,
            gate_stagnation_step: 0.03,
            smc_force_ratio: 0.6,
            templates: vec![ResidentGenerationTemplateV3 {
                feature_indices: vec![2, 18],
                weights: vec![-0.0, -0.4],
                smc_flags: 0x401,
                long_threshold: 3.0,
                short_threshold: -3.0,
                target_pips: 150.0,
                stop_pips: 75.0,
                stop_vol_multiplier: 2.5,
            }],
            initial_seen_hashes: vec![0, 9, 9, u64::MAX],
        }
    }

    fn adaptive_geometry_fixture_v3() -> ResidentGenerationPlanAuthorityInputV1 {
        let mut input = checked_geometry_fixture_v1();
        input.generation_semantics_sha256 = discovery_adaptive_generation_semantics_sha256_v3();
        input
    }

    #[test]
    fn selected_generation_backend_preserves_cuda_hashes_and_separates_hip_semantics() {
        let cuda = sha256_v1(&[DISCOVERY_GENERATION_SEMANTICS_V1.as_bytes()]);
        let cuda_adaptive = sha256_v1(&[DISCOVERY_ADAPTIVE_GENERATION_ALGORITHM_V3.as_bytes()]);
        // Frozen pre-HIP identities: changing backend selection must not change
        // existing CUDA receipts or silently relabel their algorithm.
        assert_eq!(
            cuda,
            [
                0x9f, 0xb1, 0xd7, 0xbf, 0xbf, 0x6c, 0x12, 0x2a, 0xdc, 0x16, 0x7b, 0x4b, 0x34, 0xdb,
                0xf4, 0x08, 0xe5, 0x21, 0x7a, 0xe8, 0x65, 0xfd, 0xaf, 0xc6, 0x3f, 0x68, 0xd4, 0x0a,
                0x8c, 0x43, 0xd6, 0xab,
            ]
        );
        assert_eq!(
            cuda_adaptive,
            [
                0x8d, 0x0b, 0x1f, 0xab, 0xdc, 0xe6, 0x0d, 0x6d, 0xe6, 0xee, 0x21, 0xee, 0xb4, 0x0e,
                0x45, 0x9a, 0x8f, 0xb6, 0x0e, 0x91, 0x6f, 0xb4, 0x07, 0x7f, 0x2b, 0xb0, 0xcc, 0xe1,
                0xd2, 0xf0, 0xa8, 0x26,
            ]
        );
        let hip = sha256_v1(&[DISCOVERY_GENERATION_HIP_SEMANTICS_V1.as_bytes()]);
        let hip_adaptive = adaptive_hip_generation_semantics_sha256_v3();
        assert_ne!(cuda, hip);
        assert_ne!(cuda_adaptive, hip_adaptive);
        assert!(DISCOVERY_GENERATION_HIP_SEMANTICS_V1.contains("amd-hip"));
        assert!(DISCOVERY_GENERATION_HIP_SEMANTICS_V1.contains("hipcub-rocprim"));
        assert!(!DISCOVERY_GENERATION_HIP_SEMANTICS_V1.contains("cuda"));
        assert!(!DISCOVERY_GENERATION_HIP_SEMANTICS_V1.contains("cccl"));
        let (abi, selected, selected_adaptive, other, other_adaptive) = if cfg!(feature = "cuda") {
            (1, cuda, cuda_adaptive, hip, hip_adaptive)
        } else {
            (0x0001_0001, hip, hip_adaptive, cuda, cuda_adaptive)
        };
        assert_eq!(selected_generation_abi_v1(), abi);
        assert_eq!(discovery_generation_semantics_sha256_v1(), selected);
        assert_eq!(
            discovery_adaptive_generation_semantics_sha256_v3(),
            selected_adaptive
        );
        let mut wrong_backend = checked_geometry_fixture_v1();
        wrong_backend.generation_semantics_sha256 = other;
        assert!(matches!(
            seal_resident_generation_plan_v1(wrong_backend),
            Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
                "generation semantics"
            ))
        ));
        let mut wrong_backend = adaptive_geometry_fixture_v3();
        wrong_backend.generation_semantics_sha256 = other_adaptive;
        assert!(matches!(
            seal_adaptive_resident_generation_plan_v3(
                wrong_backend,
                adaptive_controls_fixture_v3()
            ),
            Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
                "generation semantics"
            ))
        ));
    }

    #[test]
    fn selected_generation_build_digest_uses_exact_native_slot_and_binds_the_plan() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<RawGenerationPlanV1>(), 632);
        assert_eq!(align_of::<RawGenerationPlanV1>(), 8);
        #[cfg(feature = "cuda")]
        assert_eq!(
            offset_of!(RawGenerationPlanV1, cuda_build_manifest_sha256),
            536
        );
        #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
        assert_eq!(
            offset_of!(RawGenerationPlanV1, hip_build_manifest_sha256),
            536
        );
        assert_eq!(offset_of!(RawGenerationPlanV1, rng_mapping_sha256), 568);
        assert_eq!(offset_of!(RawGenerationPlanV1, plan_identity_sha256), 600);
        let base = seal_resident_generation_plan_v1(checked_geometry_fixture_v1()).unwrap();
        assert_eq!(base.raw.abi_version, selected_generation_abi_v1());
        assert_eq!(base.native_build_manifest_sha256_v1(), [8; 32]);
        #[cfg(feature = "cuda")]
        assert_eq!(base.cuda_build_manifest_sha256_v1(), [8; 32]);
        for digest in [[0; 32], [9; 32]] {
            let mut input = checked_geometry_fixture_v1();
            #[cfg(feature = "cuda")]
            {
                input.cuda_build_manifest_sha256 = digest;
            }
            #[cfg(all(not(feature = "cuda"), feature = "hip-native-kernels"))]
            {
                input.hip_native_build_manifest_sha256 = digest;
            }
            if digest == [0; 32] {
                assert!(seal_resident_generation_plan_v1(input).is_err());
            } else {
                let changed = seal_resident_generation_plan_v1(input).unwrap();
                assert_eq!(changed.native_build_manifest_sha256_v1(), digest);
                assert_ne!(
                    changed.plan_identity_sha256_v1(),
                    base.plan_identity_sha256_v1()
                );
            }
        }
    }

    #[test]
    fn adaptive_threshold_percentile_ties_preserve_exact_bits_and_identity() {
        let mut identities = Vec::new();
        for ladder in [
            [0.1_f64, 0.1, 0.3, 0.3, 0.5, 0.5],
            [0.25_f64; 6],
            [0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6],
        ] {
            let exact_bits = ladder.map(f64::to_bits);
            let mut input = adaptive_geometry_fixture_v3();
            input.threshold_ladder_bits = exact_bits;
            let plan =
                seal_adaptive_resident_generation_plan_v3(input, adaptive_controls_fixture_v3())
                    .expect("positive nondecreasing adaptive percentiles are valid");
            assert_eq!(plan.raw.threshold_ladder_bits, exact_bits);
            let identity = plan.adaptive_policy_identity_sha256_v3().unwrap();
            assert!(!identities.contains(&identity));
            identities.push(identity);
        }
        let mut legacy = checked_geometry_fixture_v1();
        legacy.threshold_ladder_bits = [0.25_f64.to_bits(); 6];
        assert!(seal_resident_generation_plan_v1(legacy).is_err());
    }

    #[test]
    fn adaptive_thresholds_reject_descending_zero_and_nonfinite_values_without_repair() {
        for invalid in [
            0.0_f64,
            -0.0,
            -0.1,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            for index in 0..6 {
                let mut input = adaptive_geometry_fixture_v3();
                input.threshold_ladder_bits = [0.25_f64.to_bits(); 6];
                input.threshold_ladder_bits[index] = invalid.to_bits();
                assert!(
                    seal_adaptive_resident_generation_plan_v3(
                        input,
                        adaptive_controls_fixture_v3(),
                    )
                    .is_err(),
                    "invalid threshold {invalid:?} at {index}"
                );
            }
        }
        for index in 1..6 {
            let mut input = adaptive_geometry_fixture_v3();
            input.threshold_ladder_bits = [0.25_f64.to_bits(); 6];
            input.threshold_ladder_bits[index] = 0.2_f64.to_bits();
            assert!(
                seal_adaptive_resident_generation_plan_v3(input, adaptive_controls_fixture_v3(),)
                    .is_err(),
                "descending threshold at {index}"
            );
        }
    }

    #[test]
    fn adaptive_generation_controls_preserve_config_and_exact_ordered_seed_bits() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<RawResidentAdaptivePolicyV3>(), 216);
        assert_eq!(align_of::<RawResidentAdaptivePolicyV3>(), 8);
        assert_eq!(
            offset_of!(RawResidentAdaptivePolicyV3, survivor_fraction),
            80
        );
        assert_eq!(
            offset_of!(RawResidentAdaptivePolicyV3, policy_identity_sha256),
            184
        );
        for (parent, expected) in [
            (ParentSelectionPolicyV1::RankWeighted, 1),
            (ParentSelectionPolicyV1::Uniform, 2),
            (ParentSelectionPolicyV1::Tournament, 4),
            (ParentSelectionPolicyV1::Softmax, 3),
        ] {
            for survivor in [
                SurvivorSelectionPolicyV1::RankWeighted,
                SurvivorSelectionPolicyV1::Elitist,
                SurvivorSelectionPolicyV1::Tournament,
                SurvivorSelectionPolicyV1::Generational,
            ] {
                let mut input = adaptive_geometry_fixture_v3();
                input.parent_selection = parent;
                input.survivor_selection = survivor;
                let plan = seal_adaptive_resident_generation_plan_v3(
                    input,
                    adaptive_controls_fixture_v3(),
                )
                .unwrap();
                let controls = plan.adaptive_controls_v3().unwrap();
                assert_eq!(plan.raw.parent_selection_policy, parent as u32);
                assert_eq!(controls.policy.parent_policy, expected);
                assert_eq!(controls.policy.survivor_policy, survivor as u32);
                assert_eq!(controls.policy.abi_version, 3);
                assert_eq!(controls.policy.algorithm_version, 1);
                assert_eq!(controls.policy.gate_start, 1.25);
                assert_eq!(controls.policy.gate_end, -0.1);
                assert_eq!(controls.template_indices, [2, 18, 0, 0]);
                assert_eq!(
                    controls
                        .template_weights
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    [-0.0_f64, -0.4, 0.0, 0.0].map(f64::to_bits)
                );
                assert_eq!(controls.template_scalars[0].short_threshold, -3.0);
                assert_eq!(controls.template_scalars[0].target_pips, 150.0);
                assert_eq!(controls.initial_seen_hashes, [0, 9, 9, u64::MAX]);
                assert_ne!(
                    plan.adaptive_policy_identity_sha256_v3().unwrap(),
                    plan.plan_identity_sha256_v1()
                );
            }
        }
        let legacy = seal_resident_generation_plan_v1(checked_geometry_fixture_v1()).unwrap();
        assert!(legacy.raw_adaptive_policy_v3().is_none());
    }

    #[test]
    fn adaptive_generation_identity_binds_policy_templates_seen_order_and_geometry() {
        let seal = |input, controls| {
            seal_adaptive_resident_generation_plan_v3(input, controls)
                .unwrap()
                .adaptive_policy_identity_sha256_v3()
                .unwrap()
        };
        let base = seal(
            adaptive_geometry_fixture_v3(),
            adaptive_controls_fixture_v3(),
        );
        for mutate in [
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.tournament_size += 1,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.seen_capacity += 1,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.seen_retry_attempts += 1,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.seed_template_count = 0,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.soft_stagnation_patience += 1,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.survivor_fraction = 0.6,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.immigrant_fraction = 0.9,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.gate_start = 1.3,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].weights[0] = 0.0,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].feature_indices[0] = 1,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].target_pips = 151.0,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.initial_seen_hashes.swap(0, 1),
        ] {
            let mut controls = adaptive_controls_fixture_v3();
            mutate(&mut controls);
            assert_ne!(base, seal(adaptive_geometry_fixture_v3(), controls));
        }
        let mut geometry = adaptive_geometry_fixture_v3();
        geometry.search_seed += 1;
        assert_ne!(base, seal(geometry, adaptive_controls_fixture_v3()));
    }

    #[test]
    fn adaptive_generation_invalid_controls_fail_before_native_admission() {
        for mutate in [
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.tournament_size = 1,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.min_structural_smc_flags = 11,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.seen_capacity = 3,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.seen_retry_attempts = 0,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.seed_template_count = 2,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.survivor_fraction = f64::NAN,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.selection_temperature = 0.0,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.gate_end = f64::INFINITY,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].feature_indices[1] = 2,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].feature_indices.swap(0, 1),
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].feature_indices[1] = 19,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].weights.clear(),
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].weights[0] = f64::NAN,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].smc_flags = 0x800,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].stop_pips = 0.0,
            |c: &mut ResidentAdaptiveGenerationInputsV3| c.templates[0].short_threshold = 3.0,
        ] {
            let mut controls = adaptive_controls_fixture_v3();
            mutate(&mut controls);
            assert!(
                seal_adaptive_resident_generation_plan_v3(adaptive_geometry_fixture_v3(), controls)
                    .is_err()
            );
        }
        assert!(
            seal_adaptive_resident_generation_plan_v3(
                checked_geometry_fixture_v1(),
                adaptive_controls_fixture_v3()
            )
            .is_err()
        );
        let mut input = adaptive_geometry_fixture_v3();
        input.stop_bounds_bits[1] = 0.1_f64.to_bits();
        assert!(
            seal_adaptive_resident_generation_plan_v3(input, adaptive_controls_fixture_v3())
                .is_err()
        );
    }

    #[test]
    fn adaptive_checkpoint_binds_actual_generation_and_bounded_control_copies() {
        let raw = RawResidentAdaptiveCheckpointV3 {
            abi_version: 3,
            algorithm_version: 1,
            run_identity: 77,
            evaluated_generation: 2,
            evaluated_generations: 3,
            evaluation_slots: 111,
            evaluated_gate_bits: 0.43_f64.to_bits(),
            stagnant_generations: 3,
            best_score_bits: f64::NEG_INFINITY.to_bits(),
            survivor_count: 7,
            immigrant_count: 20,
            rescue_count: 9,
            mutation_count: 1,
            reserved: 0,
            mutation_intensity: 1.0,
            control_copy_count: 6,
            control_copy_bytes: 552,
            initial_upload_count: 4,
            initial_upload_bytes: 168,
        };
        assert_eq!(std::mem::size_of::<RawResidentAdaptiveCheckpointV3>(), 136);
        let checkpoint = ResidentAdaptiveCheckpointV3::seal_v3(raw, 77, 3, 37).unwrap();
        assert_eq!(checkpoint.evaluated_generation(), 2);
        assert_eq!(checkpoint.evaluated_generations(), 3);
        assert_eq!(checkpoint.evaluation_slots(), 111);
        assert_eq!(checkpoint.best_score(), f64::NEG_INFINITY);
        assert_eq!(
            checkpoint.evaluated_gate().to_bits(),
            raw.evaluated_gate_bits
        );
        for mutate in [
            |r: &mut RawResidentAdaptiveCheckpointV3| r.abi_version = 1,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.algorithm_version = 3,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.run_identity += 1,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.evaluated_generation += 1,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.evaluated_generations += 1,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.evaluation_slots -= 1,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.evaluated_gate_bits = f64::NAN.to_bits(),
            |r: &mut RawResidentAdaptiveCheckpointV3| r.best_score_bits = f64::INFINITY.to_bits(),
            |r: &mut RawResidentAdaptiveCheckpointV3| r.stagnant_generations = 4,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.survivor_count = 9,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.mutation_count = 0,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.mutation_intensity = f64::NAN,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.control_copy_count = 5,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.control_copy_bytes = 553,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.initial_upload_count = 5,
            |r: &mut RawResidentAdaptiveCheckpointV3| r.reserved = 1,
        ] {
            let mut changed = raw;
            mutate(&mut changed);
            assert!(ResidentAdaptiveCheckpointV3::seal_v3(changed, 77, 3, 37).is_err());
        }
        assert!(ResidentAdaptiveCheckpointV3::seal_v3(raw, 77, 23, 37).is_err());
        assert!(ResidentAdaptiveCheckpointV3::seal_v3(raw, 77, 0, 37).is_err());
        assert!(ResidentAdaptiveCheckpointV3::seal_v3(raw, 77, 3, u64::MAX).is_err());
    }

    #[test]
    fn checked_generation_geometry_preserves_explicit_inputs_and_binds_mutations() {
        let input = checked_geometry_fixture_v1();
        let sealed = seal_resident_generation_plan_v1(input.clone()).unwrap();
        let raw = sealed.raw_plan_v1();
        assert_eq!(raw.logical_population_count, 37);
        assert_eq!(raw.retained_evaluation_capacity, 37);
        assert_eq!(raw.generation_count, 23);
        assert_eq!(raw.survivor_count, 7);
        assert_eq!(raw.immigrant_count, 11);
        assert_eq!(raw.search_seed, input.search_seed);
        assert_eq!(raw.minimum_terms_per_gene, 2);
        assert_eq!(raw.max_terms_per_gene, 4);
        assert_eq!(raw.threshold_ladder_bits, input.threshold_ladder_bits);
        assert_eq!(raw.stop_bounds_bits, input.stop_bounds_bits);
        assert_eq!(raw.smc_probability_q32, input.smc_probability_q32);
        assert_eq!(raw.mutation_intensity_q32, input.mutation_intensity_q32);
        assert_eq!(
            raw.metric_semantics_sha256,
            resident_metric_semantics_sha256_v2()
        );
        assert_eq!(raw.scoring_semantics_sha256, input.scoring_semantics_sha256);
        assert_ne!(sealed.plan_identity_sha256_v1(), [0; 32]);
        for mutate in [
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.search_seed += 1,
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.generation_count += 1,
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.survivor_count += 1,
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.immigrant_count += 1,
            |i: &mut ResidentGenerationPlanAuthorityInputV1| {
                i.stop_bounds_bits[0] = 0.3_f64.to_bits()
            },
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.smc_probability_q32[0] += 1,
            |i: &mut ResidentGenerationPlanAuthorityInputV1| {
                i.scenario_order_semantics_sha256[0] ^= 1
            },
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.scoring_semantics_sha256[0] ^= 1,
        ] {
            let mut changed = input.clone();
            mutate(&mut changed);
            let changed = seal_resident_generation_plan_v1(changed).unwrap();
            assert_ne!(
                changed.plan_identity_sha256_v1(),
                sealed.plan_identity_sha256_v1()
            );
        }
    }

    #[test]
    fn checked_generation_geometry_rejects_invalid_extents_and_unsupported_policies() {
        for mutate in [
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.generation_count = 0,
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.retained_evaluation_capacity = 38,
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.survivor_count = usize::MAX,
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.minimum_terms_per_gene = 5,
            |i: &mut ResidentGenerationPlanAuthorityInputV1| {
                i.threshold_ladder_bits[1] = i.threshold_ladder_bits[0]
            },
            |i: &mut ResidentGenerationPlanAuthorityInputV1| {
                i.stop_bounds_bits[0] = f64::NAN.to_bits()
            },
            |i: &mut ResidentGenerationPlanAuthorityInputV1| {
                i.smc_probability_q32[0] = (1_u64 << 32) + 1
            },
            |i: &mut ResidentGenerationPlanAuthorityInputV1| {
                i.mutation_intensity_q32 = (1_u64 << 32) + 1
            },
            |i: &mut ResidentGenerationPlanAuthorityInputV1| i.run_identity_sha256 = [0; 32],
            |i: &mut ResidentGenerationPlanAuthorityInputV1| {
                i.generation_semantics_sha256 = [1; 32]
            },
            |i: &mut ResidentGenerationPlanAuthorityInputV1| {
                i.parent_selection = ParentSelectionPolicyV1::Tournament
            },
            |i: &mut ResidentGenerationPlanAuthorityInputV1| {
                i.survivor_selection = SurvivorSelectionPolicyV1::Generational
            },
        ] {
            let mut input = checked_geometry_fixture_v1();
            mutate(&mut input);
            assert!(seal_resident_generation_plan_v1(input).is_err());
        }
    }

    #[test]
    fn checked_generation_geometry_requires_current_metric_protocol_and_rejection_semantics() {
        use neoethos_gpu_contracts::device::NeoPopulationMetricRow;

        assert_eq!(std::mem::size_of::<NeoPopulationMetricRow>(), 104);
        assert_eq!(
            std::mem::offset_of!(NeoPopulationMetricRow, candidate_id),
            0
        );
        assert_eq!(std::mem::offset_of!(NeoPopulationMetricRow, scenario_id), 8);
        assert_eq!(std::mem::offset_of!(NeoPopulationMetricRow, values), 16);
        let protocol_only = sha256_v1(&[RESIDENT_METRIC_ROW_PROTOCOL_V2.as_bytes()]);
        let legacy_rejection = sha256_v1(&[
            RESIDENT_METRIC_ROW_PROTOCOL_V2.as_bytes(),
            b"legacy-nonfinite-arithmetic-and-economic-rejection-indistinguishable",
        ]);
        for stale in [[4; 32], protocol_only, legacy_rejection] {
            assert_ne!(stale, resident_metric_semantics_sha256_v2());
            let mut input = checked_geometry_fixture_v1();
            input.metric_semantics_sha256 = stale;
            assert!(matches!(
                seal_resident_generation_plan_v1(input),
                Err(ResidentGenerationDeviceErrorV1::IdentityMismatch(
                    "resident metric semantics"
                ))
            ));
        }
    }

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

    #[test]
    fn fixture_policy_and_operator_surface_is_fully_exercised() {
        for (parent, expected) in [
            (
                ParentSelectionPolicyV1::Uniform,
                ResidentGenerationDeviceErrorV1::UnsupportedUniformSelection,
            ),
            (
                ParentSelectionPolicyV1::Tournament,
                ResidentGenerationDeviceErrorV1::UnsupportedTournamentSelection,
            ),
            (
                ParentSelectionPolicyV1::Softmax,
                ResidentGenerationDeviceErrorV1::UnsupportedSoftmaxSelection,
            ),
        ] {
            assert_eq!(
                format!(
                    "{:?}",
                    validate_rank_weighted_only_v1(parent, SurvivorSelectionPolicyV1::RankWeighted)
                        .expect_err("non-rank parent policy must fail")
                ),
                format!("{expected:?}")
            );
        }
        for (survivor, expected) in [
            (
                SurvivorSelectionPolicyV1::Elitist,
                ResidentGenerationDeviceErrorV1::UnsupportedElitistSelection,
            ),
            (
                SurvivorSelectionPolicyV1::Tournament,
                ResidentGenerationDeviceErrorV1::UnsupportedTournamentSelection,
            ),
            (
                SurvivorSelectionPolicyV1::Generational,
                ResidentGenerationDeviceErrorV1::UnsupportedGenerationalSelection,
            ),
        ] {
            assert_eq!(
                format!(
                    "{:?}",
                    validate_rank_weighted_only_v1(ParentSelectionPolicyV1::RankWeighted, survivor)
                        .expect_err("non-rank survivor policy must fail")
                ),
                format!("{expected:?}")
            );
        }

        let operators = [
            GeneticOperatorIdentityV1::InitializeTermCount,
            GeneticOperatorIdentityV1::InitializeIndicator,
            GeneticOperatorIdentityV1::InitializeWeightLevel,
            GeneticOperatorIdentityV1::InitializeWeightSign,
            GeneticOperatorIdentityV1::InitializeThreshold,
            GeneticOperatorIdentityV1::InitializeStopGeometry,
            GeneticOperatorIdentityV1::InitializeSmcFlag,
            GeneticOperatorIdentityV1::ParentA,
            GeneticOperatorIdentityV1::ParentB,
            GeneticOperatorIdentityV1::CrossoverScalar,
            GeneticOperatorIdentityV1::MutationKind,
            GeneticOperatorIdentityV1::MutationValue,
            GeneticOperatorIdentityV1::MutationSmc,
            GeneticOperatorIdentityV1::Survivor,
        ];
        assert_eq!(
            operators.map(|operator| operator as u32),
            std::array::from_fn(|i| i as u32 + 1)
        );

        let address = checked_philox_counter_mapping_v1(
            11,
            &[7; 32],
            2,
            3,
            GeneticOperatorIdentityV1::ParentA,
            4,
        )
        .expect("valid fixture address");
        assert_eq!(address.counter()[2], 2);
        assert_ne!(address.key(), [0; 2]);

        for error in [
            ResidentGenerationDeviceErrorV1::InvalidPlan("plan"),
            ResidentGenerationDeviceErrorV1::IdentityMismatch("identity"),
        ] {
            match error {
                ResidentGenerationDeviceErrorV1::InvalidPlan(message)
                | ResidentGenerationDeviceErrorV1::IdentityMismatch(message) => {
                    assert!(!message.is_empty());
                }
                _ => unreachable!(),
            }
        }
    }
}
