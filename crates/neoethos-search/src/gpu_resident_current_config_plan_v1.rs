//! Exact, host-metadata-only admission plan for the configured resident Search.
//!
//! This module does not execute a kernel and does not materialize feature,
//! metric, or gene values on the host. It freezes the current headless config,
//! trim geometry, immutable base-scenario extent, profitable-archive policy,
//! exact novelty workload, and already-sealed CUDA admission identities into a
//! single run identity. Native trim, archive, and generation owners may consume
//! this receipt by value in later slices; callers cannot supply raw handles.

use sha2::{Digest, Sha256};
use std::fmt;
use std::ops::Range;

use crate::DiscoveryConfig;
use crate::canonical_discovery_config_digest_v1::canonical_discovery_config_digest_v1;
use crate::discovery::{DiscoveryMode, resolve_prefilter_top_k};
use crate::genetic::{
    GeneticSearchRuntimeOverrides, ParentSelectionPolicy, SurvivorSelectionPolicy,
};

pub const CURRENT_CONFIG_RESIDENT_SEARCH_PLAN_SEMANTICS_V1: &str =
    "neoethos.current-config-resident-search-plan.v1";
pub const CURRENT_CONFIG_TRIM_SEMANTICS_V1: &str =
    "prefix-parent-exact-80-20-prefilter-fit-holdout-v1";
pub const CURRENT_CONFIG_ARCHIVE_GENE_IDENTITY_SEMANTICS_V1: &str = "full-gene-first-seen-v1";
pub const CURRENT_CONFIG_ARCHIVE_ADMISSION_SEMANTICS_V1: &str = "net-strictly-greater-than-v1";
pub const CURRENT_CONFIG_ARCHIVE_NEIGHBOR_SEMANTICS_V1: &str =
    "population-plus-permanent-hybrid-archive-knn-jaccard-v1";

const CURRENT_CONFIG_PREFILTER_INSAMPLE_BITS_V1: u64 = 0.8_f64.to_bits();
const CURRENT_CONFIG_EVALUATION_SYMBOL_V1: &str = "EURUSD";
const CURRENT_CONFIG_EVALUATION_ACCOUNT_CURRENCY_V1: &str = "GBP";
const MINIMUM_PREFILTER_FIT_ROWS_V1: usize = 100;
const BITS_PER_SIGNATURE_WORD_V1: usize = u64::BITS as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrentConfigResidentSearchPlanErrorV1 {
    InvalidAdmissionFacts,
    InvalidInputShape,
    InsufficientRows,
    InvalidNoveltyWeight,
    InvalidNoveltyNeighbors,
    UnsupportedCurrentConfigSemantics,
    ArithmeticOverflow,
    ArchiveKnnBudgetExceeded,
    CanonicalConfigEncodingFailure,
}

impl fmt::Display for CurrentConfigResidentSearchPlanErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidAdmissionFacts => {
                "current-config resident Search admission facts are incomplete"
            }
            Self::InvalidInputShape => "current-config resident Search input shape is invalid",
            Self::InsufficientRows => {
                "current-config resident Search requires at least 100 fit rows and one holdout row"
            }
            Self::InvalidNoveltyWeight => {
                "current-config resident Search novelty weight is invalid"
            }
            Self::InvalidNoveltyNeighbors => {
                "current-config resident Search novelty-neighbor count is invalid"
            }
            Self::UnsupportedCurrentConfigSemantics => {
                "current-config resident Search slice does not implement the requested semantics"
            }
            Self::ArithmeticOverflow => {
                "current-config resident Search admission arithmetic overflowed"
            }
            Self::ArchiveKnnBudgetExceeded => {
                "current-config resident Search exact archive-kNN work exceeds the measured one-hour device budget"
            }
            Self::CanonicalConfigEncodingFailure => {
                "current-config resident Search canonical configuration encoding failed"
            }
        })
    }
}

impl std::error::Error for CurrentConfigResidentSearchPlanErrorV1 {}

/// Bounded facts copied from already-sealed native receipts. The hashes bind
/// the actual run/device/build/workspace authorities; this value intentionally
/// contains no raw CUDA context, stream, event, pool, or allocation handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentConfigResidentSearchAdmissionFactsV1 {
    selected_device_ordinal: u32,
    cuda_build_identity_sha256: [u8; 32],
    runtime_device_identity_sha256: [u8; 32],
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    memory_pool_identity_sha256: [u8; 32],
    canonical_input_receipt_identity_sha256: [u8; 32],
    full_workspace_plan_identity_sha256: [u8; 32],
    archive_knn_calibration_identity_sha256: [u8; 32],
    measured_archive_knn_popcount_words_per_second: u64,
    phase_one_free_bytes_snapshot: u64,
    allocator_context_reserve_bytes: u64,
    required_workspace_bytes: u64,
    trim_prefilter_reserved_bytes: u64,
    full_discovery_reserve_bytes: u64,
}

impl CurrentConfigResidentSearchAdmissionFactsV1 {
    #[cfg(test)]
    const fn test_fixture_v1(
        selected_device_ordinal: u32,
        cuda_build_identity_sha256: [u8; 32],
        runtime_device_identity_sha256: [u8; 32],
        primary_context_identity_sha256: [u8; 32],
        run_stream_identity_sha256: [u8; 32],
        memory_pool_identity_sha256: [u8; 32],
        canonical_input_receipt_identity_sha256: [u8; 32],
        full_workspace_plan_identity_sha256: [u8; 32],
        archive_knn_calibration_identity_sha256: [u8; 32],
        measured_archive_knn_popcount_words_per_second: u64,
        phase_one_free_bytes_snapshot: u64,
        allocator_context_reserve_bytes: u64,
        required_workspace_bytes: u64,
        trim_prefilter_reserved_bytes: u64,
        full_discovery_reserve_bytes: u64,
    ) -> Self {
        Self {
            selected_device_ordinal,
            cuda_build_identity_sha256,
            runtime_device_identity_sha256,
            primary_context_identity_sha256,
            run_stream_identity_sha256,
            memory_pool_identity_sha256,
            canonical_input_receipt_identity_sha256,
            full_workspace_plan_identity_sha256,
            archive_knn_calibration_identity_sha256,
            measured_archive_knn_popcount_words_per_second,
            phase_one_free_bytes_snapshot,
            allocator_context_reserve_bytes,
            required_workspace_bytes,
            trim_prefilter_reserved_bytes,
            full_discovery_reserve_bytes,
        }
    }

    fn validate(self) -> Result<Self, CurrentConfigResidentSearchPlanErrorV1> {
        if self.cuda_build_identity_sha256 == [0; 32]
            || self.runtime_device_identity_sha256 == [0; 32]
            || self.primary_context_identity_sha256 == [0; 32]
            || self.run_stream_identity_sha256 == [0; 32]
            || self.memory_pool_identity_sha256 == [0; 32]
            || self.canonical_input_receipt_identity_sha256 == [0; 32]
            || self.full_workspace_plan_identity_sha256 == [0; 32]
            || self.archive_knn_calibration_identity_sha256 == [0; 32]
            || self.measured_archive_knn_popcount_words_per_second == 0
            || self.phase_one_free_bytes_snapshot == 0
            || self.allocator_context_reserve_bytes == 0
            || self.required_workspace_bytes == 0
            || self.trim_prefilter_reserved_bytes == 0
            || self.full_discovery_reserve_bytes == 0
            || self.trim_prefilter_reserved_bytes > self.full_discovery_reserve_bytes
            || self.required_workspace_bytes > self.full_discovery_reserve_bytes
            || self
                .allocator_context_reserve_bytes
                .checked_add(self.full_discovery_reserve_bytes)
                .is_none_or(|required| required > self.phase_one_free_bytes_snapshot)
        {
            return Err(CurrentConfigResidentSearchPlanErrorV1::InvalidAdmissionFacts);
        }
        Ok(self)
    }
}

/// Immutable metadata receipt consumed by the resident trim/admission bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedCurrentConfigResidentSearchPlanV1 {
    selected_device_ordinal: u32,
    parent_row_range: Range<usize>,
    prefilter_fit_row_range: Range<usize>,
    outer_holdout_row_range: Range<usize>,
    parent_column_count: usize,
    prefilter_top_k: usize,
    prefilter_min_per_timeframe: usize,
    population: usize,
    maximum_generations: usize,
    maximum_runtime_millis: u64,
    maximum_terms_per_gene: usize,
    immutable_base_scenario_count: usize,
    novelty_weight_bits: u64,
    novelty_neighbors: usize,
    permanent_archive_capacity: usize,
    archive_min_net_bits: u64,
    gene_signature_word_count: usize,
    maximum_archive_knn_distance_count: u64,
    maximum_archive_knn_popcount_word_count: u64,
    required_archive_knn_popcount_words_per_second: u64,
    measured_archive_knn_popcount_words_per_second: u64,
    trim_prefilter_reserved_bytes: u64,
    required_workspace_bytes: u64,
    full_discovery_reserve_bytes: u64,
    canonical_discovery_config_digest_sha256: [u8; 32],
    plan_identity_sha256: [u8; 32],
}

impl SealedCurrentConfigResidentSearchPlanV1 {
    pub const fn selected_device_ordinal(&self) -> u32 {
        self.selected_device_ordinal
    }

    pub fn parent_row_range(&self) -> Range<usize> {
        self.parent_row_range.clone()
    }

    pub fn prefilter_fit_row_range(&self) -> Range<usize> {
        self.prefilter_fit_row_range.clone()
    }

    pub fn outer_holdout_row_range(&self) -> Range<usize> {
        self.outer_holdout_row_range.clone()
    }

    pub const fn parent_column_count(&self) -> usize {
        self.parent_column_count
    }

    pub const fn prefilter_top_k(&self) -> usize {
        self.prefilter_top_k
    }

    pub const fn prefilter_min_per_timeframe(&self) -> usize {
        self.prefilter_min_per_timeframe
    }

    pub const fn population(&self) -> usize {
        self.population
    }

    pub const fn maximum_generations(&self) -> usize {
        self.maximum_generations
    }

    pub const fn maximum_runtime_millis(&self) -> u64 {
        self.maximum_runtime_millis
    }

    pub const fn maximum_terms_per_gene(&self) -> usize {
        self.maximum_terms_per_gene
    }

    pub const fn immutable_base_scenario_count(&self) -> usize {
        self.immutable_base_scenario_count
    }

    pub const fn novelty_weight(&self) -> f64 {
        f64::from_bits(self.novelty_weight_bits)
    }

    pub const fn novelty_neighbors(&self) -> usize {
        self.novelty_neighbors
    }

    pub const fn permanent_archive_capacity(&self) -> usize {
        self.permanent_archive_capacity
    }

    pub const fn archive_min_net(&self) -> f64 {
        f64::from_bits(self.archive_min_net_bits)
    }

    pub const fn archive_gene_identity_semantics(&self) -> &'static str {
        CURRENT_CONFIG_ARCHIVE_GENE_IDENTITY_SEMANTICS_V1
    }

    pub const fn archive_admission_semantics(&self) -> &'static str {
        CURRENT_CONFIG_ARCHIVE_ADMISSION_SEMANTICS_V1
    }

    pub const fn archive_neighbor_semantics(&self) -> &'static str {
        CURRENT_CONFIG_ARCHIVE_NEIGHBOR_SEMANTICS_V1
    }

    pub const fn gene_signature_word_count(&self) -> usize {
        self.gene_signature_word_count
    }

    pub const fn maximum_archive_knn_distance_count(&self) -> u64 {
        self.maximum_archive_knn_distance_count
    }

    pub const fn maximum_archive_knn_popcount_word_count(&self) -> u64 {
        self.maximum_archive_knn_popcount_word_count
    }

    pub const fn required_archive_knn_popcount_words_per_second(&self) -> u64 {
        self.required_archive_knn_popcount_words_per_second
    }

    pub const fn measured_archive_knn_popcount_words_per_second(&self) -> u64 {
        self.measured_archive_knn_popcount_words_per_second
    }

    pub const fn archive_knn_budget_admitted(&self) -> bool {
        self.measured_archive_knn_popcount_words_per_second
            >= self.required_archive_knn_popcount_words_per_second
    }

    pub const fn trim_prefilter_reserved_bytes(&self) -> u64 {
        self.trim_prefilter_reserved_bytes
    }

    pub const fn required_workspace_bytes(&self) -> u64 {
        self.required_workspace_bytes
    }

    pub const fn full_discovery_reserve_bytes(&self) -> u64 {
        self.full_discovery_reserve_bytes
    }

    pub const fn canonical_discovery_config_digest_sha256(&self) -> [u8; 32] {
        self.canonical_discovery_config_digest_sha256
    }

    pub const fn plan_identity_sha256(&self) -> [u8; 32] {
        self.plan_identity_sha256
    }
}

/// Declaration-only marker for the separate full resident-Discovery deadline
/// proof. Slice 2 cannot mint this type and does not use it for readiness.
pub struct FullResidentDiscoveryDeadlineReceiptV1 {
    _not_minted_in_slice2: core::convert::Infallible,
}

pub(crate) const CURRENT_CONFIG_RESIDENT_SEARCH_SLICE2_PLAN_SEMANTICS_V2: &str =
    "neoethos.current-config-resident-search-slice2-plan.v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CurrentConfigResidentSearchSlice2PlanFactsV2 {
    population: u64,
    maximum_generations: u64,
    maximum_runtime_millis: u64,
    maximum_terms_per_gene: u64,
    gene_signature_word_count: u64,
    novelty_weight_bits: u64,
    novelty_neighbors: u64,
    permanent_archive_capacity: u64,
    calibration_active_count: u64,
    maximum_jaccard_union: u32,
    maximum_jaccard_cross_product: u64,
    maximum_archive_knn_distance_count: u64,
    maximum_archive_knn_popcount_word_count: u64,
    required_archive_knn_distance_items_per_second: u64,
    required_archive_knn_popcount_words_per_second: u64,
    layout_alignment_bytes: u64,
    archive_gene_scalars_bytes: u64,
    archive_term_indices_bytes: u64,
    archive_term_weights_bytes: u64,
    archive_metric_rows_bytes: u64,
    archive_signatures_bytes: u64,
    archive_hashes_bytes: u64,
    current_population_signatures_bytes: u64,
    novelty_scores_bytes: u64,
    exact_top_k_keys_bytes: u64,
    admission_flags_bytes: u64,
    admission_offsets_bytes: u64,
    archive_control_and_seal_bytes: u64,
    control_subtotal_bytes: u64,
    slice2_replacement_subtotal_bytes: u64,
    replaced_v1_scoring_bytes: u64,
    slice2_net_additional_bytes: u64,
    current_source_kind_wire: u8,
    archive_source_kind_wire: u8,
    current_ordinal_exclusive_end: u64,
    archive_ordinal_exclusive_end: u64,
    binary64_operation_sequence_wire: u8,
    binary64_math_mode_wire: u8,
    binary64_tolerance_policy_wire: u8,
    binary64_absolute_tolerance_bits: u64,
    binary64_relative_tolerance_bits: u64,
    binary64_max_ulp_distance: u64,
    novelty_semantics_identity_sha256: [u8; 32],
    archive_capacity_identity_sha256: [u8; 32],
    calibration_active_count_identity_sha256: [u8; 32],
    layout_identity_sha256: [u8; 32],
    calibration_identity_sha256: [u8; 32],
    source_kind_encoding_identity_sha256: [u8; 32],
    current_ordinal_domain_identity_sha256: [u8; 32],
    archive_ordinal_domain_identity_sha256: [u8; 32],
    tie_order_identity_sha256: [u8; 32],
    binary64_operation_sequence_identity_sha256: [u8; 32],
    binary64_math_mode_identity_sha256: [u8; 32],
    binary64_tolerance_identity_sha256: [u8; 32],
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CurrentConfigResidentSearchSlice2PlanIdentityReceiptV2 {
    _identity_sha256: [u8; 32],
}

impl CurrentConfigResidentSearchSlice2PlanIdentityReceiptV2 {
    pub(crate) const fn identity_sha256(&self) -> [u8; 32] {
        self._identity_sha256
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CurrentConfigResidentSearchSlice2PlanErrorV2 {
    ImplementationPending,
    IdentityReceiptMismatch,
}

mod slice2_plan_seal_v2 {
    pub(super) struct Marker {
        _private: (),
    }
}

pub(crate) struct SealedCurrentConfigResidentSearchSlice2PlanV2 {
    base: SealedCurrentConfigResidentSearchPlanV1,
    facts: CurrentConfigResidentSearchSlice2PlanFactsV2,
    identity: CurrentConfigResidentSearchSlice2PlanIdentityReceiptV2,
    _seal: slice2_plan_seal_v2::Marker,
}

impl SealedCurrentConfigResidentSearchSlice2PlanV2 {
    pub(crate) fn facts_v2(&self) -> &CurrentConfigResidentSearchSlice2PlanFactsV2 {
        let _ = &self.base;
        &self.facts
    }

    pub(crate) fn identity_receipt_v2(
        &self,
    ) -> &CurrentConfigResidentSearchSlice2PlanIdentityReceiptV2 {
        &self.identity
    }

    pub(crate) fn validate_identity_receipt_v2(
        &self,
        _receipt: &CurrentConfigResidentSearchSlice2PlanIdentityReceiptV2,
    ) -> Result<(), CurrentConfigResidentSearchSlice2PlanErrorV2> {
        Err(CurrentConfigResidentSearchSlice2PlanErrorV2::ImplementationPending)
    }
}

pub(crate) fn seal_current_config_resident_search_slice2_plan_v2(
    base_v1: SealedCurrentConfigResidentSearchPlanV1,
    facts_v2: CurrentConfigResidentSearchSlice2PlanFactsV2,
) -> Result<
    SealedCurrentConfigResidentSearchSlice2PlanV2,
    CurrentConfigResidentSearchSlice2PlanErrorV2,
> {
    let _ = (base_v1, facts_v2);
    Err(CurrentConfigResidentSearchSlice2PlanErrorV2::ImplementationPending)
}

/// Seal the current headless config into one immutable resident trim/Search
/// receipt. This is intentionally narrower than generic Discovery: semantics
/// not exercised by the shipped configuration fail before native allocation.
pub fn seal_current_config_resident_search_plan_v1(
    config: &DiscoveryConfig,
    runtime: &GeneticSearchRuntimeOverrides,
    parent_rows: usize,
    parent_columns: usize,
    admission: CurrentConfigResidentSearchAdmissionFactsV1,
) -> Result<SealedCurrentConfigResidentSearchPlanV1, CurrentConfigResidentSearchPlanErrorV1> {
    let admission = admission.validate()?;
    let canonical_discovery_config_digest_sha256 = canonical_discovery_config_digest_v1(config)
        .map_err(|_| CurrentConfigResidentSearchPlanErrorV1::CanonicalConfigEncodingFailure)?;
    if parent_rows == 0 || parent_columns == 0 || config.population < 2 {
        return Err(CurrentConfigResidentSearchPlanErrorV1::InvalidInputShape);
    }
    if config.population_auto
        || config.max_rows != 0
        || config.max_rows_by_timeframe.values().any(|cap| *cap != 0)
        || config.runtime_overrides.prefilter_insample_frac.to_bits()
            != CURRENT_CONFIG_PREFILTER_INSAMPLE_BITS_V1
        || config.runtime_overrides.prefilter_top_k == 0
        || runtime.archive_scoring.mode != "net"
        // The resident prefilter has no host price hint. For the shipped
        // EURUSD/GBP cross-account configuration, both canonical pip-value
        // branches ignore the EURUSD spot and use USD->GBP conversion. A
        // different symbol/account pair may make the last close numerical
        // input, so this V1 refuses it rather than substituting None.
        || config.evaluation_symbol != CURRENT_CONFIG_EVALUATION_SYMBOL_V1
        || config.evaluation_account_currency
            != CURRENT_CONFIG_EVALUATION_ACCOUNT_CURRENCY_V1
    {
        return Err(CurrentConfigResidentSearchPlanErrorV1::UnsupportedCurrentConfigSemantics);
    }
    if !config.max_hours.is_finite()
        || config.max_hours <= 0.0
        || config.generations == 0
        || config.max_indicators == 0
    {
        return Err(CurrentConfigResidentSearchPlanErrorV1::UnsupportedCurrentConfigSemantics);
    }
    if !runtime.novelty_weight.is_finite()
        || runtime.novelty_weight.is_sign_negative()
        || runtime.novelty_weight > 1.0
    {
        return Err(CurrentConfigResidentSearchPlanErrorV1::InvalidNoveltyWeight);
    }
    if runtime.novelty_neighbors == 0 || runtime.novelty_neighbors >= config.population {
        return Err(CurrentConfigResidentSearchPlanErrorV1::InvalidNoveltyNeighbors);
    }
    if !runtime.archive_scoring.min_net.is_finite() {
        return Err(CurrentConfigResidentSearchPlanErrorV1::UnsupportedCurrentConfigSemantics);
    }

    let prefilter_fit_rows = parent_rows
        .checked_mul(4)
        .ok_or(CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow)?
        / 5;
    if prefilter_fit_rows < MINIMUM_PREFILTER_FIT_ROWS_V1 || prefilter_fit_rows >= parent_rows {
        return Err(CurrentConfigResidentSearchPlanErrorV1::InsufficientRows);
    }
    let prefilter_top_k = resolve_prefilter_top_k(
        config.runtime_overrides.prefilter_top_k,
        parent_columns,
        config.population,
        config.max_indicators,
    );
    if prefilter_top_k == 0 || config.max_indicators > prefilter_top_k {
        return Err(CurrentConfigResidentSearchPlanErrorV1::UnsupportedCurrentConfigSemantics);
    }
    let maximum_runtime_millis_f64 = config.max_hours * 3_600_000.0;
    if !maximum_runtime_millis_f64.is_finite()
        || maximum_runtime_millis_f64 < 1.0
        || maximum_runtime_millis_f64 > u64::MAX as f64
        || maximum_runtime_millis_f64.fract() != 0.0
    {
        return Err(CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow);
    }
    let maximum_runtime_millis = maximum_runtime_millis_f64 as u64;
    let permanent_archive_capacity = runtime
        .checked_effective_archive_cap(config.population, config.generations)
        .ok_or(CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow)?;
    let neighbors_per_candidate = permanent_archive_capacity
        .checked_add(config.population - 1)
        .ok_or(CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow)?;
    let maximum_archive_knn_distance_count = u64::try_from(config.population)
        .ok()
        .and_then(|population| population.checked_mul(u64::try_from(neighbors_per_candidate).ok()?))
        .and_then(|per_generation| {
            per_generation.checked_mul(u64::try_from(config.generations).ok()?)
        })
        .ok_or(CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow)?;
    let gene_signature_word_count = prefilter_top_k
        .checked_add(BITS_PER_SIGNATURE_WORD_V1 - 1)
        .ok_or(CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow)?
        / BITS_PER_SIGNATURE_WORD_V1;
    let maximum_archive_knn_popcount_word_count = maximum_archive_knn_distance_count
        .checked_mul(
            u64::try_from(gene_signature_word_count)
                .map_err(|_| CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow)?,
        )
        .ok_or(CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow)?;
    let required_archive_knn_popcount_words_per_second = maximum_archive_knn_popcount_word_count
        .checked_mul(1_000)
        .and_then(|scaled| scaled.checked_add(maximum_runtime_millis - 1))
        .map(|rounded| rounded / maximum_runtime_millis)
        .ok_or(CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow)?;
    if admission.measured_archive_knn_popcount_words_per_second
        < required_archive_knn_popcount_words_per_second
    {
        return Err(CurrentConfigResidentSearchPlanErrorV1::ArchiveKnnBudgetExceeded);
    }

    let parent_row_range = 0..parent_rows;
    let prefilter_fit_row_range = 0..prefilter_fit_rows;
    let outer_holdout_row_range = prefilter_fit_rows..parent_rows;
    let identity_extents = [
        parent_rows,
        parent_columns,
        prefilter_fit_rows,
        prefilter_top_k,
        config.runtime_overrides.prefilter_min_per_timeframe,
        config.population,
        config.generations,
        config.max_indicators,
        runtime.novelty_neighbors,
        permanent_archive_capacity,
        gene_signature_word_count,
        runtime.stagnation_patience,
        runtime.convergence_patience,
    ]
    .map(|value| {
        u64::try_from(value).map_err(|_| CurrentConfigResidentSearchPlanErrorV1::ArithmeticOverflow)
    });
    let identity_extents = identity_extents
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let plan_identity_sha256 = hash_plan_v1(
        config,
        runtime,
        admission,
        maximum_archive_knn_distance_count,
        maximum_archive_knn_popcount_word_count,
        required_archive_knn_popcount_words_per_second,
        maximum_runtime_millis,
        &identity_extents,
        canonical_discovery_config_digest_sha256,
    );

    Ok(SealedCurrentConfigResidentSearchPlanV1 {
        selected_device_ordinal: admission.selected_device_ordinal,
        parent_row_range,
        prefilter_fit_row_range,
        outer_holdout_row_range,
        parent_column_count: parent_columns,
        prefilter_top_k,
        prefilter_min_per_timeframe: config.runtime_overrides.prefilter_min_per_timeframe,
        population: config.population,
        maximum_generations: config.generations,
        maximum_runtime_millis,
        maximum_terms_per_gene: config.max_indicators,
        immutable_base_scenario_count: config.population,
        novelty_weight_bits: runtime.novelty_weight.to_bits(),
        novelty_neighbors: runtime.novelty_neighbors,
        permanent_archive_capacity,
        archive_min_net_bits: runtime.archive_scoring.min_net.to_bits(),
        gene_signature_word_count,
        maximum_archive_knn_distance_count,
        maximum_archive_knn_popcount_word_count,
        required_archive_knn_popcount_words_per_second,
        measured_archive_knn_popcount_words_per_second: admission
            .measured_archive_knn_popcount_words_per_second,
        trim_prefilter_reserved_bytes: admission.trim_prefilter_reserved_bytes,
        required_workspace_bytes: admission.required_workspace_bytes,
        full_discovery_reserve_bytes: admission.full_discovery_reserve_bytes,
        canonical_discovery_config_digest_sha256,
        plan_identity_sha256,
    })
}

fn hash_plan_v1(
    config: &DiscoveryConfig,
    runtime: &GeneticSearchRuntimeOverrides,
    admission: CurrentConfigResidentSearchAdmissionFactsV1,
    maximum_archive_knn_distance_count: u64,
    maximum_archive_knn_popcount_word_count: u64,
    required_archive_knn_popcount_words_per_second: u64,
    maximum_runtime_millis: u64,
    identity_extents: &[u64],
    canonical_discovery_config_digest_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(CURRENT_CONFIG_RESIDENT_SEARCH_PLAN_SEMANTICS_V1.as_bytes());
    hash.update(CURRENT_CONFIG_TRIM_SEMANTICS_V1.as_bytes());
    hash.update(CURRENT_CONFIG_ARCHIVE_GENE_IDENTITY_SEMANTICS_V1.as_bytes());
    hash.update(CURRENT_CONFIG_ARCHIVE_ADMISSION_SEMANTICS_V1.as_bytes());
    hash.update(CURRENT_CONFIG_ARCHIVE_NEIGHBOR_SEMANTICS_V1.as_bytes());
    hash.update(canonical_discovery_config_digest_sha256);
    hash.update(admission.selected_device_ordinal.to_le_bytes());
    hash.update(admission.cuda_build_identity_sha256);
    hash.update(admission.runtime_device_identity_sha256);
    hash.update(admission.primary_context_identity_sha256);
    hash.update(admission.run_stream_identity_sha256);
    hash.update(admission.memory_pool_identity_sha256);
    hash.update(admission.canonical_input_receipt_identity_sha256);
    hash.update(admission.full_workspace_plan_identity_sha256);
    hash.update(admission.archive_knn_calibration_identity_sha256);
    hash.update(
        admission
            .measured_archive_knn_popcount_words_per_second
            .to_le_bytes(),
    );
    hash.update(admission.phase_one_free_bytes_snapshot.to_le_bytes());
    hash.update(admission.allocator_context_reserve_bytes.to_le_bytes());
    hash.update(admission.required_workspace_bytes.to_le_bytes());
    hash.update(admission.trim_prefilter_reserved_bytes.to_le_bytes());
    hash.update(admission.full_discovery_reserve_bytes.to_le_bytes());
    debug_assert_eq!(identity_extents.len(), 13);
    for value in identity_extents {
        hash.update(value.to_le_bytes());
    }
    hash.update(maximum_runtime_millis.to_le_bytes());
    hash.update(runtime.novelty_weight.to_bits().to_le_bytes());
    hash.update(runtime.archive_scoring.min_net.to_bits().to_le_bytes());
    hash.update(maximum_archive_knn_distance_count.to_le_bytes());
    hash.update(maximum_archive_knn_popcount_word_count.to_le_bytes());
    hash.update(required_archive_knn_popcount_words_per_second.to_le_bytes());
    hash.update([discovery_mode_wire_v1(config.mode)]);
    hash.update(
        u64::try_from(config.candidate_count)
            .expect("identity extents validated as u64")
            .to_le_bytes(),
    );
    hash.update(
        u64::try_from(config.portfolio_size)
            .expect("identity extents validated as u64")
            .to_le_bytes(),
    );
    hash.update(
        u64::try_from(config.walkforward_splits)
            .expect("identity extents validated as u64")
            .to_le_bytes(),
    );
    hash.update([u8::from(config.enable_cpcv)]);
    for value in [
        config.cpcv_n_splits,
        config.cpcv_n_test_groups,
        config.cpcv_max_rows,
        config.embargo_minutes,
    ] {
        hash.update(
            u64::try_from(value)
                .expect("identity extents validated as u64")
                .to_le_bytes(),
        );
    }
    for value in [
        config.cpcv_embargo_pct,
        config.cpcv_purge_pct,
        config.cpcv_min_phi,
        config.max_pbo,
        config.corr_threshold,
        config.evaluation_spread_pips,
        config.evaluation_commission_per_trade,
        config.swap_long_pips_per_day,
        config.swap_short_pips_per_day,
        config.initial_balance,
        config.risk_per_trade_min,
        config.risk_per_trade_max,
        config.max_regime_loss_pct,
    ] {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash_string_v1(&mut hash, &config.timeframe_label);
    hash_string_v1(&mut hash, &config.evaluation_symbol);
    hash_string_v1(&mut hash, &config.evaluation_account_currency);
    hash.update([u8::from(config.kill_zones_enabled)]);
    hash.update([u8::from(runtime.seed.is_some())]);
    hash.update(runtime.seed.unwrap_or_default().to_le_bytes());
    let smc = runtime.resolved_smc_gate();
    hash.update(smc.start.to_bits().to_le_bytes());
    hash.update(smc.end.to_bits().to_le_bytes());
    hash.update(smc.curve.to_bits().to_le_bytes());
    hash.update(smc.stagnation_step.to_bits().to_le_bytes());
    hash.update([u8::from(smc.disable_gate)]);
    let selection = runtime.resolved_selection();
    hash.update([parent_selection_wire_v1(selection.parent)]);
    hash.update([survivor_selection_wire_v1(selection.survivor)]);
    hash.update(selection.immigrant_ratio.to_bits().to_le_bytes());
    hash.update(selection.survivor_fraction.to_bits().to_le_bytes());
    hash.update(selection.temperature.to_bits().to_le_bytes());
    hash.update(
        u64::try_from(runtime.effective_tournament_size(config.population))
            .expect("population was validated as u64")
            .to_le_bytes(),
    );
    hash.update(
        u64::try_from(runtime.effective_seen_retry_attempts())
            .expect("retry count was validated as u64")
            .to_le_bytes(),
    );
    hash.update(runtime.effective_min_improvement().to_bits().to_le_bytes());
    hash.update(
        runtime
            .convergence_min_elapsed_fraction
            .to_bits()
            .to_le_bytes(),
    );
    hash.finalize().into()
}

fn hash_string_v1(hash: &mut Sha256, value: &str) {
    hash.update(
        u64::try_from(value.len())
            .expect("resident identity string length fits u64")
            .to_le_bytes(),
    );
    hash.update(value.as_bytes());
}

const fn discovery_mode_wire_v1(mode: DiscoveryMode) -> u8 {
    match mode {
        DiscoveryMode::Strict => 1,
        DiscoveryMode::PropFirm => 2,
        DiscoveryMode::Risky => 3,
    }
}

const fn parent_selection_wire_v1(policy: ParentSelectionPolicy) -> u8 {
    match policy {
        ParentSelectionPolicy::Uniform => 1,
        ParentSelectionPolicy::RankWeighted => 2,
        ParentSelectionPolicy::Softmax => 3,
        ParentSelectionPolicy::Tournament => 4,
    }
}

const fn survivor_selection_wire_v1(policy: SurvivorSelectionPolicy) -> u8 {
    match policy {
        SurvivorSelectionPolicy::Elitist => 1,
        SurvivorSelectionPolicy::RankWeighted => 2,
        SurvivorSelectionPolicy::Tournament => 3,
        SurvivorSelectionPolicy::Generational => 4,
    }
}

#[cfg(test)]
#[path = "gpu_resident_current_config_plan_v1_tests.rs"]
mod tests;
