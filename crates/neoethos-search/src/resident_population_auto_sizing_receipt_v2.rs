//! Admission-bound population sizing for the staged resident Data+population
//! CUDA route.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;

use neoethos_data::{
    CanonicalPinnedSourceBindingFactsV1, CanonicalPinnedSourceProjectionV1,
    CanonicalPinnedSourceSegmentFactsV1, GpuOnlyFeatureMaterializationErrorV3,
    PreparedGpuOnlyFeatureMaterializationV3,
};
use neoethos_gpu_cuda::{
    DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1, DATA_POPULATION_ALLOCATOR_RESERVE_POLICY_V1,
    DataPopulationWorkspacePlanErrorCodeV1, DataPopulationWorkspacePlanErrorV1,
    PopulationEvaluationViewV1, PopulationGeneStorePlanV1, PopulationMetricsOnlyPlanV1,
    PopulationTimestampModeV1, RESIDENT_ADAPTIVE_BASE_SEMANTIC_V1, ResidentAdaptiveBaseRequestV1,
    ResidentAdaptiveBaseViewTokenV1, SealedDataPopulationExecutionLimitsV1,
    SealedDataPopulationGpuWorkspacePlanV1, SealedNativeCudaDataPopulationPreflightFactsV1,
};

pub const RESIDENT_POPULATION_AUTO_SIZING_RECEIPT_SCHEMA_VERSION_V2: u16 = 2;
const RESIDENT_POPULATION_AUTO_SIZING_RECEIPT_HASH_DOMAIN_V2: &[u8] =
    b"neoethos.search.resident-population-auto-sizing-receipt.v2\0";
pub(crate) const RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2: usize = 16_384;

fn checked_effective_hard_growth_cap_v2(
    external_hard_population_cap: usize,
) -> Result<usize, ResidentPopulationAutoSizingErrorV2> {
    if external_hard_population_cap == 0 {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::InvalidInput,
            "external resident population hard cap must be non-zero",
        ));
    }
    Ok(external_hard_population_cap.min(RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2))
}

fn checked_configured_population_against_external_hard_cap_v2(
    configured_population: usize,
    external_hard_population_cap: usize,
) -> Result<(), ResidentPopulationAutoSizingErrorV2> {
    checked_effective_hard_growth_cap_v2(external_hard_population_cap)?;
    if configured_population > external_hard_population_cap {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::InvalidInput,
            format!(
                "configured resident population {configured_population} exceeds external hard cap {external_hard_population_cap}"
            ),
        ));
    }
    Ok(())
}
const RESIDENT_SELECTION_STAGE1_ROLE_V2: &str = "selection_stage1";
type InternalWorkspaceBytesV2 = (
    PopulationGeneStorePlanV1,
    PopulationMetricsOnlyPlanV1,
    u64,
    u64,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentPopulationAutoSizingErrorCodeV2 {
    InvalidInput,
    ArithmeticOverflow,
    MigrationNotSealed,
    AdaptiveBaseNotResident,
    AdaptiveTailCapExceeded,
    ExactFinancialAuthorityUnavailable,
    ConfiguredGeneNoRoom,
    ScenarioNoRoom,
    WorkspacePlan,
    AuthorityMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentPopulationAutoSizingErrorV2 {
    code: ResidentPopulationAutoSizingErrorCodeV2,
    detail: String,
}

impl ResidentPopulationAutoSizingErrorV2 {
    fn new(code: ResidentPopulationAutoSizingErrorCodeV2, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> ResidentPopulationAutoSizingErrorCodeV2 {
        self.code
    }
}

impl fmt::Display for ResidentPopulationAutoSizingErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "resident population-auto sizing failed ({:?}): {}",
            self.code, self.detail
        )
    }
}

impl std::error::Error for ResidentPopulationAutoSizingErrorV2 {}

const ADAPTIVE_RESOLUTION_DISABLED_V2: &str = "adaptive_disabled";
const ADAPTIVE_RESOLUTION_FIXED_TOO_SHORT_V2: &str = "fixed_too_short";
const ADAPTIVE_RESOLUTION_RESIDENT_EXACT_V1: &str = "resident_exact_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdaptiveStageResolutionV2 {
    requested: bool,
    effective: bool,
    max_adaptive_row_count: usize,
    reason: &'static str,
}

fn resolve_adaptive_stage_extent_v2(
    requested: bool,
    evaluation_rows: usize,
    tail_max_bars: usize,
) -> Result<AdaptiveStageResolutionV2, ResidentPopulationAutoSizingErrorV2> {
    if !requested {
        return Ok(AdaptiveStageResolutionV2 {
            requested: false,
            effective: false,
            max_adaptive_row_count: 0,
            reason: ADAPTIVE_RESOLUTION_DISABLED_V2,
        });
    }
    if evaluation_rows < ResidentAdaptiveBaseRequestV1::MIN_VIEW_ROWS_V1 {
        return Ok(AdaptiveStageResolutionV2 {
            requested: true,
            effective: false,
            max_adaptive_row_count: 0,
            reason: ADAPTIVE_RESOLUTION_FIXED_TOO_SHORT_V2,
        });
    }
    if tail_max_bars > 0 && evaluation_rows > tail_max_bars {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::AdaptiveTailCapExceeded,
            format!(
                "resident adaptive Stage1 has {evaluation_rows} rows, exceeding tail cap {tail_max_bars}"
            ),
        ));
    }
    Ok(AdaptiveStageResolutionV2 {
        requested: true,
        effective: true,
        max_adaptive_row_count: evaluation_rows,
        reason: ADAPTIVE_RESOLUTION_RESIDENT_EXACT_V1,
    })
}

fn build_resident_adaptive_stage1_request_v2(
    parent_rows: usize,
    stage1_row_start: usize,
    stage1_row_end: usize,
    pip_size: f64,
    tail_step: usize,
    tail_max_bars: usize,
) -> Result<
    (PopulationEvaluationViewV1, ResidentAdaptiveBaseRequestV1),
    ResidentPopulationAutoSizingErrorV2,
> {
    let view = if stage1_row_start == 0 && stage1_row_end == parent_rows {
        PopulationEvaluationViewV1::full(parent_rows, PopulationTimestampModeV1::Canonical, None)
    } else {
        PopulationEvaluationViewV1::contiguous_range(
            parent_rows,
            stage1_row_start,
            stage1_row_end,
            PopulationTimestampModeV1::Canonical,
            None,
        )
    }
    .map_err(|error| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::AdaptiveBaseNotResident,
            format!("seal resident adaptive Stage1 view: {error}"),
        )
    })?;
    let request = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(
        &view,
        pip_size,
        tail_step,
        tail_max_bars,
    )
    .map_err(|error| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::AdaptiveBaseNotResident,
            format!("seal resident adaptive Stage1 request: {error}"),
        )
    })?;
    Ok((view, request))
}

pub(crate) fn canonical_pinned_source_projection_from_search_receipt_v1(
    receipt: &crate::data_selection::CanonicalSearchInputReceiptV2,
) -> Result<CanonicalPinnedSourceProjectionV1, ResidentPopulationAutoSizingErrorV2> {
    let anchor = receipt.validate().map_err(|error| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
            format!("validate financial contract input receipt: {error}"),
        )
    })?;
    let anchor_identity = anchor.to_path_component();
    let mut anchor_bindings = receipt
        .source_bindings()
        .iter()
        .filter(|binding| binding.dataset_identity() == anchor_identity);
    let anchor_binding = anchor_bindings.next().ok_or_else(|| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
            "financial contract input receipt has no anchor source binding",
        )
    })?;
    if anchor_bindings.next().is_some() {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
            "financial contract input receipt repeats its anchor source binding",
        ));
    }
    let parent_row_count = anchor_binding
        .segments()
        .last()
        .map(crate::data_selection::CanonicalSearchSourceSegmentReceiptV1::row_end)
        .ok_or_else(|| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
                "financial contract anchor source binding has no segments",
            )
        })?;
    let mut bindings = Vec::with_capacity(receipt.source_bindings().len());
    for binding in receipt.source_bindings() {
        let dataset_identity = neoethos_data::CanonicalDatasetIdentity::from_path_component(
            binding.dataset_identity(),
        )
        .map_err(|error| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
                format!("decode financial source dataset identity: {error}"),
            )
        })?;
        let mut segments = Vec::with_capacity(binding.segments().len());
        for segment in binding.segments() {
            segments.push(
                CanonicalPinnedSourceSegmentFactsV1::checked_new(
                    segment.row_start(),
                    segment.row_end(),
                    segment.timestamp_start_ms(),
                    segment.timestamp_end_ms(),
                )
                .map_err(|error| {
                    ResidentPopulationAutoSizingErrorV2::new(
                        ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
                        format!("project financial source segment: {error}"),
                    )
                })?,
            );
        }
        bindings.push(
            CanonicalPinnedSourceBindingFactsV1::checked_new(
                dataset_identity.clone(),
                binding.manifest_schema_id(),
                decode_sha256_hex_v2(binding.manifest_sha256(), "manifest SHA-256")?,
                binding.generation_id(),
                decode_sha256_hex_v2(binding.vortex_sha256(), "Vortex SHA-256")?,
                dataset_identity.bar_timestamp_convention(),
                segments,
            )
            .map_err(|error| {
                ResidentPopulationAutoSizingErrorV2::new(
                    ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
                    format!("project financial source binding: {error}"),
                )
            })?,
        );
    }
    CanonicalPinnedSourceProjectionV1::checked_from_binding_facts_v1(
        anchor,
        parent_row_count,
        bindings,
    )
    .map_err(|error| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
            format!("seal financial immutable-source projection: {error}"),
        )
    })
}

#[derive(Clone, Debug)]
pub struct ResidentPopulationAutoSizingRequestV2 {
    population_auto: bool,
    configured_population: usize,
    requested_max_indicators: usize,
    month_capacity: usize,
    stage1_role: String,
    stage1_row_start: usize,
    stage1_row_end: usize,
    max_ordered_index_count: usize,
    max_adaptive_row_count: usize,
    migration_enabled_for_run: bool,
    adaptive_stops_requested_for_run: bool,
    adaptive_base_effective_for_stage1: bool,
    adaptive_resolution_reason: &'static str,
    resident_adaptive_request_identity_sha256: [u8; 32],
    adaptive_pip_size: f64,
    pip_value_per_lot: f64,
    financial_authority_identity_sha256: String,
    financial_input_receipt_sha256: String,
    financial_source_projection_identity_sha256: [u8; 32],
    evaluation_symbol: String,
    evaluation_account_currency: String,
    adaptive_rr: f64,
    adaptive_tail_max_bars: usize,
    adaptive_tail_step: usize,
}

impl ResidentPopulationAutoSizingRequestV2 {
    /// Derive every selection/runtime fact inside Search. Application callers
    /// may choose only the already-validated `DiscoveryConfig`; they cannot
    /// invent the Stage1 range, month capacity, migration state or adaptive
    /// stop policy carried into the native stage plan.
    pub fn from_discovery_config_v2(
        config: &crate::DiscoveryConfig,
        resident_parent_rows: usize,
        financial_contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    ) -> Result<Self, ResidentPopulationAutoSizingErrorV2> {
        if resident_parent_rows == 0 {
            return Err(ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::InvalidInput,
                "resident parent is empty",
            ));
        }
        let stage1_pct = config.runtime_overrides.resolved_funnel_stage1_pct();
        let stage1_len =
            ((resident_parent_rows as f64 * stage1_pct) as usize).min(resident_parent_rows);
        let (stage1_row_start, stage1_row_end) = match config.runtime_overrides.stage1_window {
            crate::discovery::Stage1Window::MostRecent => (
                resident_parent_rows.saturating_sub(stage1_len),
                resident_parent_rows,
            ),
            crate::discovery::Stage1Window::Earliest => (0, stage1_len),
        };
        validate_canonical_trendbar_financial_contract_against_config_v2(
            config,
            financial_contract,
        )?;
        let financial_authority_identity_sha256 =
            financial_contract.identity_sha256().map_err(|error| {
                ResidentPopulationAutoSizingErrorV2::new(
                    ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
                    format!("hash canonical-trendbar financial value authority: {error:#}"),
                )
            })?;
        let financial_source_projection =
            canonical_pinned_source_projection_from_search_receipt_v1(
                financial_contract.input_receipt(),
            )?;
        let adaptive_runtime = crate::stop_target::current_stop_target_runtime_overrides();
        let adaptive_pip_size = financial_contract.pip_size();
        let adaptive_resolution = resolve_adaptive_stage_extent_v2(
            crate::stop_target::adaptive_stops_enabled(),
            stage1_len,
            adaptive_runtime.tail_max_bars,
        )?;
        let resident_adaptive_request_identity_sha256 = if adaptive_resolution.effective {
            let (_, request) = build_resident_adaptive_stage1_request_v2(
                resident_parent_rows,
                stage1_row_start,
                stage1_row_end,
                adaptive_pip_size,
                adaptive_runtime.tail_step,
                adaptive_runtime.tail_max_bars,
            )?;
            request.identity_sha256()
        } else {
            [0; 32]
        };
        Ok(Self {
            population_auto: config.population_auto,
            configured_population: config.population,
            requested_max_indicators: config.max_indicators,
            month_capacity: crate::eval::current_backtest_runtime_overrides().month_capacity,
            stage1_role: RESIDENT_SELECTION_STAGE1_ROLE_V2.to_owned(),
            stage1_row_start,
            stage1_row_end,
            max_ordered_index_count: 0,
            max_adaptive_row_count: adaptive_resolution.max_adaptive_row_count,
            migration_enabled_for_run: crate::genetic::migration_enabled(),
            adaptive_stops_requested_for_run: adaptive_resolution.requested,
            adaptive_base_effective_for_stage1: adaptive_resolution.effective,
            adaptive_resolution_reason: adaptive_resolution.reason,
            resident_adaptive_request_identity_sha256,
            adaptive_pip_size,
            pip_value_per_lot: financial_contract.pip_value_per_lot(),
            financial_authority_identity_sha256,
            financial_input_receipt_sha256: financial_contract.input_receipt_sha256().to_owned(),
            financial_source_projection_identity_sha256: financial_source_projection
                .identity_sha256(),
            evaluation_symbol: financial_contract.symbol().to_owned(),
            evaluation_account_currency: financial_contract.account_currency().to_owned(),
            adaptive_rr: crate::stop_target::adaptive_stops_rr(),
            adaptive_tail_max_bars: adaptive_runtime.tail_max_bars,
            adaptive_tail_step: adaptive_runtime.tail_step,
        })
    }
}

fn validate_canonical_trendbar_financial_contract_against_config_v2(
    config: &crate::DiscoveryConfig,
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
) -> Result<(), ResidentPopulationAutoSizingErrorV2> {
    contract.validate().map_err(|error| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
            format!("validate explicit canonical-trendbar financial contract: {error:#}"),
        )
    })?;
    let matches_config = config.evaluation_symbol == contract.symbol()
        && config.evaluation_account_currency == contract.account_currency()
        && config.session_spread_pips.is_none()
        && config.evaluation_spread_pips.to_bits()
            == contract
                .screening_spread_and_slippage_round_trip_pips()
                .to_bits()
        && config.evaluation_commission_per_trade.to_bits()
            == contract.round_trip_commission_account_per_lot().to_bits()
        && config.swap_long_pips_per_day.to_bits() == contract.swap_long_pips_per_day().to_bits()
        && config.swap_short_pips_per_day.to_bits() == contract.swap_short_pips_per_day().to_bits();
    if !matches_config {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
            "canonical-trendbar Discovery config is detached from its explicit financial contract",
        ));
    }
    Ok(())
}

/// Resolve the exact Search evaluation settings from the explicit research
/// contract without consulting symbol metadata, a typical price, or the
/// ambient installed-contract slot. The result is intended to be carried by
/// the prepared V5 native input and consumed unchanged by Generation 0.
pub(crate) fn evaluation_config_from_canonical_trendbar_contract_v2(
    config: &crate::DiscoveryConfig,
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
) -> Result<crate::genetic::EvaluationConfig, ResidentPopulationAutoSizingErrorV2> {
    validate_canonical_trendbar_financial_contract_against_config_v2(config, contract)?;
    let mut evaluation = crate::genetic::EvaluationConfig::default();
    evaluation.symbol = contract.symbol().to_owned();
    evaluation.account_currency = contract.account_currency().to_owned();
    evaluation.pip_value = contract.pip_size();
    evaluation.pip_value_per_lot = contract.pip_value_per_lot();
    evaluation.spread_pips = contract.screening_spread_and_slippage_round_trip_pips();
    evaluation.commission_per_trade = contract.round_trip_commission_account_per_lot();
    evaluation.swap_long_pips_per_day = contract.swap_long_pips_per_day();
    evaluation.swap_short_pips_per_day = contract.swap_short_pips_per_day();
    evaluation.pnl_conversion_fee_rate = contract.pnl_conversion_fee_rate();
    evaluation.growth_objective = matches!(config.mode, crate::discovery::DiscoveryMode::Risky);
    Ok(evaluation)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PopulationExtentResolutionV2 {
    resolved_population: usize,
    max_concurrent_scenario_count: usize,
    memory_one_launch_population_cap: usize,
    growth_cap: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceFitV2 {
    Fits,
    NoRoom,
}

fn highest_fitting_v2<F>(
    mut low: usize,
    mut high: usize,
    mut fits: F,
) -> Result<usize, ResidentPopulationAutoSizingErrorV2>
where
    F: FnMut(usize) -> Result<WorkspaceFitV2, ResidentPopulationAutoSizingErrorV2>,
{
    let mut best = 0usize;
    while low <= high {
        let distance = high.checked_sub(low).ok_or_else(|| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                "binary-search bounds underflowed",
            )
        })?;
        let middle = low.checked_add(distance / 2).ok_or_else(|| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                "binary-search midpoint overflowed",
            )
        })?;
        match fits(middle)? {
            WorkspaceFitV2::Fits => {
                best = middle;
                low = middle.checked_add(1).ok_or_else(|| {
                    ResidentPopulationAutoSizingErrorV2::new(
                        ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                        "binary-search lower bound overflowed",
                    )
                })?;
            }
            WorkspaceFitV2::NoRoom => {
                if middle == 0 {
                    break;
                }
                high = middle - 1;
            }
        }
    }
    Ok(best)
}

fn resolve_population_auto_extents_v2<F>(
    population_auto: bool,
    configured_population: usize,
    effective_time_cap: usize,
    hard_growth_cap: usize,
    mut workspace_fit: F,
) -> Result<PopulationExtentResolutionV2, ResidentPopulationAutoSizingErrorV2>
where
    F: FnMut(usize, usize) -> Result<WorkspaceFitV2, ResidentPopulationAutoSizingErrorV2>,
{
    if configured_population == 0 || effective_time_cap == 0 || hard_growth_cap == 0 {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::InvalidInput,
            "configured population and time/hard caps must be non-zero",
        ));
    }
    if workspace_fit(configured_population, 1)? != WorkspaceFitV2::Fits {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ConfiguredGeneNoRoom,
            "the configured unsplittable gene store plus one metrics-only scenario does not fit",
        ));
    }

    let automatic_upper = effective_time_cap.min(hard_growth_cap);
    let memory_one_launch_population_cap = highest_fitting_v2(1, automatic_upper, |candidate| {
        workspace_fit(candidate, candidate)
    })?;
    if memory_one_launch_population_cap == 0 {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ScenarioNoRoom,
            "the admitted Data workspace leaves no room for one strict metrics-only scenario",
        ));
    }
    let growth_cap = memory_one_launch_population_cap.min(automatic_upper);
    let resolved_population = if population_auto {
        configured_population.max(growth_cap)
    } else {
        configured_population
    };
    let scenario_upper = resolved_population.min(automatic_upper).max(1);
    let max_concurrent_scenario_count = highest_fitting_v2(1, scenario_upper, |scenarios| {
        workspace_fit(resolved_population, scenarios)
    })?;
    if max_concurrent_scenario_count == 0 {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ScenarioNoRoom,
            "the resolved gene store leaves no room for one strict metrics-only scenario",
        ));
    }
    Ok(PopulationExtentResolutionV2 {
        resolved_population,
        max_concurrent_scenario_count,
        memory_one_launch_population_cap,
        growth_cap,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResidentPopulationAutoSizingReceiptV2 {
    schema_version: u16,
    population_auto: bool,
    configured_population: u64,
    resolved_population: u64,
    resident_parent_rows: u64,
    feature_count: u64,
    evaluation_rows: u64,
    month_capacity: u64,
    requested_max_indicators: u64,
    term_cap: u64,
    stage1_role: String,
    stage1_row_start: u64,
    stage1_row_end: u64,
    migration_enabled_for_run: bool,
    adaptive_stops_requested_for_run: bool,
    adaptive_base_effective_for_stage1: bool,
    adaptive_resolution_reason: String,
    resident_adaptive_semantic_v1: String,
    stop_target_log_operation_schedule_v3: String,
    resident_adaptive_request_identity_sha256: [u8; 32],
    adaptive_pip_size_bits: u64,
    pip_value_per_lot_bits: u64,
    financial_authority_identity_sha256: String,
    financial_input_receipt_sha256: String,
    financial_source_projection_identity_sha256: [u8; 32],
    evaluation_symbol: String,
    evaluation_account_currency: String,
    adaptive_rr_bits: u64,
    adaptive_tail_max_bars: u64,
    adaptive_tail_step: u64,
    max_ordered_index_count: u64,
    max_adaptive_row_count: u64,
    selected_device_ordinal: u32,
    pre_materialization_free_bytes_snapshot: u64,
    allocator_context_reserve_bytes: u64,
    allocator_context_reserve_policy: String,
    admission_identity_sha256: String,
    native_preflight_facts_identity_sha256: String,
    cuda_build_manifest_sha256: String,
    cuda_build_artifact_sha256: String,
    data_peak_device_bytes: u64,
    data_steady_device_bytes: u64,
    gene_store_device_bytes: u64,
    metrics_scenario_device_bytes: u64,
    max_concurrent_scenario_count: u64,
    bounded_host_metric_readback_bytes: u64,
    required_device_bytes_excluding_reserve: u64,
    required_device_bytes_including_reserve: u64,
    raw_time_cap: u64,
    effective_time_cap: u64,
    hard_growth_cap: u64,
    memory_one_launch_population_cap: u64,
    growth_cap: u64,
    resolution_reason: String,
    workspace_plan_identity_sha256: String,
    population_sizing_authority_sha256: String,
    data_extent_identity_sha256: String,
    identity_sha256: String,
}

impl ResidentPopulationAutoSizingReceiptV2 {
    pub const fn population_auto(&self) -> bool {
        self.population_auto
    }

    pub const fn configured_population(&self) -> usize {
        self.configured_population as usize
    }

    pub const fn resolved_population(&self) -> usize {
        self.resolved_population as usize
    }

    pub const fn feature_count(&self) -> usize {
        self.feature_count as usize
    }

    pub const fn term_cap(&self) -> usize {
        self.term_cap as usize
    }

    pub const fn requested_max_indicators(&self) -> usize {
        self.requested_max_indicators as usize
    }

    pub const fn migration_enabled_for_run(&self) -> bool {
        self.migration_enabled_for_run
    }

    pub const fn month_capacity(&self) -> usize {
        self.month_capacity as usize
    }

    pub const fn stage1_row_start(&self) -> usize {
        self.stage1_row_start as usize
    }

    pub const fn stage1_row_end(&self) -> usize {
        self.stage1_row_end as usize
    }

    pub fn stage1_role(&self) -> &str {
        &self.stage1_role
    }

    pub const fn max_concurrent_scenario_count(&self) -> usize {
        self.max_concurrent_scenario_count as usize
    }

    pub const fn bounded_host_metric_readback_bytes(&self) -> u64 {
        self.bounded_host_metric_readback_bytes
    }

    pub const fn hard_growth_cap(&self) -> usize {
        self.hard_growth_cap as usize
    }

    pub const fn selected_device_ordinal(&self) -> u32 {
        self.selected_device_ordinal
    }

    pub const fn raw_time_cap(&self) -> u64 {
        self.raw_time_cap
    }

    pub const fn effective_time_cap(&self) -> u64 {
        self.effective_time_cap
    }

    pub fn resolution_reason(&self) -> &str {
        &self.resolution_reason
    }

    pub const fn adaptive_pip_size(&self) -> f64 {
        f64::from_bits(self.adaptive_pip_size_bits)
    }

    pub const fn pip_value_per_lot(&self) -> f64 {
        f64::from_bits(self.pip_value_per_lot_bits)
    }

    pub fn financial_authority_identity_sha256(&self) -> &str {
        &self.financial_authority_identity_sha256
    }

    pub const fn financial_source_projection_identity_sha256(&self) -> [u8; 32] {
        self.financial_source_projection_identity_sha256
    }

    /// Bind the financial-value contract to Data's node-name-independent
    /// immutable-source projection. This proves that the cost scalars apply to
    /// the same market rows; it deliberately does not claim that the
    /// contract's CPU V2 feature receipt is the native V3 feature input.
    pub fn validate_financial_authority_against_pinned_source_projection_v2(
        &self,
        contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
        source_projection: &CanonicalPinnedSourceProjectionV1,
    ) -> Result<(), ResidentPopulationAutoSizingErrorV2> {
        contract.validate().map_err(|error| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                format!("validate bound canonical-trendbar financial authority: {error:#}"),
            )
        })?;
        let contract_identity = contract.identity_sha256().map_err(|error| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                format!("hash bound canonical-trendbar financial authority: {error:#}"),
            )
        })?;
        let contract_source_projection =
            canonical_pinned_source_projection_from_search_receipt_v1(contract.input_receipt())?;
        let valid = self.financial_authority_identity_sha256 == contract_identity
            && self.financial_input_receipt_sha256 == contract.input_receipt_sha256()
            && self.financial_source_projection_identity_sha256
                == contract_source_projection.identity_sha256()
            && self.financial_source_projection_identity_sha256
                == source_projection.identity_sha256()
            && &contract_source_projection == source_projection
            && self.evaluation_symbol == contract.symbol()
            && self.evaluation_account_currency == contract.account_currency()
            && self.adaptive_pip_size_bits == contract.pip_size().to_bits()
            && self.pip_value_per_lot_bits == contract.pip_value_per_lot().to_bits();
        if !valid {
            return Err(ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                "resident financial value authority is detached from the immutable-source projection",
            ));
        }
        Ok(())
    }

    fn internal_workspace_bytes_v2(
        &self,
        candidate_count: usize,
        scenario_count: usize,
        term_cap: usize,
        month_capacity: u32,
    ) -> Result<InternalWorkspaceBytesV2, ResidentPopulationAutoSizingErrorV2> {
        let term_count = candidate_count.checked_mul(term_cap).ok_or_else(|| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                "receipt candidate count multiplied by term cap overflowed",
            )
        })?;
        let gene_plan =
            PopulationGeneStorePlanV1::checked_from_gene_extents_v1(candidate_count, term_count)
                .map_err(|error| map_native_plan_error_v2("rebuild receipt gene plan", error))?;
        let metrics_plan = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
            scenario_count,
            month_capacity,
        )
        .map_err(|error| map_native_plan_error_v2("rebuild receipt metric plan", error))?;
        let resident_peak = self
            .max_ordered_index_count
            .checked_mul(8)
            .and_then(|ordered| {
                self.max_adaptive_row_count
                    .checked_mul(8)
                    .and_then(|adaptive| ordered.checked_add(adaptive))
            })
            .and_then(|views| self.resident_parent_rows.checked_add(views))
            .and_then(|bytes| bytes.checked_add(gene_plan.total_device_bytes()))
            .and_then(|bytes| bytes.checked_add(metrics_plan.total_device_bytes()))
            .and_then(|bytes| self.data_steady_device_bytes.checked_add(bytes))
            .ok_or_else(|| {
                ResidentPopulationAutoSizingErrorV2::new(
                    ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                    "receipt resident workspace byte arithmetic overflowed",
                )
            })?;
        let excluding = self.data_peak_device_bytes.max(resident_peak);
        let including = excluding
            .checked_add(self.allocator_context_reserve_bytes)
            .ok_or_else(|| {
                ResidentPopulationAutoSizingErrorV2::new(
                    ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                    "receipt workspace plus allocator reserve overflowed",
                )
            })?;
        Ok((gene_plan, metrics_plan, excluding, including))
    }

    pub const fn adaptive_rr(&self) -> f64 {
        f64::from_bits(self.adaptive_rr_bits)
    }

    pub const fn adaptive_tail_max_bars(&self) -> usize {
        self.adaptive_tail_max_bars as usize
    }

    pub const fn adaptive_tail_step(&self) -> usize {
        self.adaptive_tail_step as usize
    }

    pub const fn adaptive_stops_requested_for_run(&self) -> bool {
        self.adaptive_stops_requested_for_run
    }

    pub const fn adaptive_base_effective_for_stage1(&self) -> bool {
        self.adaptive_base_effective_for_stage1
    }

    pub fn adaptive_resolution_reason(&self) -> &str {
        &self.adaptive_resolution_reason
    }

    pub fn resident_adaptive_semantic_v1(&self) -> Option<&str> {
        self.adaptive_base_effective_for_stage1
            .then_some(self.resident_adaptive_semantic_v1.as_str())
    }

    pub fn stop_target_log_operation_schedule_v3(&self) -> Option<&str> {
        self.adaptive_base_effective_for_stage1
            .then_some(self.stop_target_log_operation_schedule_v3.as_str())
    }

    /// Rebuild the exact view/request admitted before Data allocation. A
    /// too-short Stage1 intentionally returns `None`, matching the canonical
    /// CPU fixed-pip fallback. No caller may invent a different view recipe.
    pub fn resident_adaptive_view_and_request_v2(
        &self,
    ) -> Result<
        Option<(PopulationEvaluationViewV1, ResidentAdaptiveBaseRequestV1)>,
        ResidentPopulationAutoSizingErrorV2,
    > {
        if !self.adaptive_base_effective_for_stage1 {
            let valid_fixed = self.max_adaptive_row_count == 0
                && self.resident_adaptive_request_identity_sha256 == [0; 32]
                && self.resident_adaptive_semantic_v1.is_empty()
                && self.stop_target_log_operation_schedule_v3.is_empty()
                && ((!self.adaptive_stops_requested_for_run
                    && self.adaptive_resolution_reason == ADAPTIVE_RESOLUTION_DISABLED_V2)
                    || (self.adaptive_stops_requested_for_run
                        && self.adaptive_resolution_reason
                            == ADAPTIVE_RESOLUTION_FIXED_TOO_SHORT_V2
                        && self.evaluation_rows
                            < ResidentAdaptiveBaseRequestV1::MIN_VIEW_ROWS_V1 as u64));
            return if valid_fixed {
                Ok(None)
            } else {
                Err(ResidentPopulationAutoSizingErrorV2::new(
                    ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                    "resident adaptive fixed-fallback receipt is internally contradictory",
                ))
            };
        }
        if !self.adaptive_stops_requested_for_run
            || self.adaptive_resolution_reason != ADAPTIVE_RESOLUTION_RESIDENT_EXACT_V1
            || self.resident_adaptive_semantic_v1 != RESIDENT_ADAPTIVE_BASE_SEMANTIC_V1
            || self.stop_target_log_operation_schedule_v3
                != crate::stop_target::STOP_TARGET_LOG_OPERATION_SCHEDULE_V3
            || self.max_adaptive_row_count != self.evaluation_rows
            || self.evaluation_rows < ResidentAdaptiveBaseRequestV1::MIN_VIEW_ROWS_V1 as u64
            || self.resident_adaptive_request_identity_sha256 == [0; 32]
        {
            return Err(ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                "resident adaptive effective receipt is internally contradictory",
            ));
        }
        let parent_rows = usize::try_from(self.resident_parent_rows).map_err(|_| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                "resident adaptive parent rows do not fit this process",
            )
        })?;
        let (view, request) = build_resident_adaptive_stage1_request_v2(
            parent_rows,
            self.stage1_row_start(),
            self.stage1_row_end(),
            self.adaptive_pip_size(),
            self.adaptive_tail_step(),
            self.adaptive_tail_max_bars(),
        )?;
        if request.identity_sha256() != self.resident_adaptive_request_identity_sha256 {
            return Err(ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                "resident adaptive request identity drifted from the sizing receipt",
            ));
        }
        Ok(Some((view, request)))
    }

    /// Accept only the move-only token minted by the resident session for the
    /// exact request sealed into this receipt. Session/view identities remain
    /// opaque and must both be non-zero; no caller-injected hashes are used.
    pub fn validate_resident_adaptive_view_token_v2(
        &self,
        request: &ResidentAdaptiveBaseRequestV1,
        token: &ResidentAdaptiveBaseViewTokenV1,
    ) -> Result<(), ResidentPopulationAutoSizingErrorV2> {
        let (_, expected_request) =
            self.resident_adaptive_view_and_request_v2()?
                .ok_or_else(|| {
                    ResidentPopulationAutoSizingErrorV2::new(
                        ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                        "fixed-pip Stage1 cannot accept a resident adaptive token",
                    )
                })?;
        let valid = request == &expected_request
            && token.request_identity_sha256() == self.resident_adaptive_request_identity_sha256
            && token.resident_session_identity_sha256() != [0; 32]
            && token.view_identity_sha256() != [0; 32]
            && token.token_identity_sha256() != [0; 32];
        if !valid {
            return Err(ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                "resident adaptive token is detached from its admitted request/session/view",
            ));
        }
        Ok(())
    }

    pub fn workspace_plan_identity_sha256(&self) -> &str {
        &self.workspace_plan_identity_sha256
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub fn population_sizing_authority_sha256(&self) -> &str {
        &self.population_sizing_authority_sha256
    }

    pub fn data_extent_identity_sha256(&self) -> &str {
        &self.data_extent_identity_sha256
    }

    pub fn validate_against_execution_limits_v2(
        &self,
        selected_device_ordinal: u32,
        pre_materialization_free_bytes_snapshot: u64,
        parent_rows: usize,
        feature_count: usize,
        limits: &SealedDataPopulationExecutionLimitsV1,
    ) -> Result<(), ResidentPopulationAutoSizingErrorV2> {
        let parent_rows = checked_u64(parent_rows, "bound resident parent rows")?;
        let feature_count = checked_u64(feature_count, "bound resident feature count")?;
        let identity = self.computed_identity_sha256()?;
        let valid = self.schema_version
            == RESIDENT_POPULATION_AUTO_SIZING_RECEIPT_SCHEMA_VERSION_V2
            && self.identity_sha256 == identity
            && self.selected_device_ordinal == selected_device_ordinal
            && self.pre_materialization_free_bytes_snapshot
                == pre_materialization_free_bytes_snapshot
            && self.resident_parent_rows == parent_rows
            && self.feature_count == feature_count
            && self.workspace_plan_identity_sha256
                == hex_lower(limits.workspace_plan_identity_sha256())
            && self.population_sizing_authority_sha256
                == hex_lower(limits.population_sizing_authority_sha256())
            && self.data_extent_identity_sha256 == hex_lower(limits.data_extent_identity_sha256())
            && self.resident_parent_rows == limits.parent_row_count()
            && self.feature_count == limits.feature_count()
            && self.max_ordered_index_count == limits.max_ordered_index_count()
            && self.max_adaptive_row_count == limits.max_adaptive_row_count()
            && self.resolved_population == limits.max_candidate_count()
            && self
                .resolved_population
                .checked_mul(self.term_cap)
                .is_some_and(|terms| terms == limits.max_gene_term_count())
            && self.max_concurrent_scenario_count == limits.max_concurrent_scenario_count()
            && self.month_capacity == limits.month_capacity()
            && self.bounded_host_metric_readback_bytes
                == limits.bounded_host_metric_readback_bytes()
            && self.pip_value_per_lot().is_finite()
            && self.pip_value_per_lot() > 0.0
            && is_nonzero_sha256(&self.financial_authority_identity_sha256)
            && is_nonzero_sha256(&self.financial_input_receipt_sha256)
            && self.financial_source_projection_identity_sha256 != [0; 32]
            && !self.evaluation_symbol.trim().is_empty()
            && !self.evaluation_account_currency.trim().is_empty()
            && !self.migration_enabled_for_run
            && self.resident_adaptive_view_and_request_v2().is_ok();
        if !valid {
            return Err(ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                "resident sizing receipt is detached from its bound execution limits",
            ));
        }
        Ok(())
    }

    /// Revalidate only algebraic invariants sealed inside the V2 receipt.
    /// Exact prepared R/F, native ordinal/free/base-Data facts, externally
    /// selected hard policy, and identity provenance still require the existing
    /// native/workspace validator (and the later 2A2 binder).
    pub(crate) fn validate_self_v2(&self) -> Result<(), ResidentPopulationAutoSizingErrorV2> {
        let computed_identity = self.computed_identity_sha256()?;
        let to_usize = |value: u64, field: &'static str| {
            usize::try_from(value).map_err(|_| {
                ResidentPopulationAutoSizingErrorV2::new(
                    ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                    format!("receipt {field} does not fit this process"),
                )
            })
        };
        let feature_count = to_usize(self.feature_count, "feature count")?;
        let requested_max_indicators = to_usize(
            self.requested_max_indicators,
            "requested maximum indicators",
        )?;
        let configured_population = to_usize(self.configured_population, "configured population")?;
        let resolved_population = to_usize(self.resolved_population, "resolved population")?;
        let evaluation_rows = to_usize(self.evaluation_rows, "evaluation rows")?;
        let term_cap = to_usize(self.term_cap, "term cap")?;
        let scenario_count = to_usize(
            self.max_concurrent_scenario_count,
            "maximum concurrent scenario count",
        )?;
        let hard_growth_cap = to_usize(self.hard_growth_cap, "hard growth cap")?;
        let memory_population_cap = to_usize(
            self.memory_one_launch_population_cap,
            "memory one-launch population cap",
        )?;
        let growth_cap = to_usize(self.growth_cap, "growth cap")?;
        let month_capacity = u32::try_from(self.month_capacity).map_err(|_| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                "receipt month capacity does not fit the native metric planner",
            )
        })?;
        let expected_term_cap = feature_count.min(
            requested_max_indicators
                .max(crate::genetic::seed_templates::PROFESSIONAL_TEMPLATE_MAX_TERMS_V1),
        );
        let (expected_raw_time_cap, expected_effective_time_cap, _) =
            crate::gpu_native::prototype_b_population_eval::checked_candidates_for_target_launch_v1(
                evaluation_rows,
            )
            .map_err(|error| map_native_plan_error_v2("rebuild receipt time plan", error))?;
        let expected_effective_time_cap =
            to_usize(expected_effective_time_cap, "recomputed effective time cap")?;
        let expected_resolution = resolve_population_auto_extents_v2(
            self.population_auto,
            configured_population,
            expected_effective_time_cap,
            hard_growth_cap,
            |candidate_count, scenario_count| {
                let (_, _, _, required) = self.internal_workspace_bytes_v2(
                    candidate_count,
                    scenario_count,
                    expected_term_cap,
                    month_capacity,
                )?;
                let fits = required <= self.pre_materialization_free_bytes_snapshot;
                Ok(if fits {
                    WorkspaceFitV2::Fits
                } else {
                    WorkspaceFitV2::NoRoom
                })
            },
        )?;
        let (gene_plan, metrics_plan, expected_required_excluding, expected_required_including) =
            self.internal_workspace_bytes_v2(
                expected_resolution.resolved_population,
                expected_resolution.max_concurrent_scenario_count,
                expected_term_cap,
                month_capacity,
            )?;
        let expected_resolution_reason = if !self.population_auto {
            "auto_disabled"
        } else if expected_resolution.resolved_population > configured_population {
            "resident_cuda_auto_grew"
        } else if configured_population > expected_resolution.growth_cap {
            "resident_cuda_configured_above_growth_cap_no_shrink"
        } else {
            "resident_cuda_configured_at_growth_cap"
        };
        let canonical_hashes = [
            self.financial_authority_identity_sha256.as_str(),
            self.financial_input_receipt_sha256.as_str(),
            self.admission_identity_sha256.as_str(),
            self.native_preflight_facts_identity_sha256.as_str(),
            self.cuda_build_manifest_sha256.as_str(),
            self.cuda_build_artifact_sha256.as_str(),
            self.workspace_plan_identity_sha256.as_str(),
            self.population_sizing_authority_sha256.as_str(),
            self.data_extent_identity_sha256.as_str(),
            self.identity_sha256.as_str(),
        ]
        .into_iter()
        .all(is_nonzero_canonical_sha256_v2);
        let valid = self.schema_version
            == RESIDENT_POPULATION_AUTO_SIZING_RECEIPT_SCHEMA_VERSION_V2
            && self.identity_sha256 == computed_identity
            && self.configured_population > 0
            && self.resolved_population >= self.configured_population
            && resolved_population == expected_resolution.resolved_population
            && self.resident_parent_rows > 0
            && self.feature_count > 0
            && self.evaluation_rows > 0
            && self.stage1_row_end.checked_sub(self.stage1_row_start) == Some(self.evaluation_rows)
            && self.stage1_row_end <= self.resident_parent_rows
            && self.max_ordered_index_count <= self.resident_parent_rows
            && self.max_adaptive_row_count <= self.resident_parent_rows
            && self.month_capacity > 0
            && self.requested_max_indicators > 0
            && term_cap == expected_term_cap
            && self.stage1_role == RESIDENT_SELECTION_STAGE1_ROLE_V2
            && !self.migration_enabled_for_run
            && self.adaptive_pip_size().is_finite()
            && self.adaptive_pip_size() > 0.0
            && self.pip_value_per_lot().is_finite()
            && self.pip_value_per_lot() > 0.0
            && self.adaptive_rr().is_finite()
            && self.adaptive_rr() > 0.0
            && self.adaptive_tail_step > 0
            && self.financial_source_projection_identity_sha256 != [0; 32]
            && !self.evaluation_symbol.trim().is_empty()
            && !self.evaluation_account_currency.trim().is_empty()
            && self.raw_time_cap == expected_raw_time_cap
            && self.effective_time_cap == expected_effective_time_cap as u64
            && (1..=RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2).contains(&hard_growth_cap)
            && memory_population_cap == expected_resolution.memory_one_launch_population_cap
            && growth_cap == expected_resolution.growth_cap
            && scenario_count == expected_resolution.max_concurrent_scenario_count
            && self.data_peak_device_bytes > 0
            && self.data_steady_device_bytes > 0
            && self.data_peak_device_bytes >= self.data_steady_device_bytes
            && self.gene_store_device_bytes == gene_plan.total_device_bytes()
            && self.metrics_scenario_device_bytes == metrics_plan.total_device_bytes()
            && self.bounded_host_metric_readback_bytes == metrics_plan.metric_rows_bytes()
            && self.required_device_bytes_excluding_reserve == expected_required_excluding
            && self.required_device_bytes_including_reserve == expected_required_including
            && self.allocator_context_reserve_bytes == DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1
            && self.allocator_context_reserve_policy == DATA_POPULATION_ALLOCATOR_RESERVE_POLICY_V1
            && expected_required_including <= self.pre_materialization_free_bytes_snapshot
            && self.resolution_reason == expected_resolution_reason
            && canonical_hashes
            && self.resident_adaptive_view_and_request_v2().is_ok();
        if !valid {
            return Err(ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                "resident sizing receipt failed internal self-validation",
            ));
        }
        Ok(())
    }

    fn computed_identity_sha256(&self) -> Result<String, ResidentPopulationAutoSizingErrorV2> {
        let mut body = self.clone();
        body.identity_sha256.clear();
        let encoded = serde_json::to_vec(&body).map_err(|error| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                format!("serialize resident sizing receipt: {error}"),
            )
        })?;
        let mut hasher = Sha256::new();
        hasher.update(RESIDENT_POPULATION_AUTO_SIZING_RECEIPT_HASH_DOMAIN_V2);
        hasher.update(encoded);
        Ok(hex_lower(hasher.finalize().into()))
    }

    pub fn validate_against_workspace_plan_v2(
        &self,
        native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
        workspace_plan: &SealedDataPopulationGpuWorkspacePlanV1,
    ) -> Result<(), ResidentPopulationAutoSizingErrorV2> {
        let limits = workspace_plan.limits();
        let identity = self.computed_identity_sha256()?;
        let valid = self.schema_version
            == RESIDENT_POPULATION_AUTO_SIZING_RECEIPT_SCHEMA_VERSION_V2
            && self.identity_sha256 == identity
            && self.admission_identity_sha256
                == hex_lower(native_facts.admission_identity_sha256())
            && self.native_preflight_facts_identity_sha256
                == hex_lower(native_facts.facts_identity_sha256())
            && self.cuda_build_manifest_sha256
                == hex_lower(native_facts.cuda_build_manifest_sha256())
            && self.cuda_build_artifact_sha256
                == hex_lower(native_facts.cuda_build_artifact_sha256())
            && self.selected_device_ordinal == native_facts.selected_device_ordinal()
            && self.pre_materialization_free_bytes_snapshot
                == native_facts.pre_materialization_free_bytes_snapshot()
            && self.allocator_context_reserve_bytes
                == native_facts.allocator_context_reserve_bytes()
            && self.allocator_context_reserve_policy
                == native_facts.allocator_context_reserve_policy()
            && self.workspace_plan_identity_sha256
                == hex_lower(limits.workspace_plan_identity_sha256())
            && self.population_sizing_authority_sha256
                == hex_lower(limits.population_sizing_authority_sha256())
            && self.data_extent_identity_sha256 == hex_lower(limits.data_extent_identity_sha256())
            && self.resident_parent_rows == limits.parent_row_count()
            && self.feature_count == limits.feature_count()
            && self.max_ordered_index_count == limits.max_ordered_index_count()
            && self.max_adaptive_row_count == limits.max_adaptive_row_count()
            && self.resolved_population == limits.max_candidate_count()
            && self
                .resolved_population
                .checked_mul(self.term_cap)
                .is_some_and(|terms| terms == limits.max_gene_term_count())
            && self.max_concurrent_scenario_count == limits.max_concurrent_scenario_count()
            && self.month_capacity == limits.month_capacity()
            && self.bounded_host_metric_readback_bytes
                == limits.bounded_host_metric_readback_bytes()
            && self.data_peak_device_bytes == workspace_plan.data_peak_device_bytes()
            && self.data_steady_device_bytes == workspace_plan.data_steady_device_bytes()
            && self.gene_store_device_bytes == workspace_plan.gene_store_device_bytes()
            && self.metrics_scenario_device_bytes == workspace_plan.metrics_scenario_device_bytes()
            && self.required_device_bytes_excluding_reserve
                == workspace_plan.required_device_bytes_excluding_reserve()
            && self.required_device_bytes_including_reserve
                == workspace_plan.required_device_bytes_including_reserve()
            && !self.migration_enabled_for_run
            && self.resident_adaptive_view_and_request_v2().is_ok()
            && self.stage1_row_end > self.stage1_row_start
            && self.evaluation_rows == self.stage1_row_end - self.stage1_row_start
            && self.stage1_row_end <= self.resident_parent_rows
            && self.pip_value_per_lot().is_finite()
            && self.pip_value_per_lot() > 0.0
            && is_nonzero_sha256(&self.financial_authority_identity_sha256)
            && is_nonzero_sha256(&self.financial_input_receipt_sha256)
            && self.financial_source_projection_identity_sha256 != [0; 32]
            && !self.evaluation_symbol.trim().is_empty()
            && !self.evaluation_account_currency.trim().is_empty();
        if !valid {
            return Err(ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
                "resident sizing receipt is detached from its native facts or sealed workspace plan",
            ));
        }
        Ok(())
    }
}

fn hex_lower(bytes: [u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn is_nonzero_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value.bytes().any(|byte| byte != b'0')
}

fn is_nonzero_canonical_sha256_v2(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn decode_sha256_hex_v2(
    value: &str,
    field: &'static str,
) -> Result<[u8; 32], ResidentPopulationAutoSizingErrorV2> {
    if !is_nonzero_sha256(value) {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
            format!("{field} is not a non-zero SHA-256"),
        ));
    }
    let mut decoded = [0_u8; 32];
    for (index, slot) in decoded.iter_mut().enumerate() {
        let offset = index * 2;
        *slot = u8::from_str_radix(&value[offset..offset + 2], 16).map_err(|_| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ExactFinancialAuthorityUnavailable,
                format!("{field} contains invalid hexadecimal"),
            )
        })?;
    }
    Ok(decoded)
}

fn checked_u64(
    value: usize,
    field: &'static str,
) -> Result<u64, ResidentPopulationAutoSizingErrorV2> {
    u64::try_from(value).map_err(|_| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
            format!("{field} does not fit u64"),
        )
    })
}

fn map_native_plan_error_v2(
    context: &'static str,
    error: neoethos_gpu_cuda::CudaPopulationError,
) -> ResidentPopulationAutoSizingErrorV2 {
    let code = match error {
        neoethos_gpu_cuda::CudaPopulationError::InvalidInput(_) => {
            ResidentPopulationAutoSizingErrorCodeV2::InvalidInput
        }
        neoethos_gpu_cuda::CudaPopulationError::RuntimeUnavailable
        | neoethos_gpu_cuda::CudaPopulationError::Native { .. } => {
            ResidentPopulationAutoSizingErrorCodeV2::WorkspacePlan
        }
    };
    ResidentPopulationAutoSizingErrorV2::new(code, format!("{context}: {error}"))
}

fn workspace_plan_attempt_v2(
    prepared: &PreparedGpuOnlyFeatureMaterializationV3,
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
    request: &ResidentPopulationAutoSizingRequestV2,
    term_cap: usize,
    candidate_count: usize,
    scenario_count: usize,
) -> Result<SealedDataPopulationGpuWorkspacePlanV1, ResidentPopulationAutoSizingErrorV2> {
    let term_count = candidate_count.checked_mul(term_cap).ok_or_else(|| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
            "candidate count multiplied by sealed term cap overflows",
        )
    })?;
    let gene_plan =
        PopulationGeneStorePlanV1::checked_from_gene_extents_v1(candidate_count, term_count)
            .map_err(|error| map_native_plan_error_v2("resident gene plan", error))?;
    let month_capacity = u32::try_from(request.month_capacity).map_err(|_| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
            "month capacity does not fit the native u32 planner",
        )
    })?;
    let metrics_plan = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
        scenario_count,
        month_capacity,
    )
    .map_err(|error| map_native_plan_error_v2("resident metrics-only plan", error))?;
    prepared
        .seal_data_population_workspace_plan_v1(
            *native_facts,
            request.max_ordered_index_count,
            request.max_adaptive_row_count,
            gene_plan,
            metrics_plan,
        )
        .map_err(|error| {
            let no_room = match &error {
                GpuOnlyFeatureMaterializationErrorV3::Other(source) => source
                    .downcast_ref::<DataPopulationWorkspacePlanErrorV1>()
                    .is_some_and(|source| {
                        source.code()
                            == DataPopulationWorkspacePlanErrorCodeV1::InsufficientExactOrdinalMemory
                    }),
                _ => false,
            };
            if no_room {
                ResidentPopulationAutoSizingErrorV2::new(
                    ResidentPopulationAutoSizingErrorCodeV2::ScenarioNoRoom,
                    error.to_string(),
                )
            } else {
                ResidentPopulationAutoSizingErrorV2::new(
                    ResidentPopulationAutoSizingErrorCodeV2::WorkspacePlan,
                    error.to_string(),
                )
            }
        })
}

fn workspace_fit_for_extents_v2(
    prepared: &PreparedGpuOnlyFeatureMaterializationV3,
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
    request: &ResidentPopulationAutoSizingRequestV2,
    term_cap: usize,
    candidate_count: usize,
    scenario_count: usize,
) -> Result<WorkspaceFitV2, ResidentPopulationAutoSizingErrorV2> {
    match workspace_plan_attempt_v2(
        prepared,
        native_facts,
        request,
        term_cap,
        candidate_count,
        scenario_count,
    ) {
        Ok(_) => Ok(WorkspaceFitV2::Fits),
        Err(error) if error.code() == ResidentPopulationAutoSizingErrorCodeV2::ScenarioNoRoom => {
            Ok(WorkspaceFitV2::NoRoom)
        }
        Err(error) => Err(error),
    }
}

pub fn seal_resident_population_auto_sizing_receipt_v2(
    prepared: &PreparedGpuOnlyFeatureMaterializationV3,
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
    request: ResidentPopulationAutoSizingRequestV2,
) -> Result<
    (
        ResidentPopulationAutoSizingReceiptV2,
        SealedDataPopulationGpuWorkspacePlanV1,
    ),
    ResidentPopulationAutoSizingErrorV2,
> {
    seal_resident_population_auto_sizing_receipt_with_hard_cap_v2(
        prepared,
        native_facts,
        request,
        RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2,
    )
}

fn seal_resident_population_auto_sizing_receipt_with_hard_cap_v2(
    prepared: &PreparedGpuOnlyFeatureMaterializationV3,
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
    request: ResidentPopulationAutoSizingRequestV2,
    external_hard_population_cap: usize,
) -> Result<
    (
        ResidentPopulationAutoSizingReceiptV2,
        SealedDataPopulationGpuWorkspacePlanV1,
    ),
    ResidentPopulationAutoSizingErrorV2,
> {
    let hard_growth_cap = checked_effective_hard_growth_cap_v2(external_hard_population_cap)?;
    let extent = prepared.workspace_extent();
    let parent_rows = usize::try_from(extent.row_count()).map_err(|_| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
            "resident parent rows do not fit this process",
        )
    })?;
    let feature_count = usize::try_from(extent.column_count()).map_err(|_| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
            "resident feature count does not fit this process",
        )
    })?;
    if request.configured_population == 0
        || request.requested_max_indicators == 0
        || request.month_capacity == 0
        || request.stage1_role.trim().is_empty()
        || request.stage1_row_end <= request.stage1_row_start
        || request.stage1_row_end > parent_rows
        || request.max_ordered_index_count > parent_rows
        || request.max_adaptive_row_count > parent_rows
        || !(request.adaptive_pip_size.is_finite() && request.adaptive_pip_size > 0.0)
        || !(request.pip_value_per_lot.is_finite() && request.pip_value_per_lot > 0.0)
        || !is_nonzero_sha256(&request.financial_authority_identity_sha256)
        || !is_nonzero_sha256(&request.financial_input_receipt_sha256)
        || request.financial_source_projection_identity_sha256 == [0; 32]
        || request.evaluation_symbol.trim().is_empty()
        || request.evaluation_account_currency.trim().is_empty()
        || !(request.adaptive_rr.is_finite() && request.adaptive_rr > 0.0)
        || request.adaptive_tail_step == 0
    {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::InvalidInput,
            "resident population-auto request has an empty or out-of-parent extent",
        ));
    }
    if prepared.pinned_source_projection_v1().identity_sha256()
        != request.financial_source_projection_identity_sha256
    {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
            "prepared resident Data source projection does not match the explicit financial contract",
        ));
    }
    if request.migration_enabled_for_run {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::MigrationNotSealed,
            "resident receipt-governed generation requires run-scoped migration disabled",
        ));
    }
    let evaluation_rows = request.stage1_row_end - request.stage1_row_start;
    let adaptive_resolution = resolve_adaptive_stage_extent_v2(
        request.adaptive_stops_requested_for_run,
        evaluation_rows,
        request.adaptive_tail_max_bars,
    )?;
    let adaptive_request_identity = if adaptive_resolution.effective {
        let (_, adaptive_request) = build_resident_adaptive_stage1_request_v2(
            parent_rows,
            request.stage1_row_start,
            request.stage1_row_end,
            request.adaptive_pip_size,
            request.adaptive_tail_step,
            request.adaptive_tail_max_bars,
        )?;
        adaptive_request.identity_sha256()
    } else {
        [0; 32]
    };
    if request.adaptive_base_effective_for_stage1 != adaptive_resolution.effective
        || request.adaptive_resolution_reason != adaptive_resolution.reason
        || request.max_adaptive_row_count != adaptive_resolution.max_adaptive_row_count
        || request.resident_adaptive_request_identity_sha256 != adaptive_request_identity
    {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::AuthorityMismatch,
            "resident adaptive request drifted from the exact Stage1 policy",
        ));
    }
    let term_cap = feature_count.min(
        request
            .requested_max_indicators
            .max(crate::genetic::seed_templates::PROFESSIONAL_TEMPLATE_MAX_TERMS_V1),
    );
    if term_cap == 0 {
        return Err(ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::InvalidInput,
            "resident term cap resolved to zero",
        ));
    }
    let (raw_time_cap, effective_time_cap, _floor_overrode) =
        crate::gpu_native::prototype_b_population_eval::checked_candidates_for_target_launch_v1(
            evaluation_rows,
        )
        .map_err(|error| map_native_plan_error_v2("resident time plan", error))?;
    let effective_time_cap = usize::try_from(effective_time_cap).map_err(|_| {
        ResidentPopulationAutoSizingErrorV2::new(
            ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
            "effective time cap does not fit this process",
        )
    })?;
    let resolution = resolve_population_auto_extents_v2(
        request.population_auto,
        request.configured_population,
        effective_time_cap,
        hard_growth_cap,
        |candidate_count, scenario_count| {
            workspace_fit_for_extents_v2(
                prepared,
                native_facts,
                &request,
                term_cap,
                candidate_count,
                scenario_count,
            )
        },
    )?;
    let workspace_plan = workspace_plan_attempt_v2(
        prepared,
        native_facts,
        &request,
        term_cap,
        resolution.resolved_population,
        resolution.max_concurrent_scenario_count,
    )?;
    let limits = workspace_plan.limits();
    let resolution_reason = if !request.population_auto {
        "auto_disabled"
    } else if resolution.resolved_population > request.configured_population {
        "resident_cuda_auto_grew"
    } else if request.configured_population > resolution.growth_cap {
        "resident_cuda_configured_above_growth_cap_no_shrink"
    } else {
        "resident_cuda_configured_at_growth_cap"
    };
    let mut receipt = ResidentPopulationAutoSizingReceiptV2 {
        schema_version: RESIDENT_POPULATION_AUTO_SIZING_RECEIPT_SCHEMA_VERSION_V2,
        population_auto: request.population_auto,
        configured_population: checked_u64(request.configured_population, "configured population")?,
        resolved_population: checked_u64(resolution.resolved_population, "resolved population")?,
        resident_parent_rows: extent.row_count(),
        feature_count: extent.column_count(),
        evaluation_rows: checked_u64(evaluation_rows, "evaluation rows")?,
        month_capacity: checked_u64(request.month_capacity, "month capacity")?,
        requested_max_indicators: checked_u64(
            request.requested_max_indicators,
            "requested max indicators",
        )?,
        term_cap: checked_u64(term_cap, "term cap")?,
        stage1_role: request.stage1_role,
        stage1_row_start: checked_u64(request.stage1_row_start, "stage1 row start")?,
        stage1_row_end: checked_u64(request.stage1_row_end, "stage1 row end")?,
        migration_enabled_for_run: false,
        adaptive_stops_requested_for_run: adaptive_resolution.requested,
        adaptive_base_effective_for_stage1: adaptive_resolution.effective,
        adaptive_resolution_reason: adaptive_resolution.reason.to_owned(),
        resident_adaptive_semantic_v1: if adaptive_resolution.effective {
            RESIDENT_ADAPTIVE_BASE_SEMANTIC_V1.to_owned()
        } else {
            String::new()
        },
        stop_target_log_operation_schedule_v3: if adaptive_resolution.effective {
            crate::stop_target::STOP_TARGET_LOG_OPERATION_SCHEDULE_V3.to_owned()
        } else {
            String::new()
        },
        resident_adaptive_request_identity_sha256: adaptive_request_identity,
        adaptive_pip_size_bits: request.adaptive_pip_size.to_bits(),
        pip_value_per_lot_bits: request.pip_value_per_lot.to_bits(),
        financial_authority_identity_sha256: request.financial_authority_identity_sha256,
        financial_input_receipt_sha256: request.financial_input_receipt_sha256,
        financial_source_projection_identity_sha256: request
            .financial_source_projection_identity_sha256,
        evaluation_symbol: request.evaluation_symbol,
        evaluation_account_currency: request.evaluation_account_currency,
        adaptive_rr_bits: request.adaptive_rr.to_bits(),
        adaptive_tail_max_bars: checked_u64(
            request.adaptive_tail_max_bars,
            "adaptive tail bar cap",
        )?,
        adaptive_tail_step: checked_u64(request.adaptive_tail_step, "adaptive tail step")?,
        max_ordered_index_count: checked_u64(request.max_ordered_index_count, "ordered index cap")?,
        max_adaptive_row_count: checked_u64(request.max_adaptive_row_count, "adaptive row cap")?,
        selected_device_ordinal: native_facts.selected_device_ordinal(),
        pre_materialization_free_bytes_snapshot: native_facts
            .pre_materialization_free_bytes_snapshot(),
        allocator_context_reserve_bytes: native_facts.allocator_context_reserve_bytes(),
        allocator_context_reserve_policy: native_facts
            .allocator_context_reserve_policy()
            .to_owned(),
        admission_identity_sha256: hex_lower(native_facts.admission_identity_sha256()),
        native_preflight_facts_identity_sha256: hex_lower(native_facts.facts_identity_sha256()),
        cuda_build_manifest_sha256: hex_lower(native_facts.cuda_build_manifest_sha256()),
        cuda_build_artifact_sha256: hex_lower(native_facts.cuda_build_artifact_sha256()),
        data_peak_device_bytes: workspace_plan.data_peak_device_bytes(),
        data_steady_device_bytes: workspace_plan.data_steady_device_bytes(),
        gene_store_device_bytes: workspace_plan.gene_store_device_bytes(),
        metrics_scenario_device_bytes: workspace_plan.metrics_scenario_device_bytes(),
        max_concurrent_scenario_count: limits.max_concurrent_scenario_count(),
        bounded_host_metric_readback_bytes: workspace_plan.bounded_host_metric_readback_bytes(),
        required_device_bytes_excluding_reserve: workspace_plan
            .required_device_bytes_excluding_reserve(),
        required_device_bytes_including_reserve: workspace_plan
            .required_device_bytes_including_reserve(),
        raw_time_cap,
        effective_time_cap: u64::try_from(effective_time_cap).map_err(|_| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                "effective time cap does not fit u64",
            )
        })?,
        hard_growth_cap: checked_u64(hard_growth_cap, "hard growth cap")?,
        memory_one_launch_population_cap: checked_u64(
            resolution.memory_one_launch_population_cap,
            "memory one-launch cap",
        )?,
        growth_cap: checked_u64(resolution.growth_cap, "growth cap")?,
        resolution_reason: resolution_reason.to_owned(),
        workspace_plan_identity_sha256: hex_lower(limits.workspace_plan_identity_sha256()),
        population_sizing_authority_sha256: hex_lower(limits.population_sizing_authority_sha256()),
        data_extent_identity_sha256: hex_lower(limits.data_extent_identity_sha256()),
        identity_sha256: String::new(),
    };
    receipt.identity_sha256 = receipt.computed_identity_sha256()?;
    receipt.validate_self_v2()?;
    receipt.validate_against_workspace_plan_v2(native_facts, &workspace_plan)?;
    Ok((receipt, workspace_plan))
}

/// Search-owned canonical-trendbar research entrypoint. The validated
/// financial contract is explicit because sizing precedes installation of the
/// run-scoped research execution guard. General broker/TUI Discovery has no
/// sealed scalar value receipt yet and must not call this route.
pub fn seal_resident_population_auto_for_canonical_trendbar_research_v2(
    prepared: &PreparedGpuOnlyFeatureMaterializationV3,
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
    config: &crate::DiscoveryConfig,
    financial_contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
) -> Result<
    (
        ResidentPopulationAutoSizingReceiptV2,
        SealedDataPopulationGpuWorkspacePlanV1,
    ),
    ResidentPopulationAutoSizingErrorV2,
> {
    seal_resident_population_auto_for_canonical_trendbar_research_with_hard_cap_impl_v2(
        prepared,
        native_facts,
        config,
        financial_contract,
        RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2,
        false,
    )
}

pub(crate) fn seal_resident_population_auto_for_canonical_trendbar_research_with_hard_cap_v2(
    prepared: &PreparedGpuOnlyFeatureMaterializationV3,
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
    config: &crate::DiscoveryConfig,
    financial_contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    external_hard_population_cap: usize,
) -> Result<
    (
        ResidentPopulationAutoSizingReceiptV2,
        SealedDataPopulationGpuWorkspacePlanV1,
    ),
    ResidentPopulationAutoSizingErrorV2,
> {
    seal_resident_population_auto_for_canonical_trendbar_research_with_hard_cap_impl_v2(
        prepared,
        native_facts,
        config,
        financial_contract,
        external_hard_population_cap,
        true,
    )
}

fn seal_resident_population_auto_for_canonical_trendbar_research_with_hard_cap_impl_v2(
    prepared: &PreparedGpuOnlyFeatureMaterializationV3,
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
    config: &crate::DiscoveryConfig,
    financial_contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    external_hard_population_cap: usize,
    enforce_configured_population_cap: bool,
) -> Result<
    (
        ResidentPopulationAutoSizingReceiptV2,
        SealedDataPopulationGpuWorkspacePlanV1,
    ),
    ResidentPopulationAutoSizingErrorV2,
> {
    checked_effective_hard_growth_cap_v2(external_hard_population_cap)?;
    if enforce_configured_population_cap {
        checked_configured_population_against_external_hard_cap_v2(
            config.population,
            external_hard_population_cap,
        )?;
    }
    let resident_parent_rows =
        usize::try_from(prepared.workspace_extent().row_count()).map_err(|_| {
            ResidentPopulationAutoSizingErrorV2::new(
                ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                "resident parent rows do not fit this process",
            )
        })?;
    let request = ResidentPopulationAutoSizingRequestV2::from_discovery_config_v2(
        config,
        resident_parent_rows,
        financial_contract,
    )?;
    seal_resident_population_auto_sizing_receipt_with_hard_cap_v2(
        prepared,
        native_facts,
        request,
        external_hard_population_cap,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refresh_fixture_workspace_bytes_v2(receipt: &mut ResidentPopulationAutoSizingReceiptV2) {
        let (gene, metrics, required, including) = receipt
            .internal_workspace_bytes_v2(
                receipt.resolved_population as usize,
                receipt.max_concurrent_scenario_count as usize,
                receipt.term_cap as usize,
                receipt.month_capacity as u32,
            )
            .unwrap();
        receipt.gene_store_device_bytes = gene.total_device_bytes();
        receipt.metrics_scenario_device_bytes = metrics.total_device_bytes();
        receipt.bounded_host_metric_readback_bytes = metrics.metric_rows_bytes();
        receipt.required_device_bytes_excluding_reserve = required;
        receipt.required_device_bytes_including_reserve = including;
    }

    fn self_validating_receipt_fixture_v2() -> ResidentPopulationAutoSizingReceiptV2 {
        let mut receipt = ResidentPopulationAutoSizingReceiptV2 {
            schema_version: RESIDENT_POPULATION_AUTO_SIZING_RECEIPT_SCHEMA_VERSION_V2,
            population_auto: false,
            configured_population: 10,
            resolved_population: 10,
            resident_parent_rows: 1_000,
            feature_count: 5,
            evaluation_rows: 500,
            month_capacity: 12,
            requested_max_indicators: 5,
            term_cap: 5,
            stage1_role: "selection_stage1".to_owned(),
            stage1_row_start: 500,
            stage1_row_end: 1_000,
            migration_enabled_for_run: false,
            adaptive_stops_requested_for_run: false,
            adaptive_base_effective_for_stage1: false,
            adaptive_resolution_reason: ADAPTIVE_RESOLUTION_DISABLED_V2.to_owned(),
            resident_adaptive_semantic_v1: String::new(),
            stop_target_log_operation_schedule_v3: String::new(),
            resident_adaptive_request_identity_sha256: [0; 32],
            adaptive_pip_size_bits: 0.0001_f64.to_bits(),
            pip_value_per_lot_bits: 10.0_f64.to_bits(),
            financial_authority_identity_sha256: "1".repeat(64),
            financial_input_receipt_sha256: "2".repeat(64),
            financial_source_projection_identity_sha256: [3; 32],
            evaluation_symbol: "EURUSD".to_owned(),
            evaluation_account_currency: "USD".to_owned(),
            adaptive_rr_bits: 2.0_f64.to_bits(),
            adaptive_tail_max_bars: 0,
            adaptive_tail_step: 1,
            max_ordered_index_count: 0,
            max_adaptive_row_count: 0,
            selected_device_ordinal: 0,
            pre_materialization_free_bytes_snapshot: 8_000_000_000,
            allocator_context_reserve_bytes:
                neoethos_gpu_cuda::DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1,
            allocator_context_reserve_policy:
                neoethos_gpu_cuda::DATA_POPULATION_ALLOCATOR_RESERVE_POLICY_V1.to_owned(),
            admission_identity_sha256: "4".repeat(64),
            native_preflight_facts_identity_sha256: "5".repeat(64),
            cuda_build_manifest_sha256: "6".repeat(64),
            cuda_build_artifact_sha256: "7".repeat(64),
            data_peak_device_bytes: 1_000,
            data_steady_device_bytes: 900,
            gene_store_device_bytes: 0,
            metrics_scenario_device_bytes: 0,
            max_concurrent_scenario_count: 10,
            bounded_host_metric_readback_bytes: 0,
            required_device_bytes_excluding_reserve: 0,
            required_device_bytes_including_reserve: 0,
            raw_time_cap: 0,
            effective_time_cap: 0,
            hard_growth_cap: RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2 as u64,
            memory_one_launch_population_cap: 0,
            growth_cap: 0,
            resolution_reason: "auto_disabled".to_owned(),
            workspace_plan_identity_sha256: "8".repeat(64),
            population_sizing_authority_sha256: "9".repeat(64),
            data_extent_identity_sha256: "a".repeat(64),
            identity_sha256: String::new(),
        };
        let (raw_time_cap, effective_time_cap, _) =
            crate::gpu_native::prototype_b_population_eval::checked_candidates_for_target_launch_v1(
                receipt.evaluation_rows as usize,
            )
            .unwrap();
        receipt.raw_time_cap = raw_time_cap;
        receipt.effective_time_cap = effective_time_cap;
        refresh_fixture_workspace_bytes_v2(&mut receipt);
        let resolution = resolve_population_auto_extents_v2(
            receipt.population_auto,
            receipt.configured_population as usize,
            receipt.effective_time_cap as usize,
            receipt.hard_growth_cap as usize,
            |candidate_count, scenario_count| {
                let (_, _, _, required) = receipt
                    .internal_workspace_bytes_v2(
                        candidate_count,
                        scenario_count,
                        receipt.term_cap as usize,
                        receipt.month_capacity as u32,
                    )
                    .unwrap();
                let fits = required <= receipt.pre_materialization_free_bytes_snapshot;
                Ok(if fits {
                    WorkspaceFitV2::Fits
                } else {
                    WorkspaceFitV2::NoRoom
                })
            },
        )
        .unwrap();
        receipt.memory_one_launch_population_cap =
            resolution.memory_one_launch_population_cap as u64;
        receipt.growth_cap = resolution.growth_cap as u64;
        receipt.identity_sha256 = receipt.computed_identity_sha256().unwrap();
        receipt
    }

    fn recompute_fixture_identity_v2(receipt: &mut ResidentPopulationAutoSizingReceiptV2) {
        receipt.identity_sha256 = receipt.computed_identity_sha256().unwrap();
    }

    #[test]
    fn self_validation_accepts_an_existing_valid_receipt() {
        let mut receipt = self_validating_receipt_fixture_v2();
        assert_eq!(receipt.raw_time_cap, 33_720_000);
        assert_eq!(receipt.effective_time_cap, 33_720_000);
        assert_eq!(receipt.gene_store_device_bytes, 1_322);
        assert_eq!(receipt.metrics_scenario_device_bytes, 3_520);
        assert_eq!(receipt.required_device_bytes_excluding_reserve, 6_742);
        assert_eq!(receipt.required_device_bytes_including_reserve, 67_115_606);
        assert_eq!(receipt.memory_one_launch_population_cap, 16_384);
        assert_eq!(receipt.growth_cap, 16_384);
        receipt
            .validate_self_v2()
            .expect("valid algebraic V2 receipt");
        receipt.hard_growth_cap -= 1;
        receipt.memory_one_launch_population_cap -= 1;
        receipt.growth_cap -= 1;
        recompute_fixture_identity_v2(&mut receipt);
        receipt.validate_self_v2().expect("coherent lower hard cap");
    }

    #[test]
    fn self_validation_rejects_mutated_size_facts_even_after_identity_rehash() {
        for mutate in [
            |receipt: &mut ResidentPopulationAutoSizingReceiptV2| receipt.feature_count = 0,
            |receipt: &mut ResidentPopulationAutoSizingReceiptV2| receipt.hard_growth_cap = 0,
            |receipt: &mut ResidentPopulationAutoSizingReceiptV2| {
                receipt.month_capacity = u64::from(u32::MAX) + 1
            },
            |receipt: &mut ResidentPopulationAutoSizingReceiptV2| {
                receipt.bounded_host_metric_readback_bytes -= 1
            },
        ] {
            let mut receipt = self_validating_receipt_fixture_v2();
            mutate(&mut receipt);
            recompute_fixture_identity_v2(&mut receipt);
            assert!(receipt.validate_self_v2().is_err());
        }
    }

    #[rustfmt::skip]
    #[test]
    fn self_validation_rejects_correlated_internal_mutations_after_identity_rehash() {
        let mut accepted = Vec::new();
        macro_rules! reject_if_accepted {
            ($name:literal, |$receipt:ident| $($mutation:stmt);+ $(;)?) => {{
            let mut $receipt = self_validating_receipt_fixture_v2();
            $($mutation;)+
            recompute_fixture_identity_v2(&mut $receipt);
            if $receipt.validate_self_v2().is_ok() {
                accepted.push($name);
            }
            }};
        }
        reject_if_accepted!("term cap", |receipt| receipt.requested_max_indicators = 4; receipt.term_cap = 4);
        reject_if_accepted!("raw time cap plus one", |receipt| receipt.raw_time_cap += 1);
        reject_if_accepted!("raw time cap minus one", |receipt| receipt.raw_time_cap -= 1);
        reject_if_accepted!("effective time cap plus one", |receipt| receipt.effective_time_cap += 1);
        reject_if_accepted!("effective time cap minus one", |receipt| receipt.effective_time_cap -= 1);
        reject_if_accepted!("stage role", |receipt| receipt.stage1_role = "selection_stage2".to_owned());
        reject_if_accepted!("free memory admission", |receipt| receipt.pre_materialization_free_bytes_snapshot = receipt.required_device_bytes_including_reserve - 1);
        reject_if_accepted!("positive data bytes", |receipt| receipt.data_peak_device_bytes = 0; receipt.data_steady_device_bytes = 0);
        reject_if_accepted!("gene bytes plus one", |receipt| receipt.gene_store_device_bytes += 1);
        reject_if_accepted!("gene bytes minus one", |receipt| receipt.gene_store_device_bytes -= 1);
        reject_if_accepted!("metrics bytes plus one", |receipt| receipt.metrics_scenario_device_bytes += 1);
        reject_if_accepted!("metrics bytes minus one", |receipt| receipt.metrics_scenario_device_bytes -= 1);
        reject_if_accepted!("required bytes plus one", |receipt| receipt.required_device_bytes_excluding_reserve += 1; receipt.required_device_bytes_including_reserve += 1);
        reject_if_accepted!("required bytes minus one", |receipt| receipt.required_device_bytes_excluding_reserve -= 1; receipt.required_device_bytes_including_reserve -= 1);
        reject_if_accepted!("reserve and dependent total", |receipt| receipt.allocator_context_reserve_bytes += 1; receipt.required_device_bytes_including_reserve += 1);
        reject_if_accepted!("reserve policy", |receipt| receipt.allocator_context_reserve_policy.push_str("-drift"));
        reject_if_accepted!("peak below steady", |receipt| receipt.data_peak_device_bytes = receipt.data_steady_device_bytes - 1);
        reject_if_accepted!("memory and growth caps", |receipt| receipt.memory_one_launch_population_cap -= 1; receipt.growth_cap -= 1);
        reject_if_accepted!("scenario and dependent bytes", |receipt| receipt.max_concurrent_scenario_count -= 1; refresh_fixture_workspace_bytes_v2(&mut receipt));
        reject_if_accepted!("hard cap above ceiling", |receipt| receipt.hard_growth_cap = 16_385);
        assert!(accepted.is_empty(), "self-validation accepted correlated mutations: {accepted:?}");
    }

    #[test]
    fn self_validation_delegates_metric_row_width_to_the_native_plan() {
        let source = include_str!("resident_population_auto_sizing_receipt_v2.rs");
        let method = source
            .split_once("fn internal_workspace_bytes_v2")
            .expect("self validator source")
            .1
            .split_once("fn computed_identity_sha256")
            .expect("self validator end")
            .0;
        assert!(method.contains("PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1"));
        assert!(method.contains("u32::try_from(self.month_capacity)"));
        for forbidden in ["RESIDENT_METRIC_ROW_BYTES_V1", "checked_mul(104)", "11 * 8"] {
            assert!(
                !method.contains(forbidden),
                "self validation must not redeclare metric-row authority as {forbidden}"
            );
        }
    }

    #[test]
    fn self_validation_rejects_identity_mutation_without_deserializing_json() {
        let mut receipt = self_validating_receipt_fixture_v2();
        receipt.identity_sha256.replace_range(0..1, "f");
        assert!(receipt.validate_self_v2().is_err());
    }

    fn linear_fit(
        budget: usize,
    ) -> impl FnMut(usize, usize) -> Result<WorkspaceFitV2, ResidentPopulationAutoSizingErrorV2>
    {
        move |population, scenarios| {
            let required = population.checked_add(scenarios).ok_or_else(|| {
                ResidentPopulationAutoSizingErrorV2::new(
                    ResidentPopulationAutoSizingErrorCodeV2::ArithmeticOverflow,
                    "fixture extent overflow",
                )
            })?;
            Ok(if required <= budget {
                WorkspaceFitV2::Fits
            } else {
                WorkspaceFitV2::NoRoom
            })
        }
    }

    #[test]
    fn auto_grows_to_the_one_launch_memory_cap() {
        let resolved =
            resolve_population_auto_extents_v2(true, 200, 16_384, 16_384, linear_fit(10_000))
                .expect("auto resolution");
        assert_eq!(resolved.resolved_population, 5_000);
        assert_eq!(resolved.max_concurrent_scenario_count, 5_000);
        assert_eq!(resolved.memory_one_launch_population_cap, 5_000);
    }

    #[test]
    fn configured_population_never_shrinks_and_scenarios_split() {
        let resolved =
            resolve_population_auto_extents_v2(true, 8_000, 16_384, 16_384, linear_fit(9_000))
                .expect("no-shrink resolution");
        assert_eq!(resolved.resolved_population, 8_000);
        assert_eq!(resolved.memory_one_launch_population_cap, 4_500);
        assert_eq!(resolved.max_concurrent_scenario_count, 1_000);
    }

    #[test]
    fn auto_disabled_keeps_configured_population_but_still_admits_memory() {
        let resolved =
            resolve_population_auto_extents_v2(false, 200, 16_384, 16_384, linear_fit(10_000))
                .expect("disabled-auto admission");
        assert_eq!(resolved.resolved_population, 200);
        assert_eq!(resolved.max_concurrent_scenario_count, 200);
    }

    #[test]
    fn configured_gene_store_plus_one_scenario_fails_loud() {
        let error =
            resolve_population_auto_extents_v2(true, 8_000, 16_384, 16_384, linear_fit(8_000))
                .expect_err("one scenario must not fit");
        assert_eq!(
            error.code(),
            ResidentPopulationAutoSizingErrorCodeV2::ConfiguredGeneNoRoom
        );
    }

    #[test]
    fn adaptive_stage_policy_matches_cpu_too_short_and_tail_cap_ordering() {
        let disabled =
            resolve_adaptive_stage_extent_v2(false, 10_000, 1).expect("disabled adaptive policy");
        assert_eq!(disabled.reason, ADAPTIVE_RESOLUTION_DISABLED_V2);
        assert!(!disabled.effective);
        assert_eq!(disabled.max_adaptive_row_count, 0);

        let r100 = resolve_adaptive_stage_extent_v2(true, 100, 50)
            .expect("TooShort takes precedence over the tail cap");
        assert_eq!(r100.reason, ADAPTIVE_RESOLUTION_FIXED_TOO_SHORT_V2);
        assert!(r100.requested);
        assert!(!r100.effective);
        assert_eq!(r100.max_adaptive_row_count, 0);

        let r101 = resolve_adaptive_stage_extent_v2(true, 101, 0)
            .expect("the canonical minimum row count is resident-effective");
        assert_eq!(r101.reason, ADAPTIVE_RESOLUTION_RESIDENT_EXACT_V1);
        assert!(r101.effective);
        assert_eq!(r101.max_adaptive_row_count, 101);

        let error = resolve_adaptive_stage_extent_v2(true, 101, 100).expect_err(
            "an effective adaptive view above the tail cap must fail before allocation",
        );
        assert_eq!(
            error.code(),
            ResidentPopulationAutoSizingErrorCodeV2::AdaptiveTailCapExceeded
        );
    }

    #[test]
    fn adaptive_request_identity_binds_view_pip_and_runtime_semantics() {
        let request = |start, end, pip, step, cap| {
            build_resident_adaptive_stage1_request_v2(256, start, end, pip, step, cap)
                .expect("canonical resident adaptive request")
                .1
                .identity_sha256()
        };
        let canonical = request(32, 192, 0.0001, 1, 0);
        assert_ne!(canonical, request(33, 193, 0.0001, 1, 0));
        assert_ne!(canonical, request(32, 192, 0.01, 1, 0));
        assert_ne!(canonical, request(32, 192, 0.0001, 2, 0));
        assert_ne!(canonical, request(32, 192, 0.0001, 1, 256));
        assert!(RESIDENT_ADAPTIVE_BASE_SEMANTIC_V1.contains("cpu-cuda-bit-tolerance=zero"));
        assert_eq!(
            crate::stop_target::STOP_TARGET_LOG_OPERATION_SCHEDULE_V3,
            neoethos_data::QUANT_LOG_OPERATION_SCHEDULE_V3
        );
    }
}
