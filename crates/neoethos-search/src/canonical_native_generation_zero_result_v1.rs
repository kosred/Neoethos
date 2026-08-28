//! Checked pre-V5 size planning for the canonical native Generation-zero result.
//!
//! Prepared Data supplies exact `F`; this module derives pre-V5 persistence
//! admission and seals the bounded, borrow-only Generation-zero artifact.

#![cfg_attr(
    any(not(test), not(feature = "gpu-cuda")),
    expect(
        dead_code,
        reason = "the Chunk 3 executor is the first production consumer of this sealed 2A2 API"
    )
)]

use std::fmt;
use std::io::{self, Write};

use crate::canonical_native_discovery_request_v1::{
    MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1,
    MAX_CANONICAL_NATIVE_GEN0_RESOLVED_POPULATION_V1, MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
    MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1, MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1,
    MAX_CANONICAL_NATIVE_GEN0_TERMS_V1, MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1,
};
use serde::Serialize;
#[cfg(feature = "gpu-cuda")]
use serde::ser::SerializeSeq;

#[cfg(feature = "gpu-cuda")]
use crate::canonical_native_discovery_request_v1::{
    CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1,
    CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1, CanonicalNativeCostBandStatusV1,
    CanonicalNativeDiscoveryRequestV1, CanonicalNativeExecutionScopeV1,
};
#[cfg(feature = "gpu-cuda")]
use crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3;
#[cfg(feature = "gpu-cuda")]
use crate::data_selection::CanonicalGpuResidentSearchInputReceiptV3;
#[cfg(feature = "gpu-cuda")]
use crate::genetic::{EvaluationConfig, Gene};
#[cfg(feature = "gpu-cuda")]
use crate::historical_research::{
    HistoricalResearchArtifactClassV1, HistoricalResearchPromotionEligibilityV1,
};
#[cfg(feature = "gpu-cuda")]
use crate::prepared_discovery_run_input_v3::ResidentGenerationZeroMilestoneV1;
#[cfg(feature = "gpu-cuda")]
use crate::resident_population_auto_sizing_receipt_v2::{
    RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2, ResidentPopulationAutoSizingReceiptV2,
};
#[cfg(feature = "gpu-cuda")]
use neoethos_gpu_cuda::PopulationMetricsOnlyPlanV1;
#[cfg(feature = "gpu-cuda")]
use sha2::{Digest, Sha256};

pub const CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1: &str =
    "neoethos.canonical-native-generation-zero-research-result.v1";
pub const CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1: u16 = 1;

const MIN_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1: usize = 10;
const EMPTY_POPULATION_ARRAY_REPLACEMENT_BYTES_V1: u64 = 3;
const GENE_JSON_UPPER_BOUND_BASE_BYTES_V1: u64 = 1_097;
const GENE_JSON_UPPER_BOUND_PER_TERM_BYTES_V1: u64 = 46;
const METRIC_ROW_JSON_UPPER_BOUND_BYTES_V1: u64 = 276;
const METRIC_RECEIPT_LOWER_HEX_STRING_UPPER_BOUND_BYTES_V1: u64 = 66;
const PER_POPULATION_JSON_UPPER_BOUND_BASE_BYTES_V1: u64 = 1_442;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1 {
    InvalidInput,
    ArithmeticOverflow,
    PopulationCapacityZero,
    ConfiguredPopulationExceedsCapacity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalNativeGenerationZeroResultSizePlanErrorV1 {
    code: CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1,
    detail: String,
}

impl CanonicalNativeGenerationZeroResultSizePlanErrorV1 {
    fn new(
        code: CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub(crate) const fn code(&self) -> CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1 {
        self.code
    }
}

impl fmt::Display for CanonicalNativeGenerationZeroResultSizePlanErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "canonical native Generation-zero result size planning failed ({:?}): {}",
            self.code, self.detail
        )
    }
}

impl std::error::Error for CanonicalNativeGenerationZeroResultSizePlanErrorV1 {}

/// A pure pre-V5 envelope calculation. It has no CUDA or population-sizing
/// receipt input: exact prepared feature count plus bounded scalar facts are
/// sufficient to derive the persistable population ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalNativeGenerationZeroResultSizePlanV1 {
    prepared_feature_count: usize,
    raw_configured_max_indicators: usize,
    resolved_max_indicators: usize,
    term_cap: usize,
    configured_population: usize,
    fixed_metadata_upper_bound_with_empty_arrays_bytes: u64,
    fixed_metadata_without_empty_array_closers_bytes: u64,
    per_population_upper_bound_bytes: u64,
    population_cap: usize,
    configured_population_upper_bound_bytes: u64,
}

impl CanonicalNativeGenerationZeroResultSizePlanV1 {
    /// Construct from exact prepared `F` and a fixed-metadata bound produced by
    /// the future final-result sealer. The fixed bound must already include
    /// three empty population arrays (`[]`); this planner does not manufacture
    /// or estimate those final receipt/metadata bytes.
    pub(crate) fn checked_new(
        prepared_feature_count: usize,
        raw_configured_max_indicators: usize,
        configured_population: usize,
        fixed_metadata_upper_bound_with_empty_arrays_bytes: u64,
    ) -> Result<Self, CanonicalNativeGenerationZeroResultSizePlanErrorV1> {
        if prepared_feature_count == 0
            || prepared_feature_count > MAX_CANONICAL_NATIVE_GEN0_TERMS_V1
            || raw_configured_max_indicators > MAX_CANONICAL_NATIVE_GEN0_TERMS_V1
        {
            return Err(CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput,
                format!(
                    "prepared feature count and raw maximum indicators must be within 1..={MAX_CANONICAL_NATIVE_GEN0_TERMS_V1} (raw zero is the only sentinel)"
                ),
            ));
        }
        if !(MIN_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1
            ..=MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1)
            .contains(&configured_population)
        {
            return Err(CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput,
                format!(
                    "configured population must be within {MIN_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1}..={MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1}"
                ),
            ));
        }
        if fixed_metadata_upper_bound_with_empty_arrays_bytes
            > MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1
        {
            return Err(CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput,
                "fixed metadata upper bound exceeds the frozen 512 MiB result envelope",
            ));
        }

        let fixed_metadata_without_empty_array_closers_bytes =
            fixed_metadata_upper_bound_with_empty_arrays_bytes
                .checked_sub(EMPTY_POPULATION_ARRAY_REPLACEMENT_BYTES_V1)
                .ok_or_else(|| {
                    CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                        CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                        "fixed metadata upper bound cannot remove the three empty-array closers",
                    )
                })?;
        let resolved_max_indicators = if raw_configured_max_indicators == 0 {
            prepared_feature_count
        } else {
            raw_configured_max_indicators
        };
        let term_cap = prepared_feature_count.min(
            resolved_max_indicators
                .max(crate::genetic::seed_templates::PROFESSIONAL_TEMPLATE_MAX_TERMS_V1),
        );
        if term_cap == 0 {
            return Err(CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput,
                "resolved Generation-zero term cap is zero",
            ));
        }

        let term_cap_bytes = u64::try_from(term_cap).map_err(|_| {
            CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                "resolved term cap does not fit the result byte planner",
            )
        })?;
        let variable_gene_bytes = GENE_JSON_UPPER_BOUND_PER_TERM_BYTES_V1
            .checked_mul(term_cap_bytes)
            .ok_or_else(|| {
                CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                    CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                    "per-term gene upper bound overflowed",
                )
            })?;
        let gene_upper_bound_bytes = GENE_JSON_UPPER_BOUND_BASE_BYTES_V1
            .checked_add(variable_gene_bytes)
            .ok_or_else(|| {
                CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                    CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                    "gene upper bound overflowed",
                )
            })?;
        let per_population_upper_bound_bytes = PER_POPULATION_JSON_UPPER_BOUND_BASE_BYTES_V1
            .checked_add(variable_gene_bytes)
            .ok_or_else(|| {
                CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                    CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                    "per-population result upper bound overflowed",
                )
            })?;
        let independently_composed_per_population = gene_upper_bound_bytes
            .checked_add(METRIC_ROW_JSON_UPPER_BOUND_BYTES_V1)
            .and_then(|bytes| {
                bytes.checked_add(METRIC_RECEIPT_LOWER_HEX_STRING_UPPER_BOUND_BYTES_V1)
            })
            .and_then(|bytes| bytes.checked_add(3))
            .ok_or_else(|| {
                CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                    CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                    "independent per-population composition overflowed",
                )
            })?;
        if independently_composed_per_population != per_population_upper_bound_bytes {
            return Err(CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput,
                "frozen per-population schema constants are internally inconsistent",
            ));
        }

        let remaining_bytes = MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1
            .checked_sub(fixed_metadata_without_empty_array_closers_bytes)
            .ok_or_else(|| {
                CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                    CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                    "fixed metadata subtraction overflowed the result envelope",
                )
            })?;
        let population_cap_u64 = (remaining_bytes / per_population_upper_bound_bytes).min(
            u64::try_from(MAX_CANONICAL_NATIVE_GEN0_RESOLVED_POPULATION_V1).map_err(|_| {
                CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                    CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                    "named population ceiling does not fit the byte planner",
                )
            })?,
        );
        let population_cap = usize::try_from(population_cap_u64).map_err(|_| {
            CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                "derived population capacity does not fit this process",
            )
        })?;
        if population_cap == 0 {
            return Err(CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::PopulationCapacityZero,
                "fixed metadata leaves no room for one complete population entry",
            ));
        }
        if configured_population > population_cap {
            return Err(CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ConfiguredPopulationExceedsCapacity,
                format!(
                    "configured population {configured_population} exceeds persistable capacity {population_cap}"
                ),
            ));
        }
        let configured_population_upper_bound_bytes = checked_upper_bound_v1(
            fixed_metadata_without_empty_array_closers_bytes,
            per_population_upper_bound_bytes,
            configured_population,
        )?;

        Ok(Self {
            prepared_feature_count,
            raw_configured_max_indicators,
            resolved_max_indicators,
            term_cap,
            configured_population,
            fixed_metadata_upper_bound_with_empty_arrays_bytes,
            fixed_metadata_without_empty_array_closers_bytes,
            per_population_upper_bound_bytes,
            population_cap,
            configured_population_upper_bound_bytes,
        })
    }

    pub(crate) const fn prepared_feature_count(&self) -> usize {
        self.prepared_feature_count
    }

    pub(crate) const fn raw_configured_max_indicators(&self) -> usize {
        self.raw_configured_max_indicators
    }

    pub(crate) const fn resolved_max_indicators(&self) -> usize {
        self.resolved_max_indicators
    }

    pub(crate) const fn term_cap(&self) -> usize {
        self.term_cap
    }

    pub(crate) const fn configured_population(&self) -> usize {
        self.configured_population
    }

    pub(crate) const fn fixed_metadata_upper_bound_with_empty_arrays_bytes(&self) -> u64 {
        self.fixed_metadata_upper_bound_with_empty_arrays_bytes
    }

    pub(crate) const fn fixed_metadata_without_empty_array_closers_bytes(&self) -> u64 {
        self.fixed_metadata_without_empty_array_closers_bytes
    }

    pub(crate) const fn per_population_upper_bound_bytes(&self) -> u64 {
        self.per_population_upper_bound_bytes
    }

    pub(crate) const fn population_cap(&self) -> usize {
        self.population_cap
    }

    pub(crate) const fn configured_population_upper_bound_bytes(&self) -> u64 {
        self.configured_population_upper_bound_bytes
    }

    pub(crate) fn checked_upper_bound_for_population(
        &self,
        population: usize,
    ) -> Result<u64, CanonicalNativeGenerationZeroResultSizePlanErrorV1> {
        let upper_bound = checked_upper_bound_v1(
            self.fixed_metadata_without_empty_array_closers_bytes,
            self.per_population_upper_bound_bytes,
            population,
        )?;
        if population > self.population_cap
            || upper_bound > MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1
        {
            return Err(CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ConfiguredPopulationExceedsCapacity,
                format!(
                    "result population {population} exceeds persistable capacity {}",
                    self.population_cap
                ),
            ));
        }
        Ok(upper_bound)
    }
}

fn checked_upper_bound_v1(
    fixed_metadata_without_empty_array_closers_bytes: u64,
    per_population_upper_bound_bytes: u64,
    population: usize,
) -> Result<u64, CanonicalNativeGenerationZeroResultSizePlanErrorV1> {
    if population == 0 {
        return Err(CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
            CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput,
            "result population must be non-zero",
        ));
    }
    let population = u64::try_from(population).map_err(|_| {
        CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
            CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
            "population does not fit the result byte planner",
        )
    })?;
    let population_bytes = population
        .checked_mul(per_population_upper_bound_bytes)
        .ok_or_else(|| {
            CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                "population multiplied by its per-entry upper bound overflowed",
            )
        })?;
    fixed_metadata_without_empty_array_closers_bytes
        .checked_add(population_bytes)
        .ok_or_else(|| {
            CanonicalNativeGenerationZeroResultSizePlanErrorV1::new(
                CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow,
                "fixed metadata plus population upper bound overflowed",
            )
        })
}
// END CANONICAL_NATIVE_GENERATION_ZERO_SIZE_PLANNER_V1

// BEGIN CANONICAL_NATIVE_GEN0_PREFLIGHT_V1
// The fixed envelope includes exactly three arrays; the planner replaces their
// closers through EMPTY_POPULATION_ARRAY_REPLACEMENT_BYTES_V1 before charging P.
const MAX_FINITE_F64_JSON_BYTES_V1: u64 = 24;
const MAX_RESULT_STRING_JSON_CONTENT_BYTES_V1: u64 =
    (MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1 as u64) * 6;
const COST_BAND_OPTION_JSON_UPPER_BOUND_BYTES_V1: u64 = 51;
const ADAPTIVE_TOKEN_OPTION_JSON_UPPER_BOUND_BYTES_V1: u64 = 66;
const EVIDENCE_IDENTITY_JSON_STRING_BYTES_V1: u64 = 66;
const EMPTY_POPULATION_ARRAYS_COMPACT_JSON_BYTES_V1: u64 = 6;
const RESIDENT_POPULATION_SIZING_RECEIPT_V2_FIXED_JSON_BYTES_V1: u64 = 2_616;
const RESIDENT_POPULATION_SIZING_RECEIPT_V2_JSON_UPPER_BOUND_BYTES_V1: u64 =
    RESIDENT_POPULATION_SIZING_RECEIPT_V2_FIXED_JSON_BYTES_V1
        + 18 * MAX_RESULT_STRING_JSON_CONTENT_BYTES_V1;
const NATIVE_V3_FIXED_JSON_UPPER_BOUND_BYTES_V1: u64 = 393_995;
const NATIVE_V3_SOURCE_BINDING_JSON_UPPER_BOUND_BYTES_V1: u64 = 1_966_378;
const NATIVE_V3_SOURCE_SEGMENT_JSON_UPPER_BOUND_BYTES_V1: u64 = 148;
const GROUPED_FIXED_METADATA_STATIC_JSON_BYTES_V1: u64 = 791_605;
const GROUPED_FIXED_METADATA_BASE_WITH_V2_V3_JSON_BYTES_V1: u64 =
    GROUPED_FIXED_METADATA_STATIC_JSON_BYTES_V1
        + RESIDENT_POPULATION_SIZING_RECEIPT_V2_JSON_UPPER_BOUND_BYTES_V1
        + NATIVE_V3_FIXED_JSON_UPPER_BOUND_BYTES_V1;

#[derive(Debug)]
pub(crate) enum CanonicalNativeGenerationZeroResultErrorV1 {
    InvalidEvidence(String),
    ArithmeticOverflow(&'static str),
    Serialization(String),
    Io(io::Error),
}

impl fmt::Display for CanonicalNativeGenerationZeroResultErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEvidence(detail) => {
                write!(formatter, "invalid Generation-zero evidence: {detail}")
            }
            Self::ArithmeticOverflow(detail) => {
                write!(
                    formatter,
                    "Generation-zero result arithmetic overflow: {detail}"
                )
            }
            Self::Serialization(detail) => {
                write!(
                    formatter,
                    "Generation-zero result serialization failed: {detail}"
                )
            }
            Self::Io(error) => write!(formatter, "Generation-zero result write failed: {error}"),
        }
    }
}

impl std::error::Error for CanonicalNativeGenerationZeroResultErrorV1 {}

impl From<CanonicalNativeGenerationZeroResultSizePlanErrorV1>
    for CanonicalNativeGenerationZeroResultErrorV1
{
    fn from(error: CanonicalNativeGenerationZeroResultSizePlanErrorV1) -> Self {
        Self::InvalidEvidence(format!("{error}"))
    }
}

fn invalid_result_v1(detail: impl Into<String>) -> CanonicalNativeGenerationZeroResultErrorV1 {
    CanonicalNativeGenerationZeroResultErrorV1::InvalidEvidence(detail.into())
}

#[derive(Default)]
struct CompactJsonCountingWriterV1 {
    byte_count: u64,
}

impl Write for CompactJsonCountingWriterV1 {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.byte_count = self
            .byte_count
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("compact JSON byte count overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn checked_compact_json_byte_count_v1<T: Serialize + ?Sized>(
    value: &T,
) -> Result<u64, CanonicalNativeGenerationZeroResultErrorV1> {
    let mut writer = CompactJsonCountingWriterV1::default();
    value
        .serialize(&mut serde_json::Serializer::new(&mut writer))
        .map_err(|error| {
            CanonicalNativeGenerationZeroResultErrorV1::Serialization(format!("{error}"))
        })?;
    Ok(writer.byte_count)
}

fn checked_compact_json_string_byte_count_v1(
    value: &str,
) -> Result<u64, CanonicalNativeGenerationZeroResultErrorV1> {
    checked_compact_json_byte_count_v1(value)
}

fn checked_gene_json_upper_bound_bytes_v1(
    term_cap: usize,
) -> Result<u64, CanonicalNativeGenerationZeroResultErrorV1> {
    let terms = u64::try_from(term_cap)
        .map_err(|_| CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow("term cap"))?;
    GENE_JSON_UPPER_BOUND_PER_TERM_BYTES_V1
        .checked_mul(terms)
        .and_then(|bytes| GENE_JSON_UPPER_BOUND_BASE_BYTES_V1.checked_add(bytes))
        .ok_or(
            CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow("gene JSON upper bound"),
        )
}

fn checked_per_population_json_upper_bound_bytes_v1(
    term_cap: usize,
) -> Result<u64, CanonicalNativeGenerationZeroResultErrorV1> {
    let terms = u64::try_from(term_cap)
        .map_err(|_| CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow("term cap"))?;
    GENE_JSON_UPPER_BOUND_PER_TERM_BYTES_V1
        .checked_mul(terms)
        .and_then(|bytes| PER_POPULATION_JSON_UPPER_BOUND_BASE_BYTES_V1.checked_add(bytes))
        .ok_or(
            CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow(
                "population JSON upper bound",
            ),
        )
}

fn validate_strategy_id_v1(
    strategy_id: &str,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    if !(1..=128).contains(&strategy_id.len())
        || !strategy_id.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(invalid_result_v1(
            "strategy_id must contain 1..=128 ASCII graphic bytes",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
    contract_compact_json_bytes: u64,
    contract_artifact_relative_path_compact_json_bytes: u64,
    source_count: usize,
    total_source_segment_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalNativeGenerationZeroResultPreflightV1 {
    size_plan: CanonicalNativeGenerationZeroResultSizePlanV1,
    fixed_metadata_shape: CanonicalNativeGenerationZeroFixedMetadataShapeV1,
}

impl CanonicalNativeGenerationZeroResultPreflightV1 {
    pub(crate) const fn prepared_feature_count(&self) -> usize {
        self.size_plan.prepared_feature_count()
    }

    pub(crate) const fn raw_configured_max_indicators(&self) -> usize {
        self.size_plan.raw_configured_max_indicators()
    }

    pub(crate) const fn resolved_max_indicators(&self) -> usize {
        self.size_plan.resolved_max_indicators()
    }

    pub(crate) const fn term_cap(&self) -> usize {
        self.size_plan.term_cap()
    }

    pub(crate) const fn configured_population(&self) -> usize {
        self.size_plan.configured_population()
    }

    pub(crate) const fn population_cap(&self) -> usize {
        self.size_plan.population_cap()
    }

    pub(crate) const fn fixed_metadata_upper_bound_with_empty_arrays_bytes(&self) -> u64 {
        self.size_plan
            .fixed_metadata_upper_bound_with_empty_arrays_bytes()
    }

    pub(crate) fn checked_upper_bound_for_population(
        &self,
        population: usize,
    ) -> Result<u64, CanonicalNativeGenerationZeroResultErrorV1> {
        self.size_plan
            .checked_upper_bound_for_population(population)
            .map_err(Into::into)
    }
}

fn checked_native_v3_receipt_json_upper_bound_bytes_v1(
    source_count: usize,
    total_source_segment_count: usize,
) -> Result<u64, CanonicalNativeGenerationZeroResultErrorV1> {
    validate_native_v3_source_shape_counts_v1(source_count, total_source_segment_count)?;
    let sources = u64::try_from(source_count).map_err(|_| {
        CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow("native source count")
    })?;
    let segments = u64::try_from(total_source_segment_count).map_err(|_| {
        CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow("native segment count")
    })?;
    NATIVE_V3_SOURCE_BINDING_JSON_UPPER_BOUND_BYTES_V1
        .checked_mul(sources)
        .and_then(|bytes| NATIVE_V3_FIXED_JSON_UPPER_BOUND_BYTES_V1.checked_add(bytes))
        .and_then(|bytes| {
            NATIVE_V3_SOURCE_SEGMENT_JSON_UPPER_BOUND_BYTES_V1
                .checked_mul(segments)
                .and_then(|segments| bytes.checked_add(segments))
        })
        .ok_or(
            CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow(
                "native V3 JSON upper bound",
            ),
        )
}

fn checked_fixed_metadata_upper_bound_with_empty_arrays_bytes_v1(
    shape: CanonicalNativeGenerationZeroFixedMetadataShapeV1,
) -> Result<u64, CanonicalNativeGenerationZeroResultErrorV1> {
    let native = checked_native_v3_receipt_json_upper_bound_bytes_v1(
        shape.source_count,
        shape.total_source_segment_count,
    )?;
    let native_variable = native
        .checked_sub(NATIVE_V3_FIXED_JSON_UPPER_BOUND_BYTES_V1)
        .ok_or(
            CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow(
                "native V3 fixed metadata subtraction",
            ),
        )?;
    GROUPED_FIXED_METADATA_BASE_WITH_V2_V3_JSON_BYTES_V1
        .checked_add(shape.contract_compact_json_bytes)
        .and_then(|bytes| {
            bytes.checked_add(shape.contract_artifact_relative_path_compact_json_bytes)
        })
        .and_then(|bytes| bytes.checked_add(native_variable))
        .ok_or(
            CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow(
                "fixed metadata JSON upper bound",
            ),
        )
}

fn checked_preflight_from_fixed_metadata_shape_v1(
    prepared_feature_count: usize,
    raw_configured_max_indicators: usize,
    configured_population: usize,
    fixed_metadata_shape: CanonicalNativeGenerationZeroFixedMetadataShapeV1,
) -> Result<
    CanonicalNativeGenerationZeroResultPreflightV1,
    CanonicalNativeGenerationZeroResultErrorV1,
> {
    let fixed_metadata_upper_bound_with_empty_arrays_bytes =
        checked_fixed_metadata_upper_bound_with_empty_arrays_bytes_v1(fixed_metadata_shape)?;
    let size_plan = CanonicalNativeGenerationZeroResultSizePlanV1::checked_new(
        prepared_feature_count,
        raw_configured_max_indicators,
        configured_population,
        fixed_metadata_upper_bound_with_empty_arrays_bytes,
    )?;
    Ok(CanonicalNativeGenerationZeroResultPreflightV1 {
        size_plan,
        fixed_metadata_shape,
    })
}

#[allow(dead_code)]
#[cfg(feature = "gpu-cuda")]
pub(crate) fn preflight_canonical_native_generation_zero_result_v1(
    request: &CanonicalNativeDiscoveryRequestV1,
    prepared_feature_count: usize,
) -> Result<
    CanonicalNativeGenerationZeroResultPreflightV1,
    CanonicalNativeGenerationZeroResultErrorV1,
> {
    let loaded = request.loaded_contract();
    let source_count = loaded.source_projection().bindings().len();
    let total_source_segment_count = loaded
        .source_projection()
        .bindings()
        .iter()
        .try_fold(0_usize, |total, binding| {
            total.checked_add(binding.segments().len())
        })
        .ok_or(
            CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow("source segment census"),
        )?;
    checked_preflight_from_fixed_metadata_shape_v1(
        prepared_feature_count,
        request.config().max_indicators,
        request.config().population,
        CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
            contract_compact_json_bytes: checked_compact_json_byte_count_v1(loaded.contract())?,
            contract_artifact_relative_path_compact_json_bytes:
                checked_compact_json_string_byte_count_v1(loaded.relative_path())?,
            source_count,
            total_source_segment_count,
        },
    )
}
// END CANONICAL_NATIVE_GEN0_PREFLIGHT_V1

#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CanonicalNativeGenerationZeroScoringObjectiveV1 {
    PropConsistencyV4,
    RiskyKellyGrowthV5,
}

// BEGIN CANONICAL_NATIVE_GEN0_SEALED_VIEW_V1
fn validate_native_v3_source_shape_counts_v1(
    source_count: usize,
    total_source_segment_count: usize,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    if !(1..=MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1).contains(&source_count)
        || total_source_segment_count < source_count
        || total_source_segment_count > MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1
    {
        return Err(invalid_result_v1(
            "native V3 source shape exceeds V1 limits",
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
const CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_IDENTITY_DOMAIN_V1: &[u8] =
    b"neoethos.canonical-native-generation-zero-research-result.identity.v1\0";
#[cfg(feature = "gpu-cuda")]
const EFFECTIVE_SMC_GATE_SOURCE_GENETIC_SEARCH_RUNTIME_START_GENERATION_ZERO_V1: &str =
    "genetic_search_runtime_start_generation_zero_v1";

#[cfg(feature = "gpu-cuda")]
fn is_canonical_lower_hex_sha256_v1(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(feature = "gpu-cuda")]
fn hex_lower_v1(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(feature = "gpu-cuda")]
struct Sha256OnlyWriterV1(Sha256);

#[cfg(feature = "gpu-cuda")]
impl Write for Sha256OnlyWriterV1 {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "gpu-cuda")]
fn typed_identity_sha256_v1(
    domain: &[u8],
    value: &impl Serialize,
) -> Result<String, CanonicalNativeGenerationZeroResultErrorV1> {
    let mut writer = Sha256OnlyWriterV1(Sha256::new());
    writer.0.update(domain);
    value
        .serialize(&mut serde_json::Serializer::new(&mut writer))
        .map_err(|error| {
            CanonicalNativeGenerationZeroResultErrorV1::Serialization(format!("{error}"))
        })?;
    Ok(format!("{:x}", writer.0.finalize()))
}

#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Debug, PartialEq, Serialize)]
struct CanonicalNativeGenerationZeroEvaluationSnapshotV1 {
    symbol: String,
    account_currency: String,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    trailing_min_lock_pips: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
    swap_long_pips_per_day: f64,
    swap_short_pips_per_day: f64,
    pnl_conversion_fee_rate: f64,
    smc_gate_threshold: f64,
    smc_weight_ob: f64,
    smc_weight_fvg: f64,
    smc_weight_liq: f64,
    smc_weight_mtf: f64,
    smc_weight_premium: f64,
    smc_weight_inducement: f64,
    smc_weight_bos: f64,
    smc_weight_choch: f64,
    smc_weight_eqh: f64,
    smc_weight_eql: f64,
    smc_weight_displacement: f64,
    growth_objective: bool,
}

#[cfg(feature = "gpu-cuda")]
pub(crate) struct CanonicalNativeGenerationZeroEvaluationEvidenceV1 {
    snapshot_v1: CanonicalNativeGenerationZeroEvaluationSnapshotV1,
    scoring_objective: CanonicalNativeGenerationZeroScoringObjectiveV1,
    identity_sha256: String,
}

#[cfg(feature = "gpu-cuda")]
impl CanonicalNativeGenerationZeroEvaluationEvidenceV1 {
    fn checked_from_evaluation_config_v1(
        config: &EvaluationConfig,
        mode: crate::discovery::DiscoveryMode,
    ) -> Result<Self, CanonicalNativeGenerationZeroResultErrorV1> {
        if config.symbol.is_empty()
            || config.account_currency.is_empty()
            || config.symbol.len() > MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1
            || config.account_currency.len() > MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1
        {
            return Err(invalid_result_v1("evaluation string exceeds V1 limits"));
        }
        let finite = [
            config.trailing_atr_multiplier,
            config.trailing_be_trigger_r,
            config.trailing_min_lock_pips,
            config.pip_value,
            config.spread_pips,
            config.commission_per_trade,
            config.pip_value_per_lot,
            config.swap_long_pips_per_day,
            config.swap_short_pips_per_day,
            config.pnl_conversion_fee_rate,
            config.smc_gate_threshold,
            config.smc_weight_ob,
            config.smc_weight_fvg,
            config.smc_weight_liq,
            config.smc_weight_mtf,
            config.smc_weight_premium,
            config.smc_weight_inducement,
            config.smc_weight_bos,
            config.smc_weight_choch,
            config.smc_weight_eqh,
            config.smc_weight_eql,
            config.smc_weight_displacement,
        ];
        if !finite.into_iter().all(f64::is_finite) {
            return Err(invalid_result_v1(
                "evaluation evidence contains non-finite input",
            ));
        }
        let scoring_objective = if config.growth_objective {
            if !matches!(mode, crate::discovery::DiscoveryMode::Risky) {
                return Err(invalid_result_v1(
                    "growth objective disagrees with request mode",
                ));
            }
            CanonicalNativeGenerationZeroScoringObjectiveV1::RiskyKellyGrowthV5
        } else {
            if !matches!(mode, crate::discovery::DiscoveryMode::PropFirm) {
                return Err(invalid_result_v1(
                    "consistency objective disagrees with request mode",
                ));
            }
            CanonicalNativeGenerationZeroScoringObjectiveV1::PropConsistencyV4
        };
        let snapshot_v1 = CanonicalNativeGenerationZeroEvaluationSnapshotV1 {
            symbol: config.symbol.to_owned(),
            account_currency: config.account_currency.to_owned(),
            max_hold_bars: config.max_hold_bars,
            trailing_enabled: config.trailing_enabled,
            trailing_atr_multiplier: config.trailing_atr_multiplier,
            trailing_be_trigger_r: config.trailing_be_trigger_r,
            trailing_min_lock_pips: config.trailing_min_lock_pips,
            pip_value: config.pip_value,
            spread_pips: config.spread_pips,
            commission_per_trade: config.commission_per_trade,
            pip_value_per_lot: config.pip_value_per_lot,
            swap_long_pips_per_day: config.swap_long_pips_per_day,
            swap_short_pips_per_day: config.swap_short_pips_per_day,
            pnl_conversion_fee_rate: config.pnl_conversion_fee_rate,
            smc_gate_threshold: config.smc_gate_threshold,
            smc_weight_ob: config.smc_weight_ob,
            smc_weight_fvg: config.smc_weight_fvg,
            smc_weight_liq: config.smc_weight_liq,
            smc_weight_mtf: config.smc_weight_mtf,
            smc_weight_premium: config.smc_weight_premium,
            smc_weight_inducement: config.smc_weight_inducement,
            smc_weight_bos: config.smc_weight_bos,
            smc_weight_choch: config.smc_weight_choch,
            smc_weight_eqh: config.smc_weight_eqh,
            smc_weight_eql: config.smc_weight_eql,
            smc_weight_displacement: config.smc_weight_displacement,
            growth_objective: config.growth_objective,
        };
        let identity_sha256 = typed_identity_sha256_v1(
            b"neoethos.canonical-native.gen0-evaluation.v1\0",
            &(&snapshot_v1, scoring_objective),
        )?;
        Ok(Self {
            snapshot_v1,
            scoring_objective,
            identity_sha256,
        })
    }

    fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    const fn growth_objective(&self) -> bool {
        self.snapshot_v1.growth_objective
    }

    const fn scoring_objective(&self) -> CanonicalNativeGenerationZeroScoringObjectiveV1 {
        self.scoring_objective
    }
}

#[cfg(feature = "gpu-cuda")]
struct CanonicalNativeGenerationZeroEffectiveSmcGateEvidenceV1 {
    effective_smc_gate_threshold: f64,
    source: &'static str,
    identity_sha256: String,
}

#[cfg(feature = "gpu-cuda")]
impl CanonicalNativeGenerationZeroEffectiveSmcGateEvidenceV1 {
    fn checked_new(
        threshold: f64,
        source: &'static str,
        startup_settings_id: &str,
        runtime_install_receipt_id: &str,
        runtime_authority_id: &str,
    ) -> Result<Self, CanonicalNativeGenerationZeroResultErrorV1> {
        if !threshold.is_finite()
            || source != EFFECTIVE_SMC_GATE_SOURCE_GENETIC_SEARCH_RUNTIME_START_GENERATION_ZERO_V1
            || ![
                startup_settings_id,
                runtime_install_receipt_id,
                runtime_authority_id,
            ]
            .into_iter()
            .all(is_canonical_lower_hex_sha256_v1)
        {
            return Err(invalid_result_v1("invalid effective SMC gate evidence"));
        }
        let identity_sha256 = typed_identity_sha256_v1(
            b"neoethos.canonical-native.gen0-effective-smc-gate.v1\0",
            &(
                threshold,
                source,
                startup_settings_id,
                runtime_install_receipt_id,
                runtime_authority_id,
            ),
        )?;
        Ok(Self {
            effective_smc_gate_threshold: threshold,
            source,
            identity_sha256,
        })
    }

    const fn effective_smc_gate_threshold(&self) -> f64 {
        self.effective_smc_gate_threshold
    }

    const fn source(&self) -> &'static str {
        self.source
    }

    fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }
}

#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct CanonicalNativeGenerationZeroResidencyCountersSnapshotV1 {
    parent_upload_count: u64,
    parent_upload_bytes: u64,
    view_binding_count: u64,
    full_binding_count: u64,
    range_binding_count: u64,
    ordered_binding_count: u64,
    ordered_index_upload_bytes: u64,
    adaptive_upload_bytes: u64,
    stream_creation_count: u64,
    explicit_synchronization_count: u64,
    metric_rows_readback_count: u64,
    metric_rows_readback_rows: u64,
    metric_rows_readback_bytes: u64,
    diagnostic_readback_count: u64,
    diagnostic_readback_rows: u64,
    diagnostic_readback_bytes: u64,
    accepted_trade_total_readback_count: u64,
    accepted_trade_total_readback_bytes: u64,
}

#[cfg(feature = "gpu-cuda")]
impl From<neoethos_gpu_cuda::PopulationResidencyCountersV1>
    for CanonicalNativeGenerationZeroResidencyCountersSnapshotV1
{
    fn from(value: neoethos_gpu_cuda::PopulationResidencyCountersV1) -> Self {
        Self {
            parent_upload_count: value.parent_upload_count(),
            parent_upload_bytes: value.parent_upload_bytes(),
            view_binding_count: value.view_binding_count(),
            full_binding_count: value.full_binding_count(),
            range_binding_count: value.range_binding_count(),
            ordered_binding_count: value.ordered_binding_count(),
            ordered_index_upload_bytes: value.ordered_index_upload_bytes(),
            adaptive_upload_bytes: value.adaptive_upload_bytes(),
            stream_creation_count: value.stream_creation_count(),
            explicit_synchronization_count: value.explicit_synchronization_count(),
            metric_rows_readback_count: value.metric_rows_readback_count(),
            metric_rows_readback_rows: value.metric_rows_readback_rows(),
            metric_rows_readback_bytes: value.metric_rows_readback_bytes(),
            diagnostic_readback_count: value.diagnostic_readback_count(),
            diagnostic_readback_rows: value.diagnostic_readback_rows(),
            diagnostic_readback_bytes: value.diagnostic_readback_bytes(),
            accepted_trade_total_readback_count: value.accepted_trade_total_readback_count(),
            accepted_trade_total_readback_bytes: value.accepted_trade_total_readback_bytes(),
        }
    }
}

#[cfg(feature = "gpu-cuda")]
struct CanonicalNativeGenerationZeroExecutionFactsV1<R = Vec<[u8; 32]>> {
    prepared_feature_count: usize,
    native_receipt_feature_count: usize,
    request_raw_configured_max_indicators: usize,
    sizing_requested_max_indicators: usize,
    preflight_term_cap: usize,
    sizing_term_cap: usize,
    milestone_term_cap: usize,
    request_configured_population: usize,
    sizing_configured_population: usize,
    sizing_resolved_population: usize,
    milestone_resolved_population: usize,
    population_cap: usize,
    hard_growth_cap: usize,
    max_concurrent_scenario_count: usize,
    month_capacity: usize,
    sizing_stage1_row_start: usize,
    sizing_stage1_row_end: usize,
    milestone_stage1_row_start: usize,
    milestone_stage1_row_end: usize,
    sizing_selected_device_ordinal: u32,
    milestone_selected_device_ordinal: u32,
    native_input_receipt_identity_sha256: String,
    milestone_native_input_receipt_identity_sha256: String,
    population_sizing_receipt_identity_sha256: String,
    milestone_population_sizing_receipt_identity_sha256: String,
    adaptive_base_effective_for_stage1: bool,
    sizing_resident_adaptive_request_identity_sha256: [u8; 32],
    milestone_adaptive_token_identity_sha256: Option<[u8; 32]>,
    metrics_receipt_identities_sha256: R,
    counters: CanonicalNativeGenerationZeroResidencyCountersSnapshotV1,
    engine: &'static str,
    consumer_completion_confirmed: bool,
    replay_identity_sealed: bool,
}

#[cfg(feature = "gpu-cuda")]
fn validate_execution_facts_v1<R: AsRef<[[u8; 32]]>>(
    facts: &CanonicalNativeGenerationZeroExecutionFactsV1<R>,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    let resolved_max_indicators = if facts.request_raw_configured_max_indicators == 0 {
        facts.prepared_feature_count
    } else {
        facts.request_raw_configured_max_indicators
    };
    let expected_term_cap = facts.prepared_feature_count.min(
        resolved_max_indicators
            .max(crate::genetic::seed_templates::PROFESSIONAL_TEMPLATE_MAX_TERMS_V1),
    );
    let expected_hard_growth_cap =
        RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2.min(facts.population_cap);
    let launches = facts
        .sizing_resolved_population
        .checked_add(facts.max_concurrent_scenario_count.saturating_sub(1))
        .and_then(|value| value.checked_div(facts.max_concurrent_scenario_count.max(1)))
        .ok_or(
            CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow("metric launch count"),
        )?;
    let month_capacity = u32::try_from(facts.month_capacity).map_err(|_| {
        CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow("metric month capacity")
    })?;
    let metric_bytes = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
        facts.sizing_resolved_population,
        month_capacity,
    )
    .map_err(|error| invalid_result_v1(format!("metric plan: {error}")))?
    .metric_rows_bytes();
    let c = facts.counters;
    let identities_are_valid = [
        facts.native_input_receipt_identity_sha256.as_str(),
        facts
            .milestone_native_input_receipt_identity_sha256
            .as_str(),
        facts.population_sizing_receipt_identity_sha256.as_str(),
        facts
            .milestone_population_sizing_receipt_identity_sha256
            .as_str(),
    ]
    .into_iter()
    .all(is_canonical_lower_hex_sha256_v1);
    let adaptive_is_valid = if facts.adaptive_base_effective_for_stage1 {
        facts.sizing_resident_adaptive_request_identity_sha256 != [0; 32]
            && facts
                .milestone_adaptive_token_identity_sha256
                .is_some_and(|token_identity| {
                    token_identity != [0; 32]
                        && token_identity != facts.sizing_resident_adaptive_request_identity_sha256
                })
    } else {
        facts.sizing_resident_adaptive_request_identity_sha256 == [0; 32]
            && facts.milestone_adaptive_token_identity_sha256.is_none()
    };
    let counters_are_valid = c.parent_upload_count == 0
        && c.parent_upload_bytes == 0
        && c.view_binding_count == 1
        && c.full_binding_count.checked_add(c.range_binding_count) == Some(1)
        && c.ordered_binding_count == 0
        && c.ordered_index_upload_bytes == 0
        && c.adaptive_upload_bytes == 0
        && c.stream_creation_count == 0
        && c.explicit_synchronization_count == launches as u64
        && c.metric_rows_readback_count == launches as u64
        && c.metric_rows_readback_rows == facts.sizing_resolved_population as u64
        && c.metric_rows_readback_bytes == metric_bytes
        && c.diagnostic_readback_count == 0
        && c.diagnostic_readback_rows == 0
        && c.diagnostic_readback_bytes == 0
        && c.accepted_trade_total_readback_count == 0
        && c.accepted_trade_total_readback_bytes == 0;
    let valid = facts.prepared_feature_count > 0
        && facts.prepared_feature_count == facts.native_receipt_feature_count
        && facts.sizing_requested_max_indicators == resolved_max_indicators
        && facts.preflight_term_cap == expected_term_cap
        && facts.sizing_term_cap == expected_term_cap
        && facts.milestone_term_cap == expected_term_cap
        && facts.request_configured_population
            >= MIN_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1
        && facts.request_configured_population == facts.sizing_configured_population
        && facts.sizing_resolved_population >= facts.sizing_configured_population
        && facts.sizing_resolved_population == facts.milestone_resolved_population
        && facts.sizing_resolved_population <= facts.population_cap
        && facts.sizing_resolved_population
            <= facts
                .sizing_configured_population
                .max(facts.hard_growth_cap)
        && facts.hard_growth_cap == expected_hard_growth_cap
        && facts.max_concurrent_scenario_count > 0
        && facts.month_capacity > 0
        && facts.sizing_stage1_row_start < facts.sizing_stage1_row_end
        && facts.sizing_stage1_row_start == facts.milestone_stage1_row_start
        && facts.sizing_stage1_row_end == facts.milestone_stage1_row_end
        && facts.sizing_selected_device_ordinal == facts.milestone_selected_device_ordinal
        && identities_are_valid
        && facts.native_input_receipt_identity_sha256
            == facts.milestone_native_input_receipt_identity_sha256
        && facts.population_sizing_receipt_identity_sha256
            == facts.milestone_population_sizing_receipt_identity_sha256
        && facts.metrics_receipt_identities_sha256.as_ref().len() == launches
        && facts
            .metrics_receipt_identities_sha256
            .as_ref()
            .iter()
            .all(|identity| *identity != [0; 32])
        && adaptive_is_valid
        && counters_are_valid
        && facts.engine == "CudaNativeF64"
        && facts.consumer_completion_confirmed
        && !facts.replay_identity_sealed;
    if !valid {
        return Err(invalid_result_v1(
            "Generation-zero execution facts are contradictory",
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Copy)]
struct CanonicalNativeGenerationZeroPolicyFactsV1 {
    execution_scope: CanonicalNativeExecutionScopeV1,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    authorization_issued: bool,
    cost_band_status: CanonicalNativeCostBandStatusV1,
    consumer_completion_confirmed: bool,
    replay_identity_sealed: bool,
}

#[cfg(feature = "gpu-cuda")]
fn validate_policy_facts_v1(
    facts: &CanonicalNativeGenerationZeroPolicyFactsV1,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    if !matches!(
        facts.execution_scope,
        CanonicalNativeExecutionScopeV1::GenerationZeroOnly
    ) || !matches!(
        facts.artifact_class,
        HistoricalResearchArtifactClassV1::ResearchOnly
    ) || !matches!(
        facts.promotion_eligibility,
        HistoricalResearchPromotionEligibilityV1::NotPromotionEligible
    ) || facts.authorization_issued
        || !matches!(
            facts.cost_band_status,
            CanonicalNativeCostBandStatusV1::UnusedGenerationZero
        )
        || !facts.consumer_completion_confirmed
        || facts.replay_identity_sealed
    {
        return Err(invalid_result_v1(
            "ResearchOnly policy facts are contradictory",
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
fn validate_population_payload_v1(
    genes: &[Gene],
    metrics: &[[f64; 11]],
    effective_smc_gate_threshold: f64,
    resolved_population: usize,
    prepared_feature_count: usize,
    term_cap: usize,
    growth_objective: bool,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    if resolved_population == 0
        || prepared_feature_count == 0
        || term_cap == 0
        || genes.len() != resolved_population
        || metrics.len() != resolved_population
        || !effective_smc_gate_threshold.is_finite()
    {
        return Err(invalid_result_v1(
            "population cardinality or gate is invalid",
        ));
    }
    for (gene, row) in genes.iter().zip(metrics) {
        if !row.iter().all(|value| value.is_finite())
            || !row[8].is_sign_positive()
            || row[8].fract() != 0.0
            || row[8] > usize::MAX as f64
            || gene.generation != 0
            || gene.indices.is_empty()
            || gene.indices.len() != gene.weights.len()
            || gene.indices.len() > term_cap
            || gene
                .indices
                .iter()
                .any(|index| *index >= prepared_feature_count)
            || gene.indices.windows(2).any(|pair| pair[0] >= pair[1])
            || gene.weights.iter().any(|weight| {
                !weight.is_finite() || weight.abs() <= 1.0e-6 || !(-5.0..=5.0).contains(weight)
            })
            || ![
                gene.long_threshold,
                gene.short_threshold,
                gene.fitness,
                gene.sharpe_ratio,
                gene.win_rate,
                gene.max_drawdown,
                gene.profit_factor,
                gene.expectancy,
                gene.tp_pips,
                gene.sl_pips,
                gene.slice_pass_rate,
                gene.consistency,
                gene.stop_vol_mult,
            ]
            .into_iter()
            .all(f64::is_finite)
            || gene.long_threshold <= gene.short_threshold
            || gene.sl_pips <= 0.0
            || gene.tp_pips <= 0.0
            || gene.stop_vol_mult < 0.0
        {
            return Err(invalid_result_v1(
                "canonical Generation-zero gene is invalid",
            ));
        }
        validate_strategy_id_v1(&gene.strategy_id)?;
        let expected_fitness = if growth_objective {
            crate::scoring::ga_fitness_growth(row)
        } else {
            crate::scoring::ga_fitness(row)
        };
        if gene.fitness.to_bits() != expected_fitness.to_bits()
            || gene.sharpe_ratio.to_bits() != row[1].to_bits()
            || gene.max_drawdown.to_bits() != row[3].to_bits()
            || gene.win_rate.to_bits() != row[4].to_bits()
            || gene.profit_factor.to_bits() != row[5].to_bits()
            || gene.expectancy.to_bits() != row[6].to_bits()
            || gene.trades_count != row[8] as usize
            || gene.consistency.to_bits() != row[9].to_bits()
            || gene.slice_pass_rate.to_bits() != 1.0_f64.to_bits()
        {
            return Err(invalid_result_v1(
                "derived Gene fields disagree with metric row",
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
fn validate_contract_evidence_v1(
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    actual_domain_identity_sha256: &str,
    actual_file_sha256: &str,
    expected_domain_identity_sha256: &str,
    expected_file_sha256: &str,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    contract
        .validate()
        .map_err(|error| invalid_result_v1(format!("financial contract: {error:#}")))?;
    contract
        .validate_against_receipt(contract.input_receipt())
        .map_err(|error| invalid_result_v1(format!("financial receipt: {error:#}")))?;
    let computed = contract
        .identity_sha256()
        .map_err(|error| invalid_result_v1(format!("financial contract identity: {error:#}")))?;
    if ![
        actual_domain_identity_sha256,
        actual_file_sha256,
        expected_domain_identity_sha256,
        expected_file_sha256,
    ]
    .into_iter()
    .all(is_canonical_lower_hex_sha256_v1)
        || actual_domain_identity_sha256 != computed
        || actual_domain_identity_sha256 != expected_domain_identity_sha256
        || actual_file_sha256 != expected_file_sha256
    {
        return Err(invalid_result_v1(
            "contract file/domain evidence is detached",
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
fn validate_result_native_receipt_v3(
    receipt: &CanonicalGpuResidentSearchInputReceiptV3,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    receipt
        .validate()
        .map_err(|error| invalid_result_v1(format!("native V3 receipt: {error}")))?;
    let source_count = receipt.source_bindings().len();
    let total_source_segment_count =
        receipt
            .source_bindings()
            .iter()
            .try_fold(0_usize, |total, binding| {
                total.checked_add(binding.segments().len()).ok_or(
                    CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow(
                        "native V3 source segment census",
                    ),
                )
            })?;
    validate_native_v3_source_shape_counts_v1(source_count, total_source_segment_count)?;
    let general_strings_are_bounded = receipt.anchor_dataset_identity().len()
        <= MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1
        && receipt.source_bindings().iter().all(|binding| {
            [
                binding.source_node_id(),
                binding.dataset_identity(),
                binding.manifest_schema_id(),
                binding.generation_id(),
                binding.bar_timestamp_convention(),
            ]
            .into_iter()
            .all(|value| value.len() <= MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1)
        });
    let hashes_are_canonical = [
        receipt.feature_plan_identity(),
        receipt.feature_provenance_identity(),
        receipt.feature_content_merkle_sha256(),
        receipt.normalization_fit_sha256(),
    ]
    .into_iter()
    .all(is_canonical_lower_hex_sha256_v1)
        && receipt.source_bindings().iter().all(|binding| {
            is_canonical_lower_hex_sha256_v1(binding.manifest_sha256())
                && is_canonical_lower_hex_sha256_v1(binding.vortex_sha256())
        });
    if !general_strings_are_bounded || !hashes_are_canonical {
        return Err(invalid_result_v1(
            "native V3 receipt exceeds result schema bounds",
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
fn validate_native_source_projection_v1(
    receipt: &CanonicalGpuResidentSearchInputReceiptV3,
    projection: &neoethos_data::CanonicalPinnedSourceProjectionV1,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    validate_result_native_receipt_v3(receipt)?;
    if receipt.anchor_dataset_identity() != projection.anchor_dataset_identity().to_path_component()
        || receipt.row_count() != projection.parent_row_count()
        || receipt.source_bindings().len() != projection.bindings().len()
    {
        return Err(invalid_result_v1(
            "native V3 source projection header drifted",
        ));
    }
    for (actual, expected) in receipt.source_bindings().iter().zip(projection.bindings()) {
        if actual.dataset_identity() != expected.dataset_identity().to_path_component()
            || actual.manifest_schema_id() != expected.manifest_schema_id()
            || actual.manifest_sha256() != hex_lower_v1(&expected.manifest_sha256())
            || actual.generation_id() != expected.generation_id()
            || actual.vortex_sha256() != hex_lower_v1(&expected.vortex_sha256())
            || actual.bar_timestamp_convention() != expected.bar_timestamp_convention().as_str()
            || actual.segments().len() != expected.segments().len()
        {
            return Err(invalid_result_v1("native V3 source binding drifted"));
        }
        for (actual_segment, expected_segment) in actual.segments().iter().zip(expected.segments())
        {
            if actual_segment.row_start() != expected_segment.row_start()
                || actual_segment.row_end() != expected_segment.row_end()
                || actual_segment.timestamp_start_ms() != expected_segment.timestamp_start_ms()
                || actual_segment.timestamp_end_ms() != expected_segment.timestamp_end_ms()
            {
                return Err(invalid_result_v1("native V3 source segment drifted"));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
pub(crate) struct CanonicalNativeGenerationZeroRequestEvidenceV1 {
    execution_scope: CanonicalNativeExecutionScopeV1,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    authorization_issued: bool,
    contract_artifact_reference_schema: String,
    contract_artifact_reference_version: u16,
    contract_artifact_relative_path: String,
    contract_artifact_expected_sha256: String,
    contract_artifact_exact_file_sha256: String,
    contract_artifact_exact_file_byte_count: u64,
    contract_domain_identity_sha256: String,
    startup_settings_id: String,
    runtime_install_receipt_id: String,
    generation_zero_runtime_authority_id: String,
    unused_full_search_scope_id: String,
    raw_generations: usize,
    clamped_generations: usize,
    cost_band_status: CanonicalNativeCostBandStatusV1,
    cost_band: Option<(f64, f64)>,
    configured_population_cap: usize,
    resolved_population_cap: usize,
    term_cap: usize,
    string_bytes_cap: usize,
    vector_elements_cap: usize,
    source_count_cap: usize,
    result_bytes_cap: u64,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Copy)]
struct LowerHexSha256V1([u8; 32]);

#[cfg(feature = "gpu-cuda")]
impl LowerHexSha256V1 {
    fn encoded_v1(self) -> [u8; 64] {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = [0_u8; 64];
        for (index, byte) in self.0.into_iter().enumerate() {
            encoded[index * 2] = HEX[(byte >> 4) as usize];
            encoded[index * 2 + 1] = HEX[(byte & 0x0f) as usize];
        }
        encoded
    }
}

#[cfg(feature = "gpu-cuda")]
impl Serialize for LowerHexSha256V1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let encoded = self.encoded_v1();
        let value = std::str::from_utf8(&encoded).map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(value)
    }
}

#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Copy)]
struct LowerHexSha256SliceV1<'a>(&'a [[u8; 32]]);

#[cfg(feature = "gpu-cuda")]
impl Serialize for LowerHexSha256SliceV1<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for identity in self.0 {
            sequence.serialize_element(&LowerHexSha256V1(*identity))?;
        }
        sequence.end()
    }
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroArtifactReferenceWireV1<'a> {
    schema: &'static str,
    version: u16,
    relative_path: &'a str,
    expected_sha256: &'a str,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroContractArtifactWireV1<'a> {
    reference: CanonicalNativeGenerationZeroArtifactReferenceWireV1<'a>,
    exact_file_sha256: &'a str,
    exact_file_byte_count: u64,
    contract_domain_identity_sha256: &'a str,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroRuntimeAuthorityWireV1<'a> {
    startup_settings_id: &'a str,
    runtime_install_receipt_id: &'a str,
    generation_zero_runtime_authority_id: &'a str,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroUnusedFullSearchWireV1<'a> {
    scope_id: &'a str,
    raw_generations: usize,
    clamped_generations: usize,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroCostBandWireV1 {
    status: CanonicalNativeCostBandStatusV1,
    cost: Option<(f64, f64)>,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroLimitsWireV1 {
    configured_population_cap: usize,
    resolved_population_cap: usize,
    term_cap: usize,
    string_bytes_cap: usize,
    vector_elements_cap: usize,
    source_count_cap: usize,
    result_bytes_cap: u64,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroFinancialProvenanceWireV1<'a> {
    contract: &'a CanonicalTrendbarResearchExecutionContractV3,
    cpu_receipt_id: &'a str,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroEvaluatedNativeInputWireV1<'a> {
    receipt_v3: &'a CanonicalGpuResidentSearchInputReceiptV3,
    receipt_id: &'a str,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroPopulationSizingWireV1<'a> {
    receipt_v2: &'a ResidentPopulationAutoSizingReceiptV2,
    receipt_id: &'a str,
    prepared_feature_count: usize,
    raw_configured_max_indicators: usize,
    resolved_max_indicators: usize,
    term_cap: usize,
    configured_population: usize,
    resolved_population: usize,
    population_cap: usize,
    hard_growth_cap: usize,
    max_concurrent_scenario_count: usize,
    stage1_row_start: usize,
    stage1_row_end: usize,
    selected_device_ordinal: u32,
    metrics_receipt_identities_sha256: LowerHexSha256SliceV1<'a>,
    adaptive_token_identity_sha256: Option<LowerHexSha256V1>,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroEvaluationWireV1<'a> {
    snapshot_v1: &'a CanonicalNativeGenerationZeroEvaluationSnapshotV1,
    snapshot_identity_sha256: &'a str,
    scoring_objective: CanonicalNativeGenerationZeroScoringObjectiveV1,
    effective_smc_gate_threshold: f64,
    effective_smc_gate_source: &'static str,
    genes: &'a [Gene],
    metrics: &'a [[f64; 11]],
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroCompletionWireV1 {
    engine: &'static str,
    consumer_completion_confirmed: bool,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct CanonicalNativeGenerationZeroReplayWireV1 {
    replay_identity_sealed: bool,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct IdentityMaterialV1<'a> {
    schema: &'static str,
    version: u16,
    scope: CanonicalNativeExecutionScopeV1,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    authorization_issued: bool,
    contract_artifact: CanonicalNativeGenerationZeroContractArtifactWireV1<'a>,
    runtime_authority: CanonicalNativeGenerationZeroRuntimeAuthorityWireV1<'a>,
    unused_full_search: CanonicalNativeGenerationZeroUnusedFullSearchWireV1<'a>,
    cost_band_status: CanonicalNativeGenerationZeroCostBandWireV1,
    limits: CanonicalNativeGenerationZeroLimitsWireV1,
    financial_provenance_only: CanonicalNativeGenerationZeroFinancialProvenanceWireV1<'a>,
    evaluated_native_input: CanonicalNativeGenerationZeroEvaluatedNativeInputWireV1<'a>,
    population_sizing: CanonicalNativeGenerationZeroPopulationSizingWireV1<'a>,
    generation_zero_evaluation: CanonicalNativeGenerationZeroEvaluationWireV1<'a>,
    residency_counters: CanonicalNativeGenerationZeroResidencyCountersSnapshotV1,
    completion: CanonicalNativeGenerationZeroCompletionWireV1,
    replay: CanonicalNativeGenerationZeroReplayWireV1,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Serialize)]
struct ResultWireV1<'a> {
    #[serde(flatten)]
    identity_material: IdentityMaterialV1<'a>,
    evidence_identity_sha256: &'a str,
}

#[cfg(feature = "gpu-cuda")]
pub(crate) struct CanonicalNativeGenerationZeroResearchResultViewV1<'a> {
    preflight: CanonicalNativeGenerationZeroResultPreflightV1,
    request_evidence: CanonicalNativeGenerationZeroRequestEvidenceV1,
    financial_execution_contract_v3: CanonicalTrendbarResearchExecutionContractV3,
    native_input_receipt_v3: CanonicalGpuResidentSearchInputReceiptV3,
    population_sizing_receipt_v2: ResidentPopulationAutoSizingReceiptV2,
    evaluation_evidence_v1: CanonicalNativeGenerationZeroEvaluationEvidenceV1,
    milestone: &'a ResidentGenerationZeroMilestoneV1,
    evidence_identity_sha256: String,
}

#[cfg(feature = "gpu-cuda")]
impl CanonicalNativeGenerationZeroResearchResultViewV1<'_> {
    pub(crate) const fn milestone(&self) -> &ResidentGenerationZeroMilestoneV1 {
        self.milestone
    }

    pub(crate) const fn preflight(&self) -> &CanonicalNativeGenerationZeroResultPreflightV1 {
        &self.preflight
    }

    pub(crate) fn evidence_identity_sha256(&self) -> &str {
        &self.evidence_identity_sha256
    }

    pub(crate) fn financial_input_receipt_identity_sha256(&self) -> &str {
        self.financial_execution_contract_v3.input_receipt_sha256()
    }

    fn identity_material_v1(&self) -> IdentityMaterialV1<'_> {
        let request = &self.request_evidence;
        let sizing = &self.population_sizing_receipt_v2;
        let result = self.milestone.search_result();
        IdentityMaterialV1 {
            schema: CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1,
            version: CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1,
            scope: request.execution_scope,
            artifact_class: request.artifact_class,
            promotion_eligibility: request.promotion_eligibility,
            authorization_issued: request.authorization_issued,
            contract_artifact: CanonicalNativeGenerationZeroContractArtifactWireV1 {
                reference: CanonicalNativeGenerationZeroArtifactReferenceWireV1 {
                    schema: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1,
                    version: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1,
                    relative_path: &request.contract_artifact_relative_path,
                    expected_sha256: &request.contract_artifact_expected_sha256,
                },
                exact_file_sha256: &request.contract_artifact_exact_file_sha256,
                exact_file_byte_count: request.contract_artifact_exact_file_byte_count,
                contract_domain_identity_sha256: &request.contract_domain_identity_sha256,
            },
            runtime_authority: CanonicalNativeGenerationZeroRuntimeAuthorityWireV1 {
                startup_settings_id: &request.startup_settings_id,
                runtime_install_receipt_id: &request.runtime_install_receipt_id,
                generation_zero_runtime_authority_id: &request.generation_zero_runtime_authority_id,
            },
            unused_full_search: CanonicalNativeGenerationZeroUnusedFullSearchWireV1 {
                scope_id: &request.unused_full_search_scope_id,
                raw_generations: request.raw_generations,
                clamped_generations: request.clamped_generations,
            },
            cost_band_status: CanonicalNativeGenerationZeroCostBandWireV1 {
                status: request.cost_band_status,
                cost: request.cost_band,
            },
            limits: CanonicalNativeGenerationZeroLimitsWireV1 {
                configured_population_cap: request.configured_population_cap,
                resolved_population_cap: request.resolved_population_cap,
                term_cap: request.term_cap,
                string_bytes_cap: request.string_bytes_cap,
                vector_elements_cap: request.vector_elements_cap,
                source_count_cap: request.source_count_cap,
                result_bytes_cap: request.result_bytes_cap,
            },
            financial_provenance_only: CanonicalNativeGenerationZeroFinancialProvenanceWireV1 {
                contract: &self.financial_execution_contract_v3,
                cpu_receipt_id: self.financial_execution_contract_v3.input_receipt_sha256(),
            },
            evaluated_native_input: CanonicalNativeGenerationZeroEvaluatedNativeInputWireV1 {
                receipt_v3: &self.native_input_receipt_v3,
                receipt_id: self.milestone.native_input_receipt_identity_sha256(),
            },
            population_sizing: CanonicalNativeGenerationZeroPopulationSizingWireV1 {
                receipt_v2: sizing,
                receipt_id: sizing.identity_sha256(),
                prepared_feature_count: self.preflight.prepared_feature_count(),
                raw_configured_max_indicators: self.preflight.raw_configured_max_indicators(),
                resolved_max_indicators: self.preflight.resolved_max_indicators(),
                term_cap: self.preflight.term_cap(),
                configured_population: sizing.configured_population(),
                resolved_population: sizing.resolved_population(),
                population_cap: self.preflight.population_cap(),
                hard_growth_cap: sizing.hard_growth_cap(),
                max_concurrent_scenario_count: sizing.max_concurrent_scenario_count(),
                stage1_row_start: sizing.stage1_row_start(),
                stage1_row_end: sizing.stage1_row_end(),
                selected_device_ordinal: sizing.selected_device_ordinal(),
                metrics_receipt_identities_sha256: LowerHexSha256SliceV1(
                    self.milestone.metrics_receipt_identities_sha256(),
                ),
                adaptive_token_identity_sha256: self
                    .milestone
                    .adaptive_token_identity_sha256()
                    .map(LowerHexSha256V1),
            },
            generation_zero_evaluation: CanonicalNativeGenerationZeroEvaluationWireV1 {
                snapshot_v1: &self.evaluation_evidence_v1.snapshot_v1,
                snapshot_identity_sha256: self.evaluation_evidence_v1.identity_sha256(),
                scoring_objective: self.evaluation_evidence_v1.scoring_objective(),
                effective_smc_gate_threshold: result.effective_smc_gate_threshold,
                effective_smc_gate_source:
                    EFFECTIVE_SMC_GATE_SOURCE_GENETIC_SEARCH_RUNTIME_START_GENERATION_ZERO_V1,
                genes: &result.genes,
                metrics: &result.metrics,
            },
            residency_counters: self.milestone.residency_counters().into(),
            completion: CanonicalNativeGenerationZeroCompletionWireV1 {
                engine: self.milestone.engine(),
                consumer_completion_confirmed: self.milestone.consumer_completion_confirmed(),
            },
            replay: CanonicalNativeGenerationZeroReplayWireV1 {
                replay_identity_sealed: self.milestone.replay_identity_sealed(),
            },
        }
    }

    fn result_wire_v1(&self) -> ResultWireV1<'_> {
        ResultWireV1 {
            identity_material: self.identity_material_v1(),
            evidence_identity_sha256: &self.evidence_identity_sha256,
        }
    }

    fn checked_fixed_metadata_with_empty_arrays_byte_count_v1(
        &self,
    ) -> Result<u64, CanonicalNativeGenerationZeroResultErrorV1> {
        const PLACEHOLDER: &str =
            "0000000000000000000000000000000000000000000000000000000000000000";
        let mut wire = self.result_wire_v1();
        wire.identity_material
            .population_sizing
            .metrics_receipt_identities_sha256 = LowerHexSha256SliceV1(&[]);
        wire.identity_material.generation_zero_evaluation.genes = &[];
        wire.identity_material.generation_zero_evaluation.metrics = &[];
        wire.evidence_identity_sha256 = PLACEHOLDER;
        checked_compact_json_byte_count_v1(&wire)
    }
}

#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalNativeGenerationZeroCompactJsonSealV1 {
    byte_count: u64,
    sha256: String,
}

#[cfg(feature = "gpu-cuda")]
impl CanonicalNativeGenerationZeroCompactJsonSealV1 {
    pub(crate) const fn byte_count(&self) -> u64 {
        self.byte_count
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[cfg(feature = "gpu-cuda")]
fn validate_request_evidence_v1(
    evidence: &CanonicalNativeGenerationZeroRequestEvidenceV1,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    let policy = CanonicalNativeGenerationZeroPolicyFactsV1 {
        execution_scope: evidence.execution_scope,
        artifact_class: evidence.artifact_class,
        promotion_eligibility: evidence.promotion_eligibility,
        authorization_issued: evidence.authorization_issued,
        cost_band_status: evidence.cost_band_status,
        consumer_completion_confirmed: true,
        replay_identity_sealed: false,
    };
    validate_policy_facts_v1(&policy)?;
    let hashes_are_valid = [
        evidence.contract_artifact_expected_sha256.as_str(),
        evidence.contract_artifact_exact_file_sha256.as_str(),
        evidence.contract_domain_identity_sha256.as_str(),
        evidence.startup_settings_id.as_str(),
        evidence.runtime_install_receipt_id.as_str(),
        evidence.generation_zero_runtime_authority_id.as_str(),
        evidence.unused_full_search_scope_id.as_str(),
    ]
    .into_iter()
    .all(is_canonical_lower_hex_sha256_v1);
    let finite_cost = evidence.cost_band.map_or(true, |(lower, upper)| {
        lower.is_finite() && upper.is_finite()
    });
    let mut scope_digest = Sha256::new();
    scope_digest.update(b"neoethos.canonical-native.gen0-scope.v1\0");
    scope_digest.update((evidence.raw_generations as u64).to_le_bytes());
    scope_digest.update((evidence.clamped_generations as u64).to_le_bytes());
    for value in evidence
        .cost_band
        .into_iter()
        .flat_map(|pair| [pair.0, pair.1])
    {
        scope_digest.update(value.to_bits().to_le_bytes());
    }
    let scope_identity = format!("{:x}", scope_digest.finalize());
    let valid = hashes_are_valid
        && finite_cost
        && evidence.contract_artifact_reference_schema
            == CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1
        && evidence.contract_artifact_reference_version
            == CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1
        && !evidence.contract_artifact_relative_path.is_empty()
        && evidence.contract_artifact_relative_path.len()
            <= MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1
        && evidence.contract_artifact_expected_sha256
            == evidence.contract_artifact_exact_file_sha256
        && evidence.contract_artifact_exact_file_byte_count > 0
        && evidence.unused_full_search_scope_id == scope_identity
        && evidence.configured_population_cap == MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1
        && evidence.resolved_population_cap == MAX_CANONICAL_NATIVE_GEN0_RESOLVED_POPULATION_V1
        && evidence.term_cap == MAX_CANONICAL_NATIVE_GEN0_TERMS_V1
        && evidence.string_bytes_cap == MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1
        && evidence.vector_elements_cap == MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1
        && evidence.source_count_cap == MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1
        && evidence.result_bytes_cap == MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1;
    if !valid {
        return Err(invalid_result_v1("request evidence is contradictory"));
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
fn request_evidence_from_request_v1(
    request: &CanonicalNativeDiscoveryRequestV1,
) -> CanonicalNativeGenerationZeroRequestEvidenceV1 {
    let loaded = request.loaded_contract();
    let scope = request.scope();
    let limits = request.limits();
    CanonicalNativeGenerationZeroRequestEvidenceV1 {
        execution_scope: scope.execution_scope(),
        artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
        promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
        authorization_issued: false,
        contract_artifact_reference_schema: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1
            .to_owned(),
        contract_artifact_reference_version: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1,
        contract_artifact_relative_path: loaded.relative_path().to_owned(),
        contract_artifact_expected_sha256: loaded.exact_artifact_sha256().to_owned(),
        contract_artifact_exact_file_sha256: loaded.exact_artifact_sha256().to_owned(),
        contract_artifact_exact_file_byte_count: loaded.byte_len(),
        contract_domain_identity_sha256: loaded.contract_identity_sha256().to_owned(),
        startup_settings_id: request.startup_settings_sha256().to_owned(),
        runtime_install_receipt_id: request
            .runtime_install_receipt()
            .identity_sha256()
            .to_owned(),
        generation_zero_runtime_authority_id: request
            .runtime_authority()
            .identity_sha256()
            .to_owned(),
        unused_full_search_scope_id: scope.identity_sha256().to_owned(),
        raw_generations: scope.raw_legacy_generations_unused_full_search(),
        clamped_generations: scope.clamped_legacy_generations_unused_full_search(),
        cost_band_status: scope.cost_band_status(),
        cost_band: scope.cost_band_pips_unused_generation_zero(),
        configured_population_cap: limits.configured_population_cap(),
        resolved_population_cap: limits.resolved_population_cap(),
        term_cap: limits.term_cap(),
        string_bytes_cap: limits.string_bytes_cap(),
        vector_elements_cap: limits.vector_elements_cap(),
        source_count_cap: limits.source_count_cap(),
        result_bytes_cap: limits.result_bytes_cap(),
    }
}

#[cfg(feature = "gpu-cuda")]
fn validate_evaluation_evidence_v1(
    evidence: &CanonicalNativeGenerationZeroEvaluationEvidenceV1,
    contract: &CanonicalTrendbarResearchExecutionContractV3,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    let expected_identity = typed_identity_sha256_v1(
        b"neoethos.canonical-native.gen0-evaluation.v1\0",
        &(evidence.snapshot_v1.clone(), evidence.scoring_objective),
    )?;
    let snapshot = &evidence.snapshot_v1;
    let financials_match = snapshot.symbol == contract.symbol()
        && snapshot.account_currency == contract.account_currency()
        && snapshot.pip_value.to_bits() == contract.pip_size().to_bits()
        && snapshot.pip_value_per_lot.to_bits() == contract.pip_value_per_lot().to_bits()
        && snapshot.spread_pips.to_bits()
            == contract
                .screening_spread_and_slippage_round_trip_pips()
                .to_bits()
        && snapshot.commission_per_trade.to_bits()
            == contract.round_trip_commission_account_per_lot().to_bits()
        && snapshot.swap_long_pips_per_day.to_bits() == contract.swap_long_pips_per_day().to_bits()
        && snapshot.swap_short_pips_per_day.to_bits()
            == contract.swap_short_pips_per_day().to_bits()
        && snapshot.pnl_conversion_fee_rate.to_bits()
            == contract.pnl_conversion_fee_rate().to_bits();
    if expected_identity != evidence.identity_sha256 || !financials_match {
        return Err(invalid_result_v1(
            "evaluation evidence is detached from financial authority",
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
fn validate_preflight_authority_v1(
    preflight: &CanonicalNativeGenerationZeroResultPreflightV1,
    request: &CanonicalNativeGenerationZeroRequestEvidenceV1,
    contract: &CanonicalTrendbarResearchExecutionContractV3,
    native: &CanonicalGpuResidentSearchInputReceiptV3,
    sizing: &ResidentPopulationAutoSizingReceiptV2,
) -> Result<(), CanonicalNativeGenerationZeroResultErrorV1> {
    let source_count = native.source_bindings().len();
    let segment_count = native
        .source_bindings()
        .iter()
        .try_fold(0_usize, |total, binding| {
            total.checked_add(binding.segments().len())
        })
        .ok_or(
            CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow(
                "actual source segment census",
            ),
        )?;
    let actual_shape = CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
        contract_compact_json_bytes: checked_compact_json_byte_count_v1(contract)?,
        contract_artifact_relative_path_compact_json_bytes:
            checked_compact_json_string_byte_count_v1(&request.contract_artifact_relative_path)?,
        source_count,
        total_source_segment_count: segment_count,
    };
    let valid = preflight.fixed_metadata_shape == actual_shape
        && preflight.prepared_feature_count() == sizing.feature_count()
        && preflight.resolved_max_indicators() == sizing.requested_max_indicators()
        && preflight.term_cap() == sizing.term_cap()
        && preflight.configured_population() == sizing.configured_population()
        && preflight.fixed_metadata_upper_bound_with_empty_arrays_bytes()
            == checked_fixed_metadata_upper_bound_with_empty_arrays_bytes_v1(actual_shape)?;
    if !valid {
        return Err(invalid_result_v1(
            "preflight is detached from actual bounded metadata",
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
fn checked_execution_facts_from_authorities_v1<'a>(
    preflight: &CanonicalNativeGenerationZeroResultPreflightV1,
    native: &CanonicalGpuResidentSearchInputReceiptV3,
    sizing: &ResidentPopulationAutoSizingReceiptV2,
    milestone: &'a ResidentGenerationZeroMilestoneV1,
) -> Result<
    CanonicalNativeGenerationZeroExecutionFactsV1<&'a [[u8; 32]]>,
    CanonicalNativeGenerationZeroResultErrorV1,
> {
    let native_feature_count = usize::try_from(native.column_count()).map_err(|_| {
        CanonicalNativeGenerationZeroResultErrorV1::ArithmeticOverflow("native feature count")
    })?;
    let native_identity = native
        .identity_sha256()
        .map_err(|error| invalid_result_v1(format!("native identity: {error}")))?;
    let adaptive_request_identity = sizing
        .resident_adaptive_view_and_request_v2()
        .map_err(|error| invalid_result_v1(format!("adaptive sizing: {error}")))?
        .map_or([0; 32], |(_, request)| request.identity_sha256());
    Ok(CanonicalNativeGenerationZeroExecutionFactsV1 {
        prepared_feature_count: preflight.prepared_feature_count(),
        native_receipt_feature_count: native_feature_count,
        request_raw_configured_max_indicators: preflight.raw_configured_max_indicators(),
        sizing_requested_max_indicators: sizing.requested_max_indicators(),
        preflight_term_cap: preflight.term_cap(),
        sizing_term_cap: sizing.term_cap(),
        milestone_term_cap: milestone.term_cap(),
        request_configured_population: preflight.configured_population(),
        sizing_configured_population: sizing.configured_population(),
        sizing_resolved_population: sizing.resolved_population(),
        milestone_resolved_population: milestone.resolved_population(),
        population_cap: preflight.population_cap(),
        hard_growth_cap: sizing.hard_growth_cap(),
        max_concurrent_scenario_count: sizing.max_concurrent_scenario_count(),
        month_capacity: sizing.month_capacity(),
        sizing_stage1_row_start: sizing.stage1_row_start(),
        sizing_stage1_row_end: sizing.stage1_row_end(),
        milestone_stage1_row_start: milestone.stage1_row_start(),
        milestone_stage1_row_end: milestone.stage1_row_end(),
        sizing_selected_device_ordinal: sizing.selected_device_ordinal(),
        milestone_selected_device_ordinal: milestone.selected_device_ordinal(),
        native_input_receipt_identity_sha256: native_identity,
        milestone_native_input_receipt_identity_sha256: milestone
            .native_input_receipt_identity_sha256()
            .to_owned(),
        population_sizing_receipt_identity_sha256: sizing.identity_sha256().to_owned(),
        milestone_population_sizing_receipt_identity_sha256: milestone
            .population_sizing_receipt_identity_sha256()
            .to_owned(),
        adaptive_base_effective_for_stage1: sizing.adaptive_base_effective_for_stage1(),
        sizing_resident_adaptive_request_identity_sha256: adaptive_request_identity,
        milestone_adaptive_token_identity_sha256: milestone.adaptive_token_identity_sha256(),
        metrics_receipt_identities_sha256: milestone.metrics_receipt_identities_sha256(),
        counters: milestone.residency_counters().into(),
        engine: milestone.engine(),
        consumer_completion_confirmed: milestone.consumer_completion_confirmed(),
        replay_identity_sealed: milestone.replay_identity_sealed(),
    })
}

#[cfg(feature = "gpu-cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn checked_seal_canonical_native_generation_zero_research_result_from_evidence_v1<'a>(
    preflight: CanonicalNativeGenerationZeroResultPreflightV1,
    request_evidence: CanonicalNativeGenerationZeroRequestEvidenceV1,
    financial_contract: CanonicalTrendbarResearchExecutionContractV3,
    native_receipt_v3: CanonicalGpuResidentSearchInputReceiptV3,
    sizing_receipt_v2: ResidentPopulationAutoSizingReceiptV2,
    evaluation_evidence: CanonicalNativeGenerationZeroEvaluationEvidenceV1,
    milestone: &'a ResidentGenerationZeroMilestoneV1,
) -> Result<
    (
        CanonicalNativeGenerationZeroResearchResultViewV1<'a>,
        CanonicalNativeGenerationZeroCompactJsonSealV1,
    ),
    CanonicalNativeGenerationZeroResultErrorV1,
> {
    validate_request_evidence_v1(&request_evidence)?;
    validate_contract_evidence_v1(
        &financial_contract,
        &request_evidence.contract_domain_identity_sha256,
        &request_evidence.contract_artifact_exact_file_sha256,
        &request_evidence.contract_domain_identity_sha256,
        &request_evidence.contract_artifact_expected_sha256,
    )?;
    let projection = crate::resident_population_auto_sizing_receipt_v2::
        canonical_pinned_source_projection_from_search_receipt_v1(
            financial_contract.input_receipt(),
        )
        .map_err(|error| invalid_result_v1(format!("financial source projection: {error}")))?;
    validate_native_source_projection_v1(&native_receipt_v3, &projection)?;
    sizing_receipt_v2
        .validate_self_v2()
        .map_err(|error| invalid_result_v1(format!("population sizing receipt: {error}")))?;
    sizing_receipt_v2
        .validate_financial_authority_against_pinned_source_projection_v2(
            &financial_contract,
            &projection,
        )
        .map_err(|error| invalid_result_v1(format!("population financial authority: {error}")))?;
    validate_evaluation_evidence_v1(&evaluation_evidence, &financial_contract)?;
    validate_preflight_authority_v1(
        &preflight,
        &request_evidence,
        &financial_contract,
        &native_receipt_v3,
        &sizing_receipt_v2,
    )?;
    let execution = checked_execution_facts_from_authorities_v1(
        &preflight,
        &native_receipt_v3,
        &sizing_receipt_v2,
        milestone,
    )?;
    validate_execution_facts_v1(&execution)?;
    let result = milestone.search_result();
    validate_population_payload_v1(
        &result.genes,
        &result.metrics,
        result.effective_smc_gate_threshold,
        sizing_receipt_v2.resolved_population(),
        preflight.prepared_feature_count(),
        preflight.term_cap(),
        evaluation_evidence.growth_objective(),
    )?;
    let gate = CanonicalNativeGenerationZeroEffectiveSmcGateEvidenceV1::checked_new(
        result.effective_smc_gate_threshold,
        EFFECTIVE_SMC_GATE_SOURCE_GENETIC_SEARCH_RUNTIME_START_GENERATION_ZERO_V1,
        &request_evidence.startup_settings_id,
        &request_evidence.runtime_install_receipt_id,
        &request_evidence.generation_zero_runtime_authority_id,
    )?;
    if gate.effective_smc_gate_threshold().to_bits()
        != result.effective_smc_gate_threshold.to_bits()
        || gate.source()
            != EFFECTIVE_SMC_GATE_SOURCE_GENETIC_SEARCH_RUNTIME_START_GENERATION_ZERO_V1
        || !is_canonical_lower_hex_sha256_v1(gate.identity_sha256())
    {
        return Err(invalid_result_v1("effective gate evidence drifted"));
    }

    let mut view = CanonicalNativeGenerationZeroResearchResultViewV1 {
        preflight,
        request_evidence,
        financial_execution_contract_v3: financial_contract,
        native_input_receipt_v3: native_receipt_v3,
        population_sizing_receipt_v2: sizing_receipt_v2,
        evaluation_evidence_v1: evaluation_evidence,
        milestone,
        evidence_identity_sha256: String::new(),
    };
    view.evidence_identity_sha256 = typed_identity_sha256_v1(
        CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_IDENTITY_DOMAIN_V1,
        &view.identity_material_v1(),
    )?;
    let empty_count = view.checked_fixed_metadata_with_empty_arrays_byte_count_v1()?;
    if empty_count
        > view
            .preflight
            .fixed_metadata_upper_bound_with_empty_arrays_bytes()
    {
        return Err(invalid_result_v1(
            "actual fixed metadata exceeds preflight bound",
        ));
    }
    let planned = view
        .preflight
        .checked_upper_bound_for_population(view.milestone.resolved_population())?;
    let mut sink = io::sink();
    let seal =
        write_canonical_native_generation_zero_research_result_v1(&view, &mut sink, planned)?;
    if seal.byte_count() > planned || planned > MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1 {
        return Err(invalid_result_v1("sealed result exceeds planned envelope"));
    }
    Ok((view, seal))
}

#[cfg(feature = "gpu-cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn seal_canonical_native_generation_zero_research_result_v1<'a>(
    request: &CanonicalNativeDiscoveryRequestV1,
    preflight: CanonicalNativeGenerationZeroResultPreflightV1,
    financial_contract: CanonicalTrendbarResearchExecutionContractV3,
    native_receipt_v3: CanonicalGpuResidentSearchInputReceiptV3,
    sizing_receipt_v2: ResidentPopulationAutoSizingReceiptV2,
    evaluation_config: EvaluationConfig,
    milestone: &'a ResidentGenerationZeroMilestoneV1,
) -> Result<
    (
        CanonicalNativeGenerationZeroResearchResultViewV1<'a>,
        CanonicalNativeGenerationZeroCompactJsonSealV1,
    ),
    CanonicalNativeGenerationZeroResultErrorV1,
> {
    let request_evidence = request_evidence_from_request_v1(request);
    let evaluation_evidence =
        CanonicalNativeGenerationZeroEvaluationEvidenceV1::checked_from_evaluation_config_v1(
            &evaluation_config,
            request.config().mode,
        )?;
    checked_seal_canonical_native_generation_zero_research_result_from_evidence_v1(
        preflight,
        request_evidence,
        financial_contract,
        native_receipt_v3,
        sizing_receipt_v2,
        evaluation_evidence,
        milestone,
    )
}
// END CANONICAL_NATIVE_GEN0_SEALED_VIEW_V1

// BEGIN CANONICAL_NATIVE_GEN0_STREAMING_WRITER_V1
#[cfg(feature = "gpu-cuda")]
struct CappedSha256WriterV1<'a, W: Write> {
    inner: &'a mut W,
    cap: u64,
    byte_count: u64,
    sha256: Sha256,
}

#[cfg(feature = "gpu-cuda")]
impl<W: Write> Write for CappedSha256WriterV1<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let requested = u64::try_from(bytes.len())
            .map_err(|_| io::Error::other("result write size overflow"))?;
        let next = self
            .byte_count
            .checked_add(requested)
            .ok_or_else(|| io::Error::other("result byte count overflow"))?;
        if next > self.cap {
            return Err(io::Error::other("result byte cap exceeded"));
        }
        let written = self.inner.write(bytes)?;
        self.byte_count = self
            .byte_count
            .checked_add(written as u64)
            .ok_or_else(|| io::Error::other("result byte count overflow"))?;
        self.sha256.update(&bytes[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(feature = "gpu-cuda")]
pub(crate) fn write_canonical_native_generation_zero_research_result_v1<W: Write>(
    view: &CanonicalNativeGenerationZeroResearchResultViewV1<'_>,
    writer: &mut W,
    byte_cap: u64,
) -> Result<
    CanonicalNativeGenerationZeroCompactJsonSealV1,
    CanonicalNativeGenerationZeroResultErrorV1,
> {
    let mut counted = CappedSha256WriterV1 {
        inner: writer,
        cap: byte_cap.min(MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1),
        byte_count: 0,
        sha256: Sha256::new(),
    };
    view.result_wire_v1()
        .serialize(&mut serde_json::Serializer::new(&mut counted))
        .map_err(|error| CanonicalNativeGenerationZeroResultErrorV1::Io(io::Error::other(error)))?;
    counted
        .flush()
        .map_err(CanonicalNativeGenerationZeroResultErrorV1::Io)?;
    Ok(CanonicalNativeGenerationZeroCompactJsonSealV1 {
        byte_count: counted.byte_count,
        sha256: format!("{:x}", counted.sha256.finalize()),
    })
}
// END CANONICAL_NATIVE_GEN0_STREAMING_WRITER_V1

#[cfg(all(test, feature = "gpu-cuda"))]
mod bounds_v1_tests {
    include!("canonical_native_generation_zero_result_bounds_v1_tests.rs");
}

#[cfg(all(test, feature = "gpu-cuda"))]
mod payload_v1_tests {
    include!("canonical_native_generation_zero_result_schema_v1_tests.rs");
}

#[cfg(all(test, feature = "gpu-cuda"))]
mod execution_receipt_v1_tests {
    include!("canonical_native_generation_zero_result_execution_receipt_v1_tests.rs");
}

#[cfg(all(test, feature = "gpu-cuda"))]
mod execution_receipt_continuation_v1_tests {
    include!("canonical_native_generation_zero_result_execution_receipt_continuation_v1_tests.rs");
}

#[cfg(all(test, feature = "gpu-cuda"))]
mod sealed_writer_v1_tests {
    include!("canonical_native_generation_zero_result_sealed_writer_v1_tests.rs");
}

#[cfg(all(test, feature = "gpu-cuda", target_os = "linux"))]
mod publication_v1_tests {
    include!("canonical_native_generation_zero_publication_v1_behavior_tests.rs");
}

#[cfg(all(test, feature = "gpu-cuda", target_os = "linux"))]
mod high_level_orchestration_v1_tests {
    include!("canonical_native_generation_zero_result_high_level_orchestration_v1_tests.rs");
}
