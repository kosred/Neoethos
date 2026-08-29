//! Checked pre-V5 size planning for the canonical native Generation-zero result.
//!
//! Future 2A2 supplies the checked fixed-metadata bound after prepared Data reveals F; no final sealer lives here.

#![allow(dead_code)] // Private pre-V5 authority is consumed by the future 2A2 sealer.

use std::fmt;

use crate::canonical_native_discovery_request_v1::{
    MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1,
    MAX_CANONICAL_NATIVE_GEN0_RESOLVED_POPULATION_V1, MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
    MAX_CANONICAL_NATIVE_GEN0_TERMS_V1,
};

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
