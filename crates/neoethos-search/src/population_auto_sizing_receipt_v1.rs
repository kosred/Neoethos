//! Immutable, admission-bound population-auto sizing evidence.
//!
//! The receipt is minted from the exact run-owned device route. It never probes
//! a device and never samples free memory: the native route carries the one
//! pre-parent snapshot captured by strict admission.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const POPULATION_AUTO_SIZING_RECEIPT_SCHEMA_VERSION_V1: u16 = 1;
pub const POPULATION_AUTO_HARD_GROWTH_CAP_V1: usize = 16_384;
pub const POPULATION_AUTO_ALLOCATOR_RESERVE_BYTES_V1: u64 = 64 * 1024 * 1024;
#[cfg(all(test, feature = "gpu-b-adapter"))]
pub(crate) const QUALITY_SCREEN_MAX_STAGED_CLONES_V1: usize = 131_072;
#[cfg(all(test, feature = "gpu-b-adapter"))]
pub(crate) const QUALITY_SCREEN_MAX_STAGED_BASE_GENES_V1: usize = 131_072;
#[cfg(all(test, feature = "gpu-b-adapter"))]
pub(crate) const QUALITY_SCREEN_MAX_STAGED_SCENARIOS_V1: usize = 131_072;

const RECEIPT_HASH_DOMAIN_V1: &[u8] = b"neoethos.search.population-auto-sizing-receipt.v1\0";
const STAGE1_HASH_DOMAIN_V1: &[u8] = b"neoethos.search.population-auto-stage1-window.v1\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PopulationAutoSizingErrorCodeV1 {
    InvalidInput,
    ArithmeticOverflow,
    NativePlanUnavailable,
    UnboundedMigrationTerms,
    ParentNoRoom,
    GeneNoRoom,
    ScenarioNoRoom,
    QualityScreenChunkNoRoom,
    UnsupportedSchema,
    IdentityMismatch,
    InvalidReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PopulationAutoSizingErrorV1 {
    code: PopulationAutoSizingErrorCodeV1,
    message: String,
}

impl PopulationAutoSizingErrorV1 {
    pub const fn code(&self) -> PopulationAutoSizingErrorCodeV1 {
        self.code
    }
}

impl fmt::Display for PopulationAutoSizingErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PopulationAutoSizingErrorV1 {}

pub(crate) fn sizing_error_v1(
    code: PopulationAutoSizingErrorCodeV1,
    message: impl Into<String>,
) -> PopulationAutoSizingErrorV1 {
    PopulationAutoSizingErrorV1 {
        code,
        message: message.into(),
    }
}

fn checked_u64(value: usize, name: &'static str) -> Result<u64, PopulationAutoSizingErrorV1> {
    u64::try_from(value).map_err(|_| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            format!("{name} does not fit the versioned u64 sizing receipt"),
        )
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().any(|byte| byte != b'0')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn hash_serialized_v1<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, PopulationAutoSizingErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|source| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::InvalidReceipt,
            format!("serialize population sizing identity: {source}"),
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Ok(hex_lower(&hasher.finalize()))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopulationAutoStage1WindowV1 {
    role: String,
    row_start: u64,
    row_end: u64,
    identity_sha256: String,
}

impl PopulationAutoStage1WindowV1 {
    pub fn role(&self) -> &str {
        &self.role
    }

    pub const fn row_start(&self) -> u64 {
        self.row_start
    }

    pub const fn row_end(&self) -> u64 {
        self.row_end
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }
}

#[derive(Serialize)]
struct Stage1WindowIdentityBodyV1<'a> {
    parent_dataset_identity_sha256: &'a str,
    role: &'a str,
    row_start: u64,
    row_end: u64,
}

pub(crate) fn seal_population_auto_stage1_window_v1(
    parent_dataset_identity_sha256: &str,
    role: &str,
    row_start: usize,
    row_end: usize,
) -> Result<PopulationAutoStage1WindowV1, PopulationAutoSizingErrorV1> {
    if !is_sha256(parent_dataset_identity_sha256) || role.is_empty() || row_end <= row_start {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::InvalidInput,
            "population-auto stage1 window requires a parent SHA-256, non-empty role, and non-empty ordered range",
        ));
    }
    let row_start = checked_u64(row_start, "stage1 row start")?;
    let row_end = checked_u64(row_end, "stage1 row end")?;
    let body = Stage1WindowIdentityBodyV1 {
        parent_dataset_identity_sha256,
        role,
        row_start,
        row_end,
    };
    Ok(PopulationAutoStage1WindowV1 {
        role: role.to_owned(),
        row_start,
        row_end,
        identity_sha256: hash_serialized_v1(STAGE1_HASH_DOMAIN_V1, &body)?,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PopulationAutoCpuAuthorityV1 {
    LegacyCudaZero {
        probe_receipt_identity_sha256: String,
    },
    PhysicalGpuAbsence {
        platform: String,
        inventory_identity_sha256: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PopulationAutoSizingRouteV1 {
    NativeCuda {
        selected_ordinal: u32,
        pre_parent_free_memory_bytes: u64,
        cuda_device_identity_sha256: String,
        cuda_build_manifest_sha256: String,
        probe_receipt_identity_sha256: String,
    },
    CpuNoCompatibleGpu {
        authority: PopulationAutoCpuAuthorityV1,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct PopulationAutoSizingRequestV1 {
    pub(crate) population_auto: bool,
    pub(crate) configured_population: usize,
    pub(crate) resident_parent_rows: usize,
    pub(crate) evaluation_rows: usize,
    pub(crate) feature_count: usize,
    pub(crate) month_capacity: usize,
    pub(crate) requested_max_indicators: usize,
    pub(crate) migration_enabled: bool,
    pub(crate) parent_canonical_scope_identity_sha256: String,
    pub(crate) parent_dataset_identity_sha256: String,
    pub(crate) stage1_window: PopulationAutoStage1WindowV1,
    pub(crate) route: PopulationAutoSizingRouteV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativePopulationAutoPlanFactsV1 {
    pub(crate) admitted_budget_bytes: u64,
    pub(crate) parent_device_bytes: u64,
    pub(crate) gene_bytes_per_candidate_at_term_cap: u64,
    pub(crate) gene_fixed_overhead_bytes: u64,
    pub(crate) scenario_device_bytes_per_candidate: u64,
    pub(crate) configured_gene_device_bytes: u64,
    pub(crate) configured_scenario_device_bytes: u64,
    pub(crate) fixed_gene_capacity: usize,
    pub(crate) memory_population_cap: usize,
    pub(crate) raw_time_cap: usize,
    pub(crate) effective_time_cap: usize,
    pub(crate) occupancy_floor_overrode_time_target: bool,
    pub(crate) hard_growth_cap: usize,
    pub(crate) growth_cap: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopulationAutoSizingReceiptV1 {
    schema_version: u16,
    population_auto: bool,
    configured_population: u64,
    resolved_population: u64,
    resident_parent_rows: u64,
    evaluation_rows: u64,
    feature_count: u64,
    month_capacity: u64,
    requested_max_indicators: u64,
    term_cap: u64,
    term_cap_authority: String,
    migration_enabled_for_run: bool,
    migration_policy: String,
    parent_canonical_scope_identity_sha256: String,
    parent_dataset_identity_sha256: String,
    stage1_window: PopulationAutoStage1WindowV1,
    route: PopulationAutoSizingRouteV1,
    admitted_budget_bytes: u64,
    allocator_reserve_bytes: u64,
    parent_device_bytes: u64,
    gene_bytes_per_candidate_at_term_cap: u64,
    gene_fixed_overhead_bytes: u64,
    scenario_device_bytes_per_candidate: u64,
    configured_gene_device_bytes: u64,
    resolved_gene_device_bytes: u64,
    configured_scenario_device_bytes: u64,
    resolved_scenario_device_bytes: u64,
    fixed_gene_capacity: u64,
    memory_population_cap: u64,
    raw_time_cap: u64,
    effective_time_cap: u64,
    occupancy_floor_overrode_time_target: bool,
    hard_growth_cap: u64,
    growth_cap: u64,
    resolution_reason: String,
    identity_sha256: String,
}

impl PopulationAutoSizingReceiptV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn population_auto(&self) -> bool {
        self.population_auto
    }

    pub fn configured_population(&self) -> usize {
        usize::try_from(self.configured_population).expect("validated receipt population")
    }

    pub fn resolved_population(&self) -> usize {
        usize::try_from(self.resolved_population).expect("validated receipt population")
    }

    pub fn resident_parent_rows(&self) -> usize {
        usize::try_from(self.resident_parent_rows).expect("validated receipt parent rows")
    }

    pub fn evaluation_rows(&self) -> usize {
        usize::try_from(self.evaluation_rows).expect("validated receipt evaluation rows")
    }

    pub fn feature_count(&self) -> usize {
        usize::try_from(self.feature_count).expect("validated receipt feature count")
    }

    pub fn month_capacity(&self) -> usize {
        usize::try_from(self.month_capacity).expect("validated receipt month capacity")
    }

    pub fn requested_max_indicators(&self) -> usize {
        usize::try_from(self.requested_max_indicators)
            .expect("validated receipt requested max indicators")
    }

    pub fn term_cap(&self) -> usize {
        usize::try_from(self.term_cap).expect("validated receipt term cap")
    }

    pub const fn migration_enabled_for_run(&self) -> bool {
        self.migration_enabled_for_run
    }

    pub fn parent_canonical_scope_identity_sha256(&self) -> &str {
        &self.parent_canonical_scope_identity_sha256
    }

    pub fn parent_dataset_identity_sha256(&self) -> &str {
        &self.parent_dataset_identity_sha256
    }

    pub const fn parent_device_bytes(&self) -> u64 {
        self.parent_device_bytes
    }

    pub const fn scenario_device_bytes_per_candidate(&self) -> u64 {
        self.scenario_device_bytes_per_candidate
    }

    pub fn fixed_gene_capacity(&self) -> usize {
        usize::try_from(self.fixed_gene_capacity).expect("validated receipt gene capacity")
    }

    pub fn memory_population_cap(&self) -> usize {
        usize::try_from(self.memory_population_cap).expect("validated receipt memory cap")
    }

    pub fn raw_time_cap(&self) -> usize {
        usize::try_from(self.raw_time_cap).expect("validated receipt time cap")
    }

    pub fn effective_time_cap(&self) -> usize {
        usize::try_from(self.effective_time_cap).expect("validated receipt time cap")
    }

    pub const fn occupancy_floor_overrode_time_target(&self) -> bool {
        self.occupancy_floor_overrode_time_target
    }

    pub fn hard_growth_cap(&self) -> usize {
        usize::try_from(self.hard_growth_cap).expect("validated receipt hard cap")
    }

    pub fn resolution_reason(&self) -> &str {
        &self.resolution_reason
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub const fn stage1_window(&self) -> &PopulationAutoStage1WindowV1 {
        &self.stage1_window
    }

    pub const fn route(&self) -> &PopulationAutoSizingRouteV1 {
        &self.route
    }

    fn computed_identity_sha256(&self) -> Result<String, PopulationAutoSizingErrorV1> {
        let mut body = self.clone();
        body.identity_sha256.clear();
        hash_serialized_v1(RECEIPT_HASH_DOMAIN_V1, &body)
    }

    pub fn validate(&self) -> Result<(), PopulationAutoSizingErrorV1> {
        if self.schema_version != POPULATION_AUTO_SIZING_RECEIPT_SCHEMA_VERSION_V1 {
            return Err(sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::UnsupportedSchema,
                format!(
                    "unsupported population-auto sizing receipt schema {}",
                    self.schema_version
                ),
            ));
        }
        for (name, value) in [
            (
                "parent canonical scope",
                self.parent_canonical_scope_identity_sha256.as_str(),
            ),
            (
                "parent dataset",
                self.parent_dataset_identity_sha256.as_str(),
            ),
            ("stage1 window", self.stage1_window.identity_sha256()),
            ("receipt", self.identity_sha256.as_str()),
        ] {
            if !is_sha256(value) {
                return Err(sizing_error_v1(
                    PopulationAutoSizingErrorCodeV1::InvalidReceipt,
                    format!("{name} identity is not a lowercase SHA-256"),
                ));
            }
        }
        let to_usize = |name: &'static str, value: u64| {
            usize::try_from(value).map_err(|_| {
                sizing_error_v1(
                    PopulationAutoSizingErrorCodeV1::InvalidReceipt,
                    format!("{name} does not fit this process"),
                )
            })
        };
        let configured_population = to_usize("configured population", self.configured_population)?;
        let resident_parent_rows = to_usize("resident parent rows", self.resident_parent_rows)?;
        let evaluation_rows = to_usize("evaluation rows", self.evaluation_rows)?;
        let feature_count = to_usize("feature count", self.feature_count)?;
        let month_capacity = to_usize("month capacity", self.month_capacity)?;
        let requested_max_indicators =
            to_usize("requested max indicators", self.requested_max_indicators)?;
        if configured_population == 0
            || resident_parent_rows == 0
            || evaluation_rows == 0
            || feature_count == 0
            || month_capacity == 0
        {
            return Err(sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::InvalidReceipt,
                "population-auto sizing receipt has a zero primitive extent",
            ));
        }
        match &self.route {
            PopulationAutoSizingRouteV1::NativeCuda {
                pre_parent_free_memory_bytes,
                cuda_device_identity_sha256,
                cuda_build_manifest_sha256,
                probe_receipt_identity_sha256,
                ..
            } => {
                if *pre_parent_free_memory_bytes == 0
                    || !is_sha256(cuda_device_identity_sha256)
                    || !is_sha256(cuda_build_manifest_sha256)
                    || !is_sha256(probe_receipt_identity_sha256)
                    || self.admitted_budget_bytes == 0
                    || self.parent_device_bytes == 0
                    || self.gene_bytes_per_candidate_at_term_cap == 0
                    || self.gene_fixed_overhead_bytes == 0
                    || self.scenario_device_bytes_per_candidate == 0
                    || self.fixed_gene_capacity == 0
                {
                    return Err(sizing_error_v1(
                        PopulationAutoSizingErrorCodeV1::InvalidReceipt,
                        "native population-auto sizing receipt has inconsistent plan facts",
                    ));
                }
            }
            PopulationAutoSizingRouteV1::CpuNoCompatibleGpu { authority } => {
                let identity = match authority {
                    PopulationAutoCpuAuthorityV1::LegacyCudaZero {
                        probe_receipt_identity_sha256,
                    } => probe_receipt_identity_sha256,
                    PopulationAutoCpuAuthorityV1::PhysicalGpuAbsence {
                        platform,
                        inventory_identity_sha256,
                    } => {
                        if platform.is_empty() {
                            return Err(sizing_error_v1(
                                PopulationAutoSizingErrorCodeV1::InvalidReceipt,
                                "physical-GPU absence receipt has an empty platform",
                            ));
                        }
                        inventory_identity_sha256
                    }
                };
                if !is_sha256(identity) {
                    return Err(sizing_error_v1(
                        PopulationAutoSizingErrorCodeV1::InvalidReceipt,
                        "CPU route authority is not a lowercase SHA-256",
                    ));
                }
            }
        }
        let expected = self.computed_identity_sha256()?;
        if expected != self.identity_sha256 {
            return Err(sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::IdentityMismatch,
                "population-auto sizing receipt identity does not match its bound fields",
            ));
        }
        let primitive_request = PopulationAutoSizingRequestV1 {
            population_auto: self.population_auto,
            configured_population,
            resident_parent_rows,
            evaluation_rows,
            feature_count,
            month_capacity,
            requested_max_indicators,
            migration_enabled: self.migration_enabled_for_run,
            parent_canonical_scope_identity_sha256: self
                .parent_canonical_scope_identity_sha256
                .clone(),
            parent_dataset_identity_sha256: self.parent_dataset_identity_sha256.clone(),
            stage1_window: self.stage1_window.clone(),
            route: self.route.clone(),
        };
        let rebuilt = build_population_auto_sizing_receipt_v1(primitive_request)?;
        if rebuilt != *self {
            return Err(sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::InvalidReceipt,
                "population-auto sizing receipt derived facts do not match its primitive admission inputs",
            ));
        }
        Ok(())
    }
}

fn validate_request_v1(
    request: &PopulationAutoSizingRequestV1,
) -> Result<usize, PopulationAutoSizingErrorV1> {
    if request.configured_population == 0
        || request.resident_parent_rows == 0
        || request.evaluation_rows == 0
        || request.feature_count == 0
        || request.month_capacity == 0
        || !is_sha256(&request.parent_canonical_scope_identity_sha256)
        || !is_sha256(&request.parent_dataset_identity_sha256)
    {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::InvalidInput,
            "population-auto sizing requires non-zero extents and exact parent identities",
        ));
    }
    let stage1_rows = request
        .stage1_window
        .row_end
        .checked_sub(request.stage1_window.row_start)
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::InvalidInput,
                "population-auto stage1 range is reversed",
            )
        })?;
    if stage1_rows != checked_u64(request.evaluation_rows, "evaluation rows")? {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::InvalidInput,
            "population-auto stage1 identity does not match the evaluation extent",
        ));
    }
    if request.stage1_window.role() != "selection_stage1"
        || request.stage1_window.row_end
            > checked_u64(request.resident_parent_rows, "resident parent rows")?
    {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::InvalidInput,
            "population-auto stage1 window must be the selection_stage1 role inside the resident parent",
        ));
    }
    let expected_stage1 = seal_population_auto_stage1_window_v1(
        &request.parent_dataset_identity_sha256,
        request.stage1_window.role(),
        usize::try_from(request.stage1_window.row_start).map_err(|_| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "stage1 start does not fit this process",
            )
        })?,
        usize::try_from(request.stage1_window.row_end).map_err(|_| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "stage1 end does not fit this process",
            )
        })?,
    )?;
    if expected_stage1 != request.stage1_window {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::InvalidInput,
            "population-auto stage1 window identity is detached from the parent/range",
        ));
    }
    let template_floor = crate::genetic::seed_templates::PROFESSIONAL_TEMPLATE_MAX_TERMS_V1;
    Ok(request
        .feature_count
        .min(request.requested_max_indicators.max(template_floor)))
}

pub(crate) fn seal_population_auto_sizing_receipt_v1(
    request: PopulationAutoSizingRequestV1,
) -> Result<PopulationAutoSizingReceiptV1, PopulationAutoSizingErrorV1> {
    let receipt = build_population_auto_sizing_receipt_v1(request)?;
    receipt.validate()?;
    Ok(receipt)
}

fn build_population_auto_sizing_receipt_v1(
    request: PopulationAutoSizingRequestV1,
) -> Result<PopulationAutoSizingReceiptV1, PopulationAutoSizingErrorV1> {
    let term_cap = validate_request_v1(&request)?;
    if request.migration_enabled {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::UnboundedMigrationTerms,
            "every receipt-governed search requires run-scoped migration to be disabled until migrant term extents are enforced before every ingest and upload",
        ));
    }

    let native_plan: Option<NativePopulationAutoPlanFactsV1> = match &request.route {
        PopulationAutoSizingRouteV1::NativeCuda {
            pre_parent_free_memory_bytes,
            ..
        } => {
            #[cfg(feature = "gpu-b-adapter")]
            {
                Some(
                    crate::gpu_native::prototype_b_population_eval::population_auto_plan_for_pre_parent_free_memory_v1(
                        *pre_parent_free_memory_bytes,
                        request.resident_parent_rows,
                        request.evaluation_rows,
                        request.feature_count,
                        request.month_capacity,
                        request.configured_population,
                        term_cap,
                    )?,
                )
            }
            #[cfg(not(feature = "gpu-b-adapter"))]
            {
                let _ = pre_parent_free_memory_bytes;
                return Err(sizing_error_v1(
                    PopulationAutoSizingErrorCodeV1::NativePlanUnavailable,
                    "cannot mint a native population-auto receipt without the reviewed native planner",
                ));
            }
        }
        PopulationAutoSizingRouteV1::CpuNoCompatibleGpu { .. } => None,
    };

    // The occupancy floor and hard growth cap are both 16,384 today. Thus a
    // raw 20-second target below that knee is evidence (and an explicit
    // override flag), not a smaller growth limit. Memory still wins below it.
    let (resolved_population, resolution_reason) = match native_plan {
        Some(plan) if request.population_auto => {
            let resolved = request.configured_population.max(plan.growth_cap);
            let reason = if resolved > request.configured_population {
                "native_cuda_auto_grew"
            } else if request.configured_population > plan.growth_cap {
                "native_cuda_configured_above_growth_cap_no_shrink"
            } else {
                "native_cuda_configured_at_growth_cap"
            };
            (resolved, reason)
        }
        Some(_) => (request.configured_population, "auto_disabled"),
        None if request.population_auto => (request.configured_population, "cpu_no_compatible_gpu"),
        None => (request.configured_population, "auto_disabled"),
    };

    let (resolved_gene_device_bytes, resolved_scenario_device_bytes) = match native_plan {
        Some(_) => {
            #[cfg(feature = "gpu-b-adapter")]
            {
                crate::gpu_native::prototype_b_population_eval::population_auto_resolved_bytes_v1(
                    resolved_population,
                    term_cap,
                    request.month_capacity,
                )?
            }
            #[cfg(not(feature = "gpu-b-adapter"))]
            unreachable!("native plan cannot exist without the native adapter")
        }
        None => (0, 0),
    };

    let plan = native_plan.unwrap_or(NativePopulationAutoPlanFactsV1 {
        admitted_budget_bytes: 0,
        parent_device_bytes: 0,
        gene_bytes_per_candidate_at_term_cap: 0,
        gene_fixed_overhead_bytes: 0,
        scenario_device_bytes_per_candidate: 0,
        configured_gene_device_bytes: 0,
        configured_scenario_device_bytes: 0,
        fixed_gene_capacity: 0,
        memory_population_cap: 0,
        raw_time_cap: 0,
        effective_time_cap: 0,
        occupancy_floor_overrode_time_target: false,
        hard_growth_cap: POPULATION_AUTO_HARD_GROWTH_CAP_V1,
        growth_cap: 0,
    });
    let mut receipt = PopulationAutoSizingReceiptV1 {
        schema_version: POPULATION_AUTO_SIZING_RECEIPT_SCHEMA_VERSION_V1,
        population_auto: request.population_auto,
        configured_population: checked_u64(request.configured_population, "configured population")?,
        resolved_population: checked_u64(resolved_population, "resolved population")?,
        resident_parent_rows: checked_u64(request.resident_parent_rows, "resident parent rows")?,
        evaluation_rows: checked_u64(request.evaluation_rows, "evaluation rows")?,
        feature_count: checked_u64(request.feature_count, "feature count")?,
        month_capacity: checked_u64(request.month_capacity, "month capacity")?,
        requested_max_indicators: checked_u64(
            request.requested_max_indicators,
            "requested max indicators",
        )?,
        term_cap: checked_u64(term_cap, "term cap")?,
        term_cap_authority:
            "min(feature_count,max(requested_max_indicators,professional_template_max_terms_v1))"
                .to_owned(),
        migration_enabled_for_run: request.migration_enabled,
        migration_policy: if request.population_auto {
            "run_scoped_disabled_for_population_auto"
        } else {
            "run_scoped_configured_value"
        }
        .to_owned(),
        parent_canonical_scope_identity_sha256: request.parent_canonical_scope_identity_sha256,
        parent_dataset_identity_sha256: request.parent_dataset_identity_sha256,
        stage1_window: request.stage1_window,
        route: request.route,
        admitted_budget_bytes: plan.admitted_budget_bytes,
        allocator_reserve_bytes: if native_plan.is_some() {
            POPULATION_AUTO_ALLOCATOR_RESERVE_BYTES_V1
        } else {
            0
        },
        parent_device_bytes: plan.parent_device_bytes,
        gene_bytes_per_candidate_at_term_cap: plan.gene_bytes_per_candidate_at_term_cap,
        gene_fixed_overhead_bytes: plan.gene_fixed_overhead_bytes,
        scenario_device_bytes_per_candidate: plan.scenario_device_bytes_per_candidate,
        configured_gene_device_bytes: plan.configured_gene_device_bytes,
        resolved_gene_device_bytes,
        configured_scenario_device_bytes: plan.configured_scenario_device_bytes,
        resolved_scenario_device_bytes,
        fixed_gene_capacity: checked_u64(plan.fixed_gene_capacity, "fixed gene capacity")?,
        memory_population_cap: checked_u64(plan.memory_population_cap, "memory population cap")?,
        raw_time_cap: checked_u64(plan.raw_time_cap, "raw time cap")?,
        effective_time_cap: checked_u64(plan.effective_time_cap, "effective time cap")?,
        occupancy_floor_overrode_time_target: plan.occupancy_floor_overrode_time_target,
        hard_growth_cap: checked_u64(plan.hard_growth_cap, "hard growth cap")?,
        growth_cap: checked_u64(plan.growth_cap, "growth cap")?,
        resolution_reason: resolution_reason.to_owned(),
        identity_sha256: String::new(),
    };
    receipt.identity_sha256 = receipt.computed_identity_sha256()?;
    Ok(receipt)
}

#[cfg(all(test, feature = "gpu-b-adapter"))]
pub(crate) fn recompute_population_auto_receipt_identity_for_test_v1(
    receipt: &mut PopulationAutoSizingReceiptV1,
) -> Result<(), PopulationAutoSizingErrorV1> {
    receipt.identity_sha256 = receipt.computed_identity_sha256()?;
    Ok(())
}

/// Bound one quality-screen launch before either genes or descriptors are
/// materialised. The evaluator may split descriptors later, but that cannot
/// repair an already-allocated host Vec or an unsplittable gene store.
#[cfg(all(test, feature = "gpu-b-adapter"))]
pub(crate) fn quality_screen_candidate_chunk_v1(
    receipt: &PopulationAutoSizingReceiptV1,
    candidates: usize,
    mc_runs: usize,
    device_monte_carlo: bool,
    fused_cost_scenario: bool,
) -> Result<usize, PopulationAutoSizingErrorV1> {
    receipt.validate()?;
    if candidates == 0 {
        return Ok(0);
    }
    let scenario_multiplier = mc_runs
        .checked_add(usize::from(fused_cost_scenario))
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "quality-screen scenario multiplier overflows usize",
            )
        })?;
    let gene_multiplier = if device_monte_carlo || mc_runs == 0 {
        1
    } else {
        mc_runs.checked_add(1).ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "quality-screen gene multiplier overflows usize",
            )
        })?
    };
    let route_gene_capacity = match receipt.route() {
        PopulationAutoSizingRouteV1::NativeCuda { .. } => receipt.fixed_gene_capacity(),
        PopulationAutoSizingRouteV1::CpuNoCompatibleGpu { .. } => usize::MAX,
    };
    let mut chunk = candidates
        .min(QUALITY_SCREEN_MAX_STAGED_BASE_GENES_V1)
        .min(route_gene_capacity / gene_multiplier);
    if !device_monte_carlo && mc_runs > 0 {
        chunk = chunk.min(QUALITY_SCREEN_MAX_STAGED_CLONES_V1 / mc_runs);
    }
    if scenario_multiplier > 0 {
        chunk = chunk.min(QUALITY_SCREEN_MAX_STAGED_SCENARIOS_V1 / scenario_multiplier);
    }
    if chunk == 0 {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::QualityScreenChunkNoRoom,
            format!(
                "one quality-screen candidate needs {gene_multiplier} resident genes and {scenario_multiplier} staged scenarios, beyond the sealed capacities"
            ),
        ));
    }
    Ok(chunk)
}
