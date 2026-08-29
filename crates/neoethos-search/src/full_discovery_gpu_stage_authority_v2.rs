//! Opaque authority for an exact, strict-GPU full Discovery stage set.
//!
//! This module deliberately has no crate export or construction API yet.  The
//! sixteen downstream stage implementations must eventually seal the two input
//! authorities below; callers cannot replace those receipts with booleans,
//! ordinals, byte counts, or hashes.  Until that wiring exists, no permit can be
//! minted and no partial GPU pipeline can present itself as full Discovery.

use crate::gpu_native::capability::{GpuCapabilityManifest, PipelineStage, StageGpuCapability};
use sha2::{Digest, Sha256};
use thiserror::Error;

const FULL_DISCOVERY_STAGE_COUNT_V2: usize = 16;
const SHA256_HEX_LENGTH_V2: usize = 64;
const FULL_DISCOVERY_STAGE_MANIFEST_DOMAIN_V2: &[u8] =
    b"neoethos.search.full-discovery-stage-manifest.v2\0";
const STRICT_GPU_ONLY_FULL_DISCOVERY_PERMIT_DOMAIN_V2: &[u8] =
    b"neoethos.search.strict-gpu-only-full-discovery-permit.v2\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrictGpuRouteRefusalV2 {
    CpuRoute,
    CpuAllowed,
}

/// Sealed by the future resident-input completion boundary.  Private fields and
/// the absence of a constructor prevent App, CLI, or config from minting device
/// or input evidence.
#[derive(Debug)]
pub struct SealedResidentFullDiscoveryInputAuthorityV2 {
    route_refusal: Option<StrictGpuRouteRefusalV2>,
    selected_cuda_ordinal: Option<u32>,
    cuda_device_identity_sha256: String,
    cuda_build_manifest_sha256: String,
    canonical_search_input_receipt_sha256: String,
    resident_input_content_sha256: String,
    full_discovery_workspace_plan_identity_sha256: String,
}

/// Sealed only after every stage-owned preflight exists.  Stage identities are
/// in `PipelineStage::FULL_DISCOVERY` order and cannot be supplied independently
/// by a frontend.
#[derive(Debug)]
pub struct ResolvedFullDiscoveryPlanAuthorityV2 {
    stage_capabilities: GpuCapabilityManifest,
    ordered_stage_identity_sha256: [String; FULL_DISCOVERY_STAGE_COUNT_V2],
    selected_cuda_ordinal: u32,
    cuda_device_identity_sha256: String,
    cuda_build_manifest_sha256: String,
    canonical_search_input_receipt_sha256: String,
    resident_input_content_sha256: String,
    full_discovery_workspace_plan_identity_sha256: String,
    resolved_search_config_sha256: String,
    strategy_gene_schema_sha256: String,
    fitness_ordering_semantics_sha256: String,
    crossover_semantics_sha256: String,
    mutation_semantics_sha256: String,
    strategy_metrics_semantics_sha256: String,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FullDiscoveryGpuStageAuthorityErrorV2 {
    #[error("full Discovery stage count does not equal the canonical sixteen-stage set")]
    FullDiscoveryStageCountMismatch,
    #[error("full Discovery capability manifest is missing stage {0:?}")]
    MissingStageCapability(PipelineStage),
    #[error("full Discovery stage {stage:?} is not strict GPU: {capability:?}")]
    NonStrictFullDiscoveryStage {
        stage: PipelineStage,
        capability: StageGpuCapability,
    },
    #[error("card-present full Discovery cannot execute a CPU route")]
    CardPresentCpuExecutionForbidden,
    #[error("card-present full Discovery cannot authorize CPU fallback")]
    CardPresentAllowCpuForbidden,
    #[error("strict full Discovery requires one exact CUDA ordinal")]
    ExactCudaOrdinalRequired,
    #[error("full Discovery authority identity is malformed: {0}")]
    InvalidAuthorityIdentity(&'static str),
    #[error("resident input and resolved full Discovery plan disagree on {0}")]
    ResidentInputPlanMismatch(&'static str),
}

#[derive(Debug)]
struct ExactOrdinalDeviceBuildInputBindingV2 {
    selected_cuda_ordinal: u32,
    cuda_device_identity_sha256: String,
    cuda_build_manifest_sha256: String,
    canonical_search_input_receipt_sha256: String,
    resident_input_content_sha256: String,
    full_discovery_workspace_plan_identity_sha256: String,
}

/// Move-only proof consumed by the first resident generation stage.  It is
/// intentionally public only within this unexported module boundary; fields
/// remain private and there is no default, deserializer, or caller constructor.
#[derive(Debug)]
pub struct StrictGpuOnlyFullDiscoveryPermitV2 {
    selected_cuda_ordinal: u32,
    cuda_device_identity_sha256: String,
    cuda_build_manifest_sha256: String,
    canonical_search_input_receipt_sha256: String,
    resident_input_content_sha256: String,
    resolved_search_config_sha256: String,
    full_discovery_stage_manifest_sha256: String,
    full_discovery_workspace_plan_identity_sha256: String,
    strategy_gene_schema_sha256: String,
    fitness_ordering_semantics_sha256: String,
    crossover_semantics_sha256: String,
    mutation_semantics_sha256: String,
    strategy_metrics_semantics_sha256: String,
    permit_identity_sha256: String,
}

impl StrictGpuOnlyFullDiscoveryPermitV2 {
    pub const fn selected_cuda_ordinal(&self) -> Option<u32> {
        Some(self.selected_cuda_ordinal)
    }

    pub fn cuda_device_identity_sha256(&self) -> &str {
        &self.cuda_device_identity_sha256
    }

    pub fn cuda_build_manifest_sha256(&self) -> &str {
        &self.cuda_build_manifest_sha256
    }

    pub fn canonical_search_input_receipt_sha256(&self) -> &str {
        &self.canonical_search_input_receipt_sha256
    }

    pub fn resident_input_content_sha256(&self) -> &str {
        &self.resident_input_content_sha256
    }

    pub fn resolved_search_config_sha256(&self) -> &str {
        &self.resolved_search_config_sha256
    }

    pub fn full_discovery_stage_manifest_sha256(&self) -> &str {
        &self.full_discovery_stage_manifest_sha256
    }

    pub fn full_discovery_workspace_plan_identity_sha256(&self) -> &str {
        &self.full_discovery_workspace_plan_identity_sha256
    }

    pub fn strategy_gene_schema_sha256(&self) -> &str {
        &self.strategy_gene_schema_sha256
    }

    pub fn fitness_ordering_semantics_sha256(&self) -> &str {
        &self.fitness_ordering_semantics_sha256
    }

    pub fn crossover_semantics_sha256(&self) -> &str {
        &self.crossover_semantics_sha256
    }

    pub fn mutation_semantics_sha256(&self) -> &str {
        &self.mutation_semantics_sha256
    }

    pub fn strategy_metrics_semantics_sha256(&self) -> &str {
        &self.strategy_metrics_semantics_sha256
    }

    pub fn permit_identity_sha256(&self) -> &str {
        &self.permit_identity_sha256
    }

    pub const fn require_card_present_strict_gpu_v2(&self) -> Result<(), StrictGpuRouteRefusalV2> {
        Ok(())
    }
}

pub fn acquire_strict_gpu_only_full_discovery_permit_v2(
    sealed_resident_input_authority: SealedResidentFullDiscoveryInputAuthorityV2,
    resolved_full_discovery_plan_authority: ResolvedFullDiscoveryPlanAuthorityV2,
) -> Result<StrictGpuOnlyFullDiscoveryPermitV2, FullDiscoveryGpuStageAuthorityErrorV2> {
    reject_card_present_cpu_or_fallback_v2(&sealed_resident_input_authority)?;
    require_all_full_discovery_stages_strict_gpu_v2(
        &resolved_full_discovery_plan_authority.stage_capabilities,
    )?;
    let binding = bind_exact_ordinal_device_build_and_input_v2(
        &sealed_resident_input_authority,
        &resolved_full_discovery_plan_authority,
    )?;
    validate_resolved_plan_against_resident_input_v2(
        &binding,
        &resolved_full_discovery_plan_authority,
    )?;
    let full_discovery_stage_manifest_sha256 = compute_full_discovery_stage_manifest_identity_v2(
        &resolved_full_discovery_plan_authority.stage_capabilities,
        &resolved_full_discovery_plan_authority.ordered_stage_identity_sha256,
    )?;
    validate_plan_semantics_identities_v2(&resolved_full_discovery_plan_authority)?;
    let permit_identity_sha256 = compute_strict_gpu_only_full_discovery_permit_identity_v2(
        &binding,
        &resolved_full_discovery_plan_authority,
        &full_discovery_stage_manifest_sha256,
    );

    Ok(StrictGpuOnlyFullDiscoveryPermitV2 {
        selected_cuda_ordinal: binding.selected_cuda_ordinal,
        cuda_device_identity_sha256: binding.cuda_device_identity_sha256,
        cuda_build_manifest_sha256: binding.cuda_build_manifest_sha256,
        canonical_search_input_receipt_sha256: binding.canonical_search_input_receipt_sha256,
        resident_input_content_sha256: binding.resident_input_content_sha256,
        resolved_search_config_sha256: resolved_full_discovery_plan_authority
            .resolved_search_config_sha256,
        full_discovery_stage_manifest_sha256,
        full_discovery_workspace_plan_identity_sha256: binding
            .full_discovery_workspace_plan_identity_sha256,
        strategy_gene_schema_sha256: resolved_full_discovery_plan_authority
            .strategy_gene_schema_sha256,
        fitness_ordering_semantics_sha256: resolved_full_discovery_plan_authority
            .fitness_ordering_semantics_sha256,
        crossover_semantics_sha256: resolved_full_discovery_plan_authority
            .crossover_semantics_sha256,
        mutation_semantics_sha256: resolved_full_discovery_plan_authority.mutation_semantics_sha256,
        strategy_metrics_semantics_sha256: resolved_full_discovery_plan_authority
            .strategy_metrics_semantics_sha256,
        permit_identity_sha256,
    })
}

fn reject_card_present_cpu_or_fallback_v2(
    authority: &SealedResidentFullDiscoveryInputAuthorityV2,
) -> Result<(), FullDiscoveryGpuStageAuthorityErrorV2> {
    match authority.route_refusal {
        None => Ok(()),
        Some(StrictGpuRouteRefusalV2::CpuRoute) => {
            Err(FullDiscoveryGpuStageAuthorityErrorV2::CardPresentCpuExecutionForbidden)
        }
        Some(StrictGpuRouteRefusalV2::CpuAllowed) => {
            Err(FullDiscoveryGpuStageAuthorityErrorV2::CardPresentAllowCpuForbidden)
        }
    }
}

fn require_all_full_discovery_stages_strict_gpu_v2(
    manifest: &GpuCapabilityManifest,
) -> Result<(), FullDiscoveryGpuStageAuthorityErrorV2> {
    if PipelineStage::FULL_DISCOVERY.len() != FULL_DISCOVERY_STAGE_COUNT_V2
        || manifest.entries().len() != FULL_DISCOVERY_STAGE_COUNT_V2
    {
        return Err(FullDiscoveryGpuStageAuthorityErrorV2::FullDiscoveryStageCountMismatch);
    }
    for expected in PipelineStage::FULL_DISCOVERY {
        let matching = manifest
            .entries()
            .iter()
            .filter(|entry| entry.stage == expected)
            .collect::<Vec<_>>();
        if matching.is_empty() {
            return Err(FullDiscoveryGpuStageAuthorityErrorV2::MissingStageCapability(expected));
        }
        if matching.len() != 1 {
            return Err(FullDiscoveryGpuStageAuthorityErrorV2::FullDiscoveryStageCountMismatch);
        }
        match matching[0].capability {
            StageGpuCapability::StrictGpu => {}
            StageGpuCapability::HybridOnly
            | StageGpuCapability::CpuOnly
            | StageGpuCapability::Unsupported => {
                return Err(
                    FullDiscoveryGpuStageAuthorityErrorV2::NonStrictFullDiscoveryStage {
                        stage: expected,
                        capability: matching[0].capability,
                    },
                );
            }
        }
    }
    Ok(())
}

fn bind_exact_ordinal_device_build_and_input_v2(
    resident: &SealedResidentFullDiscoveryInputAuthorityV2,
    plan: &ResolvedFullDiscoveryPlanAuthorityV2,
) -> Result<ExactOrdinalDeviceBuildInputBindingV2, FullDiscoveryGpuStageAuthorityErrorV2> {
    let selected_cuda_ordinal = resident
        .selected_cuda_ordinal
        .ok_or(FullDiscoveryGpuStageAuthorityErrorV2::ExactCudaOrdinalRequired)?;
    if selected_cuda_ordinal != plan.selected_cuda_ordinal {
        return Err(
            FullDiscoveryGpuStageAuthorityErrorV2::ResidentInputPlanMismatch(
                "selected CUDA ordinal",
            ),
        );
    }
    for (field, value) in [
        (
            "CUDA device identity",
            resident.cuda_device_identity_sha256.as_str(),
        ),
        (
            "CUDA build manifest",
            resident.cuda_build_manifest_sha256.as_str(),
        ),
        (
            "canonical Search input receipt",
            resident.canonical_search_input_receipt_sha256.as_str(),
        ),
        (
            "resident input content",
            resident.resident_input_content_sha256.as_str(),
        ),
        (
            "full Discovery workspace plan",
            resident
                .full_discovery_workspace_plan_identity_sha256
                .as_str(),
        ),
    ] {
        require_sha256_hex_v2(field, value)?;
    }
    Ok(ExactOrdinalDeviceBuildInputBindingV2 {
        selected_cuda_ordinal,
        cuda_device_identity_sha256: resident.cuda_device_identity_sha256.clone(),
        cuda_build_manifest_sha256: resident.cuda_build_manifest_sha256.clone(),
        canonical_search_input_receipt_sha256: resident
            .canonical_search_input_receipt_sha256
            .clone(),
        resident_input_content_sha256: resident.resident_input_content_sha256.clone(),
        full_discovery_workspace_plan_identity_sha256: resident
            .full_discovery_workspace_plan_identity_sha256
            .clone(),
    })
}

fn validate_resolved_plan_against_resident_input_v2(
    binding: &ExactOrdinalDeviceBuildInputBindingV2,
    plan: &ResolvedFullDiscoveryPlanAuthorityV2,
) -> Result<(), FullDiscoveryGpuStageAuthorityErrorV2> {
    for (field, resident, resolved) in [
        (
            "CUDA device identity",
            binding.cuda_device_identity_sha256.as_str(),
            plan.cuda_device_identity_sha256.as_str(),
        ),
        (
            "CUDA build manifest",
            binding.cuda_build_manifest_sha256.as_str(),
            plan.cuda_build_manifest_sha256.as_str(),
        ),
        (
            "canonical Search input receipt",
            binding.canonical_search_input_receipt_sha256.as_str(),
            plan.canonical_search_input_receipt_sha256.as_str(),
        ),
        (
            "resident input content",
            binding.resident_input_content_sha256.as_str(),
            plan.resident_input_content_sha256.as_str(),
        ),
        (
            "full Discovery workspace plan",
            binding
                .full_discovery_workspace_plan_identity_sha256
                .as_str(),
            plan.full_discovery_workspace_plan_identity_sha256.as_str(),
        ),
    ] {
        require_sha256_hex_v2(field, resolved)?;
        if resident != resolved {
            return Err(FullDiscoveryGpuStageAuthorityErrorV2::ResidentInputPlanMismatch(field));
        }
    }
    Ok(())
}

fn compute_full_discovery_stage_manifest_identity_v2(
    manifest: &GpuCapabilityManifest,
    ordered_stage_identity_sha256: &[String; FULL_DISCOVERY_STAGE_COUNT_V2],
) -> Result<String, FullDiscoveryGpuStageAuthorityErrorV2> {
    let mut hasher = Sha256::new();
    hasher.update(FULL_DISCOVERY_STAGE_MANIFEST_DOMAIN_V2);
    hasher.update((FULL_DISCOVERY_STAGE_COUNT_V2 as u64).to_le_bytes());
    for (index, stage) in PipelineStage::FULL_DISCOVERY.into_iter().enumerate() {
        let capability = manifest
            .capability(stage)
            .ok_or(FullDiscoveryGpuStageAuthorityErrorV2::MissingStageCapability(stage))?;
        if capability.capability != StageGpuCapability::StrictGpu {
            return Err(
                FullDiscoveryGpuStageAuthorityErrorV2::NonStrictFullDiscoveryStage {
                    stage,
                    capability: capability.capability,
                },
            );
        }
        require_sha256_hex_v2(
            "stage semantics identity",
            &ordered_stage_identity_sha256[index],
        )?;
        hash_length_prefixed_v2(&mut hasher, pipeline_stage_wire_name_v2(stage).as_bytes());
        hash_length_prefixed_v2(&mut hasher, capability.detail.as_bytes());
        hash_length_prefixed_v2(&mut hasher, ordered_stage_identity_sha256[index].as_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_plan_semantics_identities_v2(
    plan: &ResolvedFullDiscoveryPlanAuthorityV2,
) -> Result<(), FullDiscoveryGpuStageAuthorityErrorV2> {
    for (field, value) in [
        (
            "resolved Search config",
            plan.resolved_search_config_sha256.as_str(),
        ),
        (
            "strategy gene schema",
            plan.strategy_gene_schema_sha256.as_str(),
        ),
        (
            "fitness ordering semantics",
            plan.fitness_ordering_semantics_sha256.as_str(),
        ),
        (
            "crossover semantics",
            plan.crossover_semantics_sha256.as_str(),
        ),
        (
            "mutation semantics",
            plan.mutation_semantics_sha256.as_str(),
        ),
        (
            "strategy metrics semantics",
            plan.strategy_metrics_semantics_sha256.as_str(),
        ),
    ] {
        require_sha256_hex_v2(field, value)?;
    }
    Ok(())
}

fn compute_strict_gpu_only_full_discovery_permit_identity_v2(
    binding: &ExactOrdinalDeviceBuildInputBindingV2,
    plan: &ResolvedFullDiscoveryPlanAuthorityV2,
    full_discovery_stage_manifest_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(STRICT_GPU_ONLY_FULL_DISCOVERY_PERMIT_DOMAIN_V2);
    hasher.update(binding.selected_cuda_ordinal.to_le_bytes());
    for value in [
        binding.cuda_device_identity_sha256.as_str(),
        binding.cuda_build_manifest_sha256.as_str(),
        binding.canonical_search_input_receipt_sha256.as_str(),
        binding.resident_input_content_sha256.as_str(),
        binding
            .full_discovery_workspace_plan_identity_sha256
            .as_str(),
        plan.resolved_search_config_sha256.as_str(),
        full_discovery_stage_manifest_sha256,
        plan.strategy_gene_schema_sha256.as_str(),
        plan.fitness_ordering_semantics_sha256.as_str(),
        plan.crossover_semantics_sha256.as_str(),
        plan.mutation_semantics_sha256.as_str(),
        plan.strategy_metrics_semantics_sha256.as_str(),
    ] {
        hash_length_prefixed_v2(&mut hasher, value.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn require_sha256_hex_v2(
    field: &'static str,
    value: &str,
) -> Result<(), FullDiscoveryGpuStageAuthorityErrorV2> {
    if value.len() != SHA256_HEX_LENGTH_V2
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(FullDiscoveryGpuStageAuthorityErrorV2::InvalidAuthorityIdentity(field));
    }
    Ok(())
}

fn hash_length_prefixed_v2(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

const fn pipeline_stage_wire_name_v2(stage: PipelineStage) -> &'static str {
    match stage {
        PipelineStage::FeaturePreparation => "feature-preparation",
        PipelineStage::GaGenerationSelection => "ga-generation-selection",
        PipelineStage::PopulationEvaluation => "population-evaluation",
        PipelineStage::SignalAndMinTradeFilter => "signal-and-min-trade-filter",
        PipelineStage::QualityScreen => "quality-screen",
        PipelineStage::MonteCarlo => "monte-carlo",
        PipelineStage::PropFirmWindow => "prop-firm-window",
        PipelineStage::CandidateCorrelation => "candidate-correlation",
        PipelineStage::WalkForward => "walk-forward",
        PipelineStage::Cpcv => "cpcv",
        PipelineStage::Pbo => "pbo",
        PipelineStage::RobustnessPermutationPlateau => "robustness-permutation-plateau",
        PipelineStage::RiskDiagnostics => "risk-diagnostics",
        PipelineStage::CanonicalReplay => "canonical-replay",
        PipelineStage::ForwardTailReplay => "forward-tail-replay",
        PipelineStage::SurvivorRanking => "survivor-ranking",
    }
}
