//! Lifetime-aware, one-shot full Discovery CUDA workspace admission.

#[cfg(feature = "cuda")]
use crate::resident_feature_store_v3::{
    GpuOnlyRunDeviceAdmissionRequestV3, GpuOnlyRunDeviceAdmissionV3,
    seal_gpu_only_run_device_admission_v3,
};
use crate::run_device_admission_v1::{
    DiscoveryRunDeviceAdmissionErrorV1, SealedDiscoveryRunDeviceAdmissionV1,
};
#[cfg(feature = "cuda")]
use crate::run_device_admission_v1::{
    SealedCudaNativeBuildIdentityV1, SealedNativeCudaRunDeviceAdmissionV1,
};
use sha2::{Digest, Sha256};
use thiserror::Error;

const FULL_DISCOVERY_WORKSPACE_PLAN_SCHEMA_V1: &str =
    "neoethos.full-discovery-gpu-workspace-plan.v1";
const DISCOVERY_SEMANTICS_VERSION: &str = "neoethos.discovery-semantics.v1";
const MAX_FINAL_COMPACT_READBACK_BYTES_V1: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullDiscoveryWorkspaceStageV1 {
    ResidentFeatureStore,
    PopulationParentAndViews,
    ResidentGeneticEvolution,
    WalkForwardValidation,
    CpcvAndPbo,
    OuterHoldoutAndOos,
    PortfolioConstraints,
    RobustnessTails,
    FinalCompactReadback,
    WorkspaceSemantics,
}

impl FullDiscoveryWorkspaceStageV1 {
    const fn wire_name(self) -> &'static str {
        match self {
            Self::ResidentFeatureStore => "resident-feature-store",
            Self::PopulationParentAndViews => "population-parent-and-views",
            Self::ResidentGeneticEvolution => "resident-genetic-evolution",
            Self::WalkForwardValidation => "walk-forward-validation",
            Self::CpcvAndPbo => "cpcv-and-pbo",
            Self::OuterHoldoutAndOos => "outer-holdout-and-oos",
            Self::PortfolioConstraints => "portfolio-constraints",
            Self::RobustnessTails => "robustness-tails",
            Self::FinalCompactReadback => "final-compact-readback",
            Self::WorkspaceSemantics => "workspace-semantics",
        }
    }
}

#[derive(Debug)]
pub struct OpaqueFullDiscoveryStagePreflightV1 {
    stage: FullDiscoveryWorkspaceStageV1,
    device_bytes: u64,
    stage_identity_sha256: [u8; 32],
}

impl OpaqueFullDiscoveryStagePreflightV1 {
    pub const fn stage(&self) -> FullDiscoveryWorkspaceStageV1 {
        self.stage
    }

    pub const fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    pub const fn stage_identity_sha256(&self) -> [u8; 32] {
        self.stage_identity_sha256
    }
}

#[derive(Debug)]
pub struct OpaqueFullDiscoveryPhasePreflightV1 {
    stage: FullDiscoveryWorkspaceStageV1,
    scratch_bytes: u64,
    lifetime_interval_start: u32,
    lifetime_interval_end: u32,
    producer_completion_event_identity_sha256: [u8; 32],
    consumer_wait_event_identity_sha256: [u8; 32],
    phase_identity_sha256: [u8; 32],
}

impl OpaqueFullDiscoveryPhasePreflightV1 {
    pub const fn stage(&self) -> FullDiscoveryWorkspaceStageV1 {
        self.stage
    }

    pub const fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }
}

#[derive(Debug)]
pub struct OpaqueFullDiscoveryFinalReadbackPreflightV1 {
    bounded_bytes: u64,
    stage_identity_sha256: [u8; 32],
}

#[derive(Debug)]
pub struct OpaqueFullDiscoveryWorkspaceSemanticsPreflightV1 {
    discovery_semantics_version: String,
    discovery_semantics_sha256: [u8; 32],
    population_gap_flags_bytes: u64,
    population_gene_rank_rng_bytes: u64,
    population_fixed_native_buffers_bytes: u64,
    population_view_index_adaptive_bytes: u64,
    allocator_context_reserve_bytes: u64,
    vector_ta_build_sha256: [u8; 32],
    exact_math_authority: String,
    stage_identity_sha256: [u8; 32],
}

#[derive(Debug)]
pub struct FullDiscoveryWorkspacePreflightBundleV1 {
    resident_feature_store: OpaqueFullDiscoveryStagePreflightV1,
    population_parent_and_views: OpaqueFullDiscoveryStagePreflightV1,
    resident_genetic_evolution: OpaqueFullDiscoveryStagePreflightV1,
    walk_forward_validation: OpaqueFullDiscoveryPhasePreflightV1,
    cpcv_and_pbo: OpaqueFullDiscoveryPhasePreflightV1,
    outer_holdout_and_oos: OpaqueFullDiscoveryPhasePreflightV1,
    portfolio_constraints: OpaqueFullDiscoveryStagePreflightV1,
    robustness_tails: OpaqueFullDiscoveryPhasePreflightV1,
    final_compact_readback: OpaqueFullDiscoveryFinalReadbackPreflightV1,
    workspace_semantics: OpaqueFullDiscoveryWorkspaceSemanticsPreflightV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullDiscoveryWorkspacePlanErrorCodeV1 {
    MissingStageRequirement,
    StageIdentityMismatch,
    WorkspaceExtentOverflow,
    InvalidPhaseInterval,
    PhaseOverlap,
    PhaseEventReuseMismatch,
    FinalReadbackNotBounded,
    InvalidSemantics,
    InsufficientExactOrdinalMemory,
    CpuRouteCannotBindGpuWorkspace,
    RunDeviceCarrierSealingFailure,
    InvalidCompletionEvidence,
    RunDeviceAdmissionFailure,
}

#[derive(Debug, Error)]
#[error("full Discovery GPU workspace admission failed ({code:?}): {detail}")]
pub struct FullDiscoveryWorkspacePlanErrorV1 {
    code: FullDiscoveryWorkspacePlanErrorCodeV1,
    detail: String,
}

impl FullDiscoveryWorkspacePlanErrorV1 {
    fn new(code: FullDiscoveryWorkspacePlanErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> FullDiscoveryWorkspacePlanErrorCodeV1 {
        self.code
    }
}

impl From<DiscoveryRunDeviceAdmissionErrorV1> for FullDiscoveryWorkspacePlanErrorV1 {
    fn from(error: DiscoveryRunDeviceAdmissionErrorV1) -> Self {
        Self::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::RunDeviceAdmissionFailure,
            error.to_string(),
        )
    }
}

#[derive(Debug)]
struct FullDiscoveryWorkspacePhaseV1 {
    stage: FullDiscoveryWorkspaceStageV1,
    scratch_bytes: u64,
    lifetime_interval_start: u32,
    lifetime_interval_end: u32,
    producer_completion_event_identity_sha256: [u8; 32],
    consumer_wait_event_identity_sha256: [u8; 32],
    phase_identity_sha256: [u8; 32],
}

#[derive(Debug)]
struct PhaseArenaReuseProofV1 {
    reusable_phase_arena_bytes: u64,
    ordered_phase_identity_sha256: Vec<[u8; 32]>,
    event_chain_identity_sha256: [u8; 32],
}

#[derive(Debug)]
pub struct SealedFullDiscoveryGpuWorkspacePlanV1 {
    always_resident_bytes: u64,
    reusable_phase_arena_bytes: u64,
    bounded_final_readback_bytes: u64,
    required_workspace_bytes: u64,
    allocator_context_reserve_bytes: u64,
    phase_lifetime_plan: Vec<FullDiscoveryWorkspacePhaseV1>,
    phase_arena_reuse_proof: PhaseArenaReuseProofV1,
    discovery_semantics_sha256: [u8; 32],
    vector_ta_build_sha256: [u8; 32],
    exact_math_authority: String,
    workspace_plan_identity_sha256: [u8; 32],
}

impl SealedFullDiscoveryGpuWorkspacePlanV1 {
    pub const fn always_resident_bytes(&self) -> u64 {
        self.always_resident_bytes
    }

    pub const fn reusable_phase_arena_bytes(&self) -> u64 {
        self.reusable_phase_arena_bytes
    }

    pub const fn bounded_final_readback_bytes(&self) -> u64 {
        self.bounded_final_readback_bytes
    }

    pub const fn required_workspace_bytes(&self) -> u64 {
        self.required_workspace_bytes
    }

    pub const fn workspace_plan_identity_sha256(&self) -> [u8; 32] {
        self.workspace_plan_identity_sha256
    }

    pub const fn allocator_context_reserve_bytes(&self) -> u64 {
        self.allocator_context_reserve_bytes
    }

    pub fn phase_count(&self) -> usize {
        self.phase_lifetime_plan.len()
    }

    pub const fn phase_reuse_event_chain_identity_sha256(&self) -> [u8; 32] {
        self.phase_arena_reuse_proof.event_chain_identity_sha256
    }

    pub const fn discovery_semantics_sha256(&self) -> [u8; 32] {
        self.discovery_semantics_sha256
    }

    pub const fn vector_ta_build_sha256(&self) -> [u8; 32] {
        self.vector_ta_build_sha256
    }

    pub fn exact_math_authority(&self) -> &str {
        &self.exact_math_authority
    }
}

pub fn seal_full_discovery_gpu_workspace_plan_v1(
    preflight: FullDiscoveryWorkspacePreflightBundleV1,
) -> Result<SealedFullDiscoveryGpuWorkspacePlanV1, FullDiscoveryWorkspacePlanErrorV1> {
    require_stage_v1(
        &preflight.resident_feature_store,
        FullDiscoveryWorkspaceStageV1::ResidentFeatureStore,
    )?;
    require_stage_v1(
        &preflight.population_parent_and_views,
        FullDiscoveryWorkspaceStageV1::PopulationParentAndViews,
    )?;
    require_stage_v1(
        &preflight.resident_genetic_evolution,
        FullDiscoveryWorkspaceStageV1::ResidentGeneticEvolution,
    )?;
    require_stage_v1(
        &preflight.portfolio_constraints,
        FullDiscoveryWorkspaceStageV1::PortfolioConstraints,
    )?;
    require_workspace_semantics_v1(&preflight.workspace_semantics)?;

    let always_resident_bytes = checked_sum_always_resident_bytes_v1(&preflight)?;
    let (phase_lifetime_plan, phase_arena_reuse_proof) = seal_mutually_exclusive_phase_arena_v1(
        preflight.walk_forward_validation,
        preflight.cpcv_and_pbo,
        preflight.outer_holdout_and_oos,
        preflight.robustness_tails,
    )?;
    let reusable_phase_arena_bytes = phase_arena_reuse_proof.reusable_phase_arena_bytes;
    let bounded_final_readback_bytes = preflight.final_compact_readback.bounded_bytes;
    if bounded_final_readback_bytes == 0
        || bounded_final_readback_bytes > MAX_FINAL_COMPACT_READBACK_BYTES_V1
        || is_zero_sha256_v1(preflight.final_compact_readback.stage_identity_sha256)
    {
        return Err(FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::FinalReadbackNotBounded,
            "final compact readback must have a nonzero identity and reviewed bound",
        ));
    }
    let resident_plus_phase = always_resident_bytes.checked_add(reusable_phase_arena_bytes);
    let resident_and_phase_bytes = resident_plus_phase.ok_or_else(|| {
        FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::WorkspaceExtentOverflow,
            "resident plus reusable phase arena overflowed",
        )
    })?;
    let required_workspace_bytes = resident_and_phase_bytes
        .checked_add(bounded_final_readback_bytes)
        .ok_or_else(|| {
            FullDiscoveryWorkspacePlanErrorV1::new(
                FullDiscoveryWorkspacePlanErrorCodeV1::WorkspaceExtentOverflow,
                "workspace plus bounded final readback overflowed",
            )
        })?;
    let component_identity_sha256 = [
        preflight.resident_feature_store.stage_identity_sha256,
        preflight.population_parent_and_views.stage_identity_sha256,
        preflight.resident_genetic_evolution.stage_identity_sha256,
        preflight.portfolio_constraints.stage_identity_sha256,
        preflight.final_compact_readback.stage_identity_sha256,
        preflight.workspace_semantics.stage_identity_sha256,
        preflight.workspace_semantics.discovery_semantics_sha256,
        preflight.workspace_semantics.vector_ta_build_sha256,
    ];
    let workspace_plan_identity_sha256 =
        hash_workspace_plan_v1(&FullDiscoveryWorkspacePlanHashInputV1 {
            always_resident_bytes,
            reusable_phase_arena_bytes,
            bounded_final_readback_bytes,
            required_workspace_bytes,
            phases: &phase_lifetime_plan,
            reuse_proof: &phase_arena_reuse_proof,
            component_identity_sha256: &component_identity_sha256,
            exact_math_authority: &preflight.workspace_semantics.exact_math_authority,
        });
    Ok(SealedFullDiscoveryGpuWorkspacePlanV1 {
        always_resident_bytes,
        reusable_phase_arena_bytes,
        bounded_final_readback_bytes,
        required_workspace_bytes,
        allocator_context_reserve_bytes: preflight
            .workspace_semantics
            .allocator_context_reserve_bytes,
        phase_lifetime_plan,
        phase_arena_reuse_proof,
        discovery_semantics_sha256: preflight.workspace_semantics.discovery_semantics_sha256,
        vector_ta_build_sha256: preflight.workspace_semantics.vector_ta_build_sha256,
        exact_math_authority: preflight.workspace_semantics.exact_math_authority,
        workspace_plan_identity_sha256,
    })
}

fn checked_sum_always_resident_bytes_v1(
    preflight: &FullDiscoveryWorkspacePreflightBundleV1,
) -> Result<u64, FullDiscoveryWorkspacePlanErrorV1> {
    let semantics = &preflight.workspace_semantics;
    [
        preflight.resident_feature_store.device_bytes,
        preflight.population_parent_and_views.device_bytes,
        preflight.resident_genetic_evolution.device_bytes,
        preflight.portfolio_constraints.device_bytes,
        semantics.population_gap_flags_bytes,
        semantics.population_gene_rank_rng_bytes,
        semantics.population_fixed_native_buffers_bytes,
        semantics.population_view_index_adaptive_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        total.checked_add(bytes).ok_or_else(|| {
            FullDiscoveryWorkspacePlanErrorV1::new(
                FullDiscoveryWorkspacePlanErrorCodeV1::WorkspaceExtentOverflow,
                "always-resident device extent overflowed",
            )
        })
    })
}

fn seal_mutually_exclusive_phase_arena_v1(
    walk_forward_validation: OpaqueFullDiscoveryPhasePreflightV1,
    cpcv_and_pbo: OpaqueFullDiscoveryPhasePreflightV1,
    outer_holdout_and_oos: OpaqueFullDiscoveryPhasePreflightV1,
    robustness_tails: OpaqueFullDiscoveryPhasePreflightV1,
) -> Result<
    (Vec<FullDiscoveryWorkspacePhaseV1>, PhaseArenaReuseProofV1),
    FullDiscoveryWorkspacePlanErrorV1,
> {
    let phases = vec![
        seal_phase_v1(
            walk_forward_validation,
            FullDiscoveryWorkspaceStageV1::WalkForwardValidation,
        )?,
        seal_phase_v1(cpcv_and_pbo, FullDiscoveryWorkspaceStageV1::CpcvAndPbo)?,
        seal_phase_v1(
            outer_holdout_and_oos,
            FullDiscoveryWorkspaceStageV1::OuterHoldoutAndOos,
        )?,
        seal_phase_v1(
            robustness_tails,
            FullDiscoveryWorkspaceStageV1::RobustnessTails,
        )?,
    ];
    for pair in phases.windows(2) {
        require_non_overlapping_interval_v1(&pair[0], &pair[1])?;
    }
    let reusable_phase_arena_bytes = checked_max_mutually_exclusive_phase_bytes_v1(&phases)?;
    let event_chain_identity_sha256 = hash_phase_event_chain_v1(&phases);
    let proof = PhaseArenaReuseProofV1 {
        reusable_phase_arena_bytes,
        ordered_phase_identity_sha256: phases
            .iter()
            .map(|phase| phase.phase_identity_sha256)
            .collect(),
        event_chain_identity_sha256,
    };
    Ok((phases, proof))
}

fn checked_max_mutually_exclusive_phase_bytes_v1(
    phases: &[FullDiscoveryWorkspacePhaseV1],
) -> Result<u64, FullDiscoveryWorkspacePlanErrorV1> {
    phases
        .iter()
        .map(|phase| phase.scratch_bytes)
        .max()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            FullDiscoveryWorkspacePlanErrorV1::new(
                FullDiscoveryWorkspacePlanErrorCodeV1::MissingStageRequirement,
                "full Discovery requires nonzero reusable phase scratch",
            )
        })
}

fn require_non_overlapping_interval_v1(
    producer: &FullDiscoveryWorkspacePhaseV1,
    consumer: &FullDiscoveryWorkspacePhaseV1,
) -> Result<(), FullDiscoveryWorkspacePlanErrorV1> {
    if producer.lifetime_interval_end > consumer.lifetime_interval_start {
        return Err(FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::PhaseOverlap,
            "mutually exclusive phase lifetimes overlap",
        ));
    }
    if producer.producer_completion_event_identity_sha256
        != consumer.consumer_wait_event_identity_sha256
    {
        return Err(FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::PhaseEventReuseMismatch,
            "phase arena reuse lacks an exact producer-completion/consumer-wait event chain",
        ));
    }
    Ok(())
}

fn seal_phase_v1(
    preflight: OpaqueFullDiscoveryPhasePreflightV1,
    expected: FullDiscoveryWorkspaceStageV1,
) -> Result<FullDiscoveryWorkspacePhaseV1, FullDiscoveryWorkspacePlanErrorV1> {
    if preflight.stage != expected
        || preflight.scratch_bytes == 0
        || is_zero_sha256_v1(preflight.phase_identity_sha256)
        || is_zero_sha256_v1(preflight.producer_completion_event_identity_sha256)
        || is_zero_sha256_v1(preflight.consumer_wait_event_identity_sha256)
    {
        return Err(FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::MissingStageRequirement,
            format!(
                "missing or invalid {} phase preflight",
                expected.wire_name()
            ),
        ));
    }
    if preflight.lifetime_interval_start >= preflight.lifetime_interval_end {
        return Err(FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::InvalidPhaseInterval,
            format!("invalid {} phase lifetime", expected.wire_name()),
        ));
    }
    Ok(FullDiscoveryWorkspacePhaseV1 {
        stage: expected,
        scratch_bytes: preflight.scratch_bytes,
        lifetime_interval_start: preflight.lifetime_interval_start,
        lifetime_interval_end: preflight.lifetime_interval_end,
        producer_completion_event_identity_sha256: preflight
            .producer_completion_event_identity_sha256,
        consumer_wait_event_identity_sha256: preflight.consumer_wait_event_identity_sha256,
        phase_identity_sha256: preflight.phase_identity_sha256,
    })
}

fn require_stage_v1(
    preflight: &OpaqueFullDiscoveryStagePreflightV1,
    expected: FullDiscoveryWorkspaceStageV1,
) -> Result<(), FullDiscoveryWorkspacePlanErrorV1> {
    if preflight.stage != expected
        || preflight.device_bytes == 0
        || is_zero_sha256_v1(preflight.stage_identity_sha256)
    {
        return Err(FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::MissingStageRequirement,
            format!("missing or invalid {} preflight", expected.wire_name()),
        ));
    }
    Ok(())
}

fn require_workspace_semantics_v1(
    semantics: &OpaqueFullDiscoveryWorkspaceSemanticsPreflightV1,
) -> Result<(), FullDiscoveryWorkspacePlanErrorV1> {
    if semantics.discovery_semantics_version != DISCOVERY_SEMANTICS_VERSION
        || is_zero_sha256_v1(semantics.discovery_semantics_sha256)
        || is_zero_sha256_v1(semantics.vector_ta_build_sha256)
        || semantics.exact_math_authority.trim().is_empty()
        || is_zero_sha256_v1(semantics.stage_identity_sha256)
    {
        return Err(FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::InvalidSemantics,
            "workspace semantics/build/math identity is incomplete",
        ));
    }
    Ok(())
}

fn hash_phase_event_chain_v1(phases: &[FullDiscoveryWorkspacePhaseV1]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.full-discovery-phase-event-chain.v1");
    for phase in phases {
        hasher.update(phase.stage.wire_name().as_bytes());
        hasher.update(phase.lifetime_interval_start.to_le_bytes());
        hasher.update(phase.lifetime_interval_end.to_le_bytes());
        hasher.update(phase.producer_completion_event_identity_sha256);
        hasher.update(phase.consumer_wait_event_identity_sha256);
        hasher.update(phase.phase_identity_sha256);
    }
    hasher.finalize().into()
}

struct FullDiscoveryWorkspacePlanHashInputV1<'a> {
    always_resident_bytes: u64,
    reusable_phase_arena_bytes: u64,
    bounded_final_readback_bytes: u64,
    required_workspace_bytes: u64,
    phases: &'a [FullDiscoveryWorkspacePhaseV1],
    reuse_proof: &'a PhaseArenaReuseProofV1,
    component_identity_sha256: &'a [[u8; 32]],
    exact_math_authority: &'a str,
}

fn hash_workspace_plan_v1(input: &FullDiscoveryWorkspacePlanHashInputV1<'_>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(FULL_DISCOVERY_WORKSPACE_PLAN_SCHEMA_V1.as_bytes());
    hasher.update(input.always_resident_bytes.to_le_bytes());
    hasher.update(input.reusable_phase_arena_bytes.to_le_bytes());
    hasher.update(input.bounded_final_readback_bytes.to_le_bytes());
    hasher.update(input.required_workspace_bytes.to_le_bytes());
    hasher.update(input.reuse_proof.event_chain_identity_sha256);
    for identity in &input.reuse_proof.ordered_phase_identity_sha256 {
        hasher.update(identity);
    }
    for phase in input.phases {
        hasher.update(phase.stage.wire_name().as_bytes());
        hasher.update(phase.scratch_bytes.to_le_bytes());
        hasher.update(phase.phase_identity_sha256);
    }
    for identity in input.component_identity_sha256 {
        hasher.update(identity);
    }
    hasher.update(input.exact_math_authority.as_bytes());
    hasher.finalize().into()
}

const fn is_zero_sha256_v1(value: [u8; 32]) -> bool {
    let mut index = 0;
    while index < value.len() {
        if value[index] != 0 {
            return false;
        }
        index += 1;
    }
    true
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct AdmittedNativeCudaFullDiscoveryRunV1 {
    physical_inventory_identity_sha256: [u8; 32],
    admission_identity_sha256: [u8; 32],
    workspace_plan_identity_sha256: [u8; 32],
    device_uuid: [u8; 16],
    selected_device_ordinal: u32,
    pci_identity: crate::physical_gpu_inventory_v1::PhysicalGpuInventoryRecordV1,
    primary_context: std::sync::Arc<cust::context::Context>,
    run_stream: std::sync::Arc<cust::stream::Stream>,
    cuda_build_identity: SealedCudaNativeBuildIdentityV1,
    sass_target: String,
    driver_version: String,
    context_api_version: String,
    compute_capability_major: u16,
    compute_capability_minor: u16,
    free_memory_bytes_snapshot: u64,
    plan: SealedFullDiscoveryGpuWorkspacePlanV1,
}

#[derive(Debug)]
pub enum AdmittedFullDiscoveryGpuRunV1 {
    #[cfg(feature = "cuda")]
    NativeCuda(AdmittedNativeCudaFullDiscoveryRunV1),
}

pub fn bind_full_discovery_workspace_plan_v1(
    admission: SealedDiscoveryRunDeviceAdmissionV1,
    #[cfg(feature = "cuda")] plan: SealedFullDiscoveryGpuWorkspacePlanV1,
    #[cfg(not(feature = "cuda"))] _plan: SealedFullDiscoveryGpuWorkspacePlanV1,
) -> Result<AdmittedFullDiscoveryGpuRunV1, FullDiscoveryWorkspacePlanErrorV1> {
    let probe_counters = admission.probe_counters();
    probe_counters.require_exact_single_run_device_acquisition_v1()?;
    match admission {
        SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(_) => {
            Err(FullDiscoveryWorkspacePlanErrorV1::new(
                FullDiscoveryWorkspacePlanErrorCodeV1::CpuRouteCannotBindGpuWorkspace,
                "a complete no-physical-GPU CPU route cannot bind a CUDA workspace plan",
            ))
        }
        #[cfg(feature = "cuda")]
        SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native) => {
            let native = *native;
            let admission_identity_sha256 = native.admission_identity_sha256;
            let workspace_plan_identity_sha256 = plan.workspace_plan_identity_sha256;
            let free_memory_bytes_snapshot = native.free_memory_bytes_snapshot;
            let required_workspace_bytes = plan.required_workspace_bytes;
            let allocatable_after_reserve = free_memory_bytes_snapshot
                .checked_sub(plan.allocator_context_reserve_bytes)
                .ok_or_else(|| {
                    FullDiscoveryWorkspacePlanErrorV1::new(
                        FullDiscoveryWorkspacePlanErrorCodeV1::InsufficientExactOrdinalMemory,
                        "allocator/context reserve exceeds the admitted free-memory snapshot",
                    )
                })?;
            let _remaining_after_full_workspace = allocatable_after_reserve
                .checked_sub(required_workspace_bytes)
                .ok_or_else(|| {
                    FullDiscoveryWorkspacePlanErrorV1::new(
                        FullDiscoveryWorkspacePlanErrorCodeV1::InsufficientExactOrdinalMemory,
                        "selected CUDA ordinal cannot fit the sealed full Discovery workspace",
                    )
                })?;
            bind_native_full_discovery_workspace_v1(
                native,
                plan,
                admission_identity_sha256,
                workspace_plan_identity_sha256,
            )
        }
    }
}

#[cfg(feature = "cuda")]
fn bind_native_full_discovery_workspace_v1(
    admission: SealedNativeCudaRunDeviceAdmissionV1,
    plan: SealedFullDiscoveryGpuWorkspacePlanV1,
    admission_identity_sha256: [u8; 32],
    workspace_plan_identity_sha256: [u8; 32],
) -> Result<AdmittedFullDiscoveryGpuRunV1, FullDiscoveryWorkspacePlanErrorV1> {
    let free_memory_bytes_snapshot = admission.free_memory_bytes_snapshot;
    Ok(AdmittedFullDiscoveryGpuRunV1::NativeCuda(
        AdmittedNativeCudaFullDiscoveryRunV1 {
            physical_inventory_identity_sha256: admission.physical_inventory_identity_sha256,
            admission_identity_sha256,
            workspace_plan_identity_sha256,
            device_uuid: admission.device_uuid,
            selected_device_ordinal: admission.ordinal,
            pci_identity: admission.pci_identity,
            primary_context: admission.primary_context,
            run_stream: admission.run_stream,
            cuda_build_identity: admission.cuda_build_identity,
            sass_target: admission.sass_target,
            driver_version: admission.driver_version,
            context_api_version: admission.context_api_version,
            compute_capability_major: admission.compute_capability_major,
            compute_capability_minor: admission.compute_capability_minor,
            free_memory_bytes_snapshot,
            plan,
        },
    ))
}

#[cfg(feature = "cuda")]
impl AdmittedNativeCudaFullDiscoveryRunV1 {
    pub fn into_gpu_only_run_device_admission_v3(
        self,
    ) -> Result<GpuOnlyRunDeviceAdmissionV3, FullDiscoveryWorkspacePlanErrorV1> {
        let Self {
            admission_identity_sha256,
            workspace_plan_identity_sha256,
            device_uuid,
            selected_device_ordinal,
            primary_context,
            run_stream,
            cuda_build_identity,
            sass_target,
            driver_version,
            context_api_version,
            compute_capability_major,
            compute_capability_minor,
            free_memory_bytes_snapshot,
            plan,
            ..
        } = self;
        let SealedCudaNativeBuildIdentityV1 {
            artifact_sha256: gpu_cuda_build_sha256,
            nvcc_version,
            ..
        } = cuda_build_identity;
        let SealedFullDiscoveryGpuWorkspacePlanV1 {
            allocator_context_reserve_bytes,
            vector_ta_build_sha256,
            exact_math_authority,
            ..
        } = plan;
        seal_gpu_only_run_device_admission_v3(GpuOnlyRunDeviceAdmissionRequestV3 {
            source_admission_identity_sha256: admission_identity_sha256,
            workspace_plan_identity_sha256,
            selected_device_ordinal,
            device_uuid,
            compute_capability_major,
            compute_capability_minor,
            primary_context,
            run_stream,
            driver_version,
            context_api_version,
            nvcc_version,
            native_sass_target: sass_target,
            vector_ta_build_sha256,
            gpu_cuda_build_sha256,
            exact_math_authority,
            phase_one_free_bytes_snapshot: free_memory_bytes_snapshot,
            allocator_context_reserve_bytes,
            data_population_limits: None,
        })
        .map_err(|error| {
            FullDiscoveryWorkspacePlanErrorV1::new(
                FullDiscoveryWorkspacePlanErrorCodeV1::RunDeviceCarrierSealingFailure,
                error.to_string(),
            )
        })
    }

    pub const fn admission_identity_sha256(&self) -> [u8; 32] {
        self.admission_identity_sha256
    }

    pub const fn workspace_plan_identity_sha256(&self) -> [u8; 32] {
        self.workspace_plan_identity_sha256
    }

    pub const fn selected_device_ordinal(&self) -> u32 {
        self.selected_device_ordinal
    }
}

#[cfg(any(all(test, feature = "cuda"), feature = "cuda-device-fixtures"))]
pub fn seal_test_full_discovery_run_v1(
    admission: SealedDiscoveryRunDeviceAdmissionV1,
    always_resident_fixture_bytes: u64,
    phase_scratch_fixture_bytes: u64,
) -> Result<AdmittedNativeCudaFullDiscoveryRunV1, FullDiscoveryWorkspacePlanErrorV1> {
    if always_resident_fixture_bytes == 0 || phase_scratch_fixture_bytes == 0 {
        return Err(FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::MissingStageRequirement,
            "device fixture extents must be nonzero",
        ));
    }
    let identity = |label: &str| -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"neoethos.gpu-cuda.real-device-fixture.v1");
        hasher.update(label.as_bytes());
        hasher.finalize().into()
    };
    let phase = |stage: FullDiscoveryWorkspaceStageV1,
                 scratch_bytes: u64,
                 start: u32,
                 end: u32,
                 consumer_event: &str,
                 producer_event: &str| {
        OpaqueFullDiscoveryPhasePreflightV1 {
            stage,
            scratch_bytes,
            lifetime_interval_start: start,
            lifetime_interval_end: end,
            producer_completion_event_identity_sha256: identity(producer_event),
            consumer_wait_event_identity_sha256: identity(consumer_event),
            phase_identity_sha256: identity(stage.wire_name()),
        }
    };
    let stage = |stage: FullDiscoveryWorkspaceStageV1, device_bytes: u64| {
        OpaqueFullDiscoveryStagePreflightV1 {
            stage,
            device_bytes,
            stage_identity_sha256: identity(stage.wire_name()),
        }
    };
    let plan =
        seal_full_discovery_gpu_workspace_plan_v1(FullDiscoveryWorkspacePreflightBundleV1 {
            resident_feature_store: stage(
                FullDiscoveryWorkspaceStageV1::ResidentFeatureStore,
                always_resident_fixture_bytes,
            ),
            population_parent_and_views: stage(
                FullDiscoveryWorkspaceStageV1::PopulationParentAndViews,
                1,
            ),
            resident_genetic_evolution: stage(
                FullDiscoveryWorkspaceStageV1::ResidentGeneticEvolution,
                1,
            ),
            walk_forward_validation: phase(
                FullDiscoveryWorkspaceStageV1::WalkForwardValidation,
                phase_scratch_fixture_bytes,
                0,
                1,
                "fixture-phase-event-0",
                "fixture-phase-event-1",
            ),
            cpcv_and_pbo: phase(
                FullDiscoveryWorkspaceStageV1::CpcvAndPbo,
                phase_scratch_fixture_bytes,
                1,
                2,
                "fixture-phase-event-1",
                "fixture-phase-event-2",
            ),
            outer_holdout_and_oos: phase(
                FullDiscoveryWorkspaceStageV1::OuterHoldoutAndOos,
                phase_scratch_fixture_bytes,
                2,
                3,
                "fixture-phase-event-2",
                "fixture-phase-event-3",
            ),
            portfolio_constraints: stage(FullDiscoveryWorkspaceStageV1::PortfolioConstraints, 1),
            robustness_tails: phase(
                FullDiscoveryWorkspaceStageV1::RobustnessTails,
                phase_scratch_fixture_bytes,
                3,
                4,
                "fixture-phase-event-3",
                "fixture-phase-event-4",
            ),
            final_compact_readback: OpaqueFullDiscoveryFinalReadbackPreflightV1 {
                bounded_bytes: 160,
                stage_identity_sha256: identity("terminal-compact-result-v1"),
            },
            workspace_semantics: OpaqueFullDiscoveryWorkspaceSemanticsPreflightV1 {
                discovery_semantics_version: DISCOVERY_SEMANTICS_VERSION.to_owned(),
                discovery_semantics_sha256: identity(DISCOVERY_SEMANTICS_VERSION),
                population_gap_flags_bytes: 1,
                population_gene_rank_rng_bytes: 1,
                population_fixed_native_buffers_bytes: 1,
                population_view_index_adaptive_bytes: 1,
                allocator_context_reserve_bytes: 64 * 1024 * 1024,
                vector_ta_build_sha256: identity(
                    "fixture-vector-ta-build-not-executed-by-this-seam-test",
                ),
                exact_math_authority: vector_ta::cuda::F64_EXACT_MATH_AUTHORITY_V3.to_owned(),
                stage_identity_sha256: identity("workspace-semantics"),
            },
        })?;
    match bind_full_discovery_workspace_plan_v1(admission, plan)? {
        AdmittedFullDiscoveryGpuRunV1::NativeCuda(run) => Ok(run),
    }
}

#[cfg(any(all(test, feature = "cuda"), feature = "cuda-device-fixtures"))]
pub(crate) fn seal_test_full_discovery_run_device_v3(
    admission: SealedDiscoveryRunDeviceAdmissionV1,
    always_resident_fixture_bytes: u64,
    phase_scratch_fixture_bytes: u64,
) -> Result<GpuOnlyRunDeviceAdmissionV3, FullDiscoveryWorkspacePlanErrorV1> {
    seal_test_full_discovery_run_v1(
        admission,
        always_resident_fixture_bytes,
        phase_scratch_fixture_bytes,
    )?
    .into_gpu_only_run_device_admission_v3()
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct SealedFullDiscoveryGpuExecutionEvidenceV1 {
    intermediate_device_to_host_count: u64,
    intermediate_device_to_host_bytes: u64,
    final_compact_readback_count: u64,
    final_compact_readback_bytes: u64,
}

#[derive(Debug)]
pub struct FullDiscoveryGpuRunReceiptV1 {
    admission_identity_sha256: [u8; 32],
    workspace_plan_identity_sha256: [u8; 32],
    intermediate_device_to_host_count: u64,
    intermediate_device_to_host_bytes: u64,
    final_compact_readback_count: u64,
    final_compact_readback_bytes: u64,
    final_compact_readback_limit_bytes: u64,
    receipt_identity_sha256: [u8; 32],
}

#[cfg(feature = "cuda")]
impl AdmittedNativeCudaFullDiscoveryRunV1 {
    pub fn finish(
        self,
        evidence: SealedFullDiscoveryGpuExecutionEvidenceV1,
    ) -> Result<FullDiscoveryGpuRunReceiptV1, FullDiscoveryWorkspacePlanErrorV1> {
        seal_full_discovery_gpu_run_receipt_v1(self, evidence)
    }
}

#[cfg(feature = "cuda")]
fn seal_full_discovery_gpu_run_receipt_v1(
    run: AdmittedNativeCudaFullDiscoveryRunV1,
    evidence: SealedFullDiscoveryGpuExecutionEvidenceV1,
) -> Result<FullDiscoveryGpuRunReceiptV1, FullDiscoveryWorkspacePlanErrorV1> {
    let intermediate_device_to_host_count = evidence.intermediate_device_to_host_count;
    let intermediate_device_to_host_bytes = evidence.intermediate_device_to_host_bytes;
    let final_compact_readback_count = evidence.final_compact_readback_count;
    let final_compact_readback_bytes = evidence.final_compact_readback_bytes;
    let final_compact_readback_limit_bytes = run.plan.bounded_final_readback_bytes;
    if !(intermediate_device_to_host_count == 0
        && intermediate_device_to_host_bytes == 0
        && final_compact_readback_count == 1
        && final_compact_readback_bytes <= final_compact_readback_limit_bytes)
    {
        return Err(FullDiscoveryWorkspacePlanErrorV1::new(
            FullDiscoveryWorkspacePlanErrorCodeV1::InvalidCompletionEvidence,
            "strict GPU completion requires zero intermediate D2H and one bounded final readback",
        ));
    }
    let receipt_identity_sha256 =
        hash_full_discovery_receipt_v1(&run, final_compact_readback_bytes);
    Ok(FullDiscoveryGpuRunReceiptV1 {
        admission_identity_sha256: run.admission_identity_sha256,
        workspace_plan_identity_sha256: run.workspace_plan_identity_sha256,
        intermediate_device_to_host_count,
        intermediate_device_to_host_bytes,
        final_compact_readback_count,
        final_compact_readback_bytes,
        final_compact_readback_limit_bytes,
        receipt_identity_sha256,
    })
}

#[cfg(feature = "cuda")]
fn hash_full_discovery_receipt_v1(
    run: &AdmittedNativeCudaFullDiscoveryRunV1,
    final_compact_readback_bytes: u64,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.full-discovery-gpu-run-receipt.v1");
    hasher.update(run.physical_inventory_identity_sha256);
    hasher.update(run.admission_identity_sha256);
    hasher.update(run.workspace_plan_identity_sha256);
    hasher.update(run.device_uuid);
    hasher.update(run.pci_identity.pci_domain().to_le_bytes());
    hasher.update([
        run.pci_identity.pci_bus(),
        run.pci_identity.pci_device(),
        run.pci_identity.pci_function(),
    ]);
    hasher.update((run.primary_context.as_raw() as usize as u64).to_le_bytes());
    hasher.update((run.run_stream.as_inner() as usize as u64).to_le_bytes());
    hasher.update(run.cuda_build_identity.manifest_sha256);
    hasher.update(run.cuda_build_identity.artifact_sha256);
    hasher.update(run.cuda_build_identity.nvcc_version.as_bytes());
    hasher.update(run.sass_target.as_bytes());
    hasher.update(run.driver_version.as_bytes());
    hasher.update(run.context_api_version.as_bytes());
    hasher.update(run.compute_capability_major.to_le_bytes());
    hasher.update(run.compute_capability_minor.to_le_bytes());
    hasher.update(run.free_memory_bytes_snapshot.to_le_bytes());
    hasher.update(final_compact_readback_bytes.to_le_bytes());
    hasher.finalize().into()
}

impl FullDiscoveryGpuRunReceiptV1 {
    pub const fn receipt_identity_sha256(&self) -> [u8; 32] {
        self.receipt_identity_sha256
    }

    pub const fn intermediate_device_to_host_count(&self) -> u64 {
        self.intermediate_device_to_host_count
    }

    pub const fn final_compact_readback_bytes(&self) -> u64 {
        self.final_compact_readback_bytes
    }

    pub const fn admission_identity_sha256(&self) -> [u8; 32] {
        self.admission_identity_sha256
    }

    pub const fn workspace_plan_identity_sha256(&self) -> [u8; 32] {
        self.workspace_plan_identity_sha256
    }

    pub const fn intermediate_device_to_host_bytes(&self) -> u64 {
        self.intermediate_device_to_host_bytes
    }

    pub const fn final_compact_readback_count(&self) -> u64 {
        self.final_compact_readback_count
    }

    pub const fn final_compact_readback_limit_bytes(&self) -> u64 {
        self.final_compact_readback_limit_bytes
    }
}
