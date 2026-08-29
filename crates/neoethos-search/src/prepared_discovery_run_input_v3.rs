//! Exclusive CPU-or-native input preparation for one canonical Discovery run.
//!
//! The dispatcher performs the physical inventory/CUDA admission exactly once
//! before either factory can materialize data. A physical-GPU-free machine may
//! build the owned host input. A selected CUDA device instead consumes the
//! admitted full-workspace carrier into Data's sealed resident store.

use crate::data_selection::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchInput, CanonicalSearchInputReceiptV2,
    CanonicalSearchWindowRoleV1,
};
use crate::strict_discovery_device_route_v1::SealedStrictDiscoveryDeviceAdmissionV1;
use crate::strict_resident_feature_store_v3::{
    StrictResidentPopulationExecutionRunV3, bind_strict_resident_feature_store_v3_run_input,
    record_resident_feature_store_consumer_completion_v3,
};
use crate::{DiscoveryConfig, DiscoveryProgress, DiscoveryResult, PropFirmRiskRules};
use anyhow::{Context, Result, bail, ensure};
use neoethos_data::SealedGpuResidentFeatureStoreV3;
use neoethos_gpu_cuda::full_discovery_workspace_plan_v1::AdmittedNativeCudaFullDiscoveryRunV1;
use neoethos_gpu_cuda::run_device_admission_v1::SealedCpuNoPhysicalGpuRunDeviceAdmissionV1;
use neoethos_gpu_cuda::{
    AdmittedFullDiscoveryGpuRunV1, SealedDiscoveryRunDeviceAdmissionV1,
    SealedFullDiscoveryGpuWorkspacePlanV1, acquire_discovery_run_device_admission_v1,
    bind_full_discovery_workspace_plan_v1,
};

#[derive(Debug)]
pub struct PreparedCpuCanonicalDiscoveryRunInputV3 {
    input: CanonicalSearchInput,
    receipt: CanonicalSearchInputReceiptV2,
    feature_names: Vec<String>,
    no_physical_gpu_admission: SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
}

#[derive(Debug)]
pub struct PreparedNativeCudaCanonicalDiscoveryRunInputV3 {
    receipt: CanonicalSearchInputReceiptV2,
    feature_names: Vec<String>,
    sealed_store: SealedGpuResidentFeatureStoreV3,
}

#[derive(Debug)]
pub enum PreparedCanonicalDiscoveryRunInputV3 {
    Cpu(PreparedCpuCanonicalDiscoveryRunInputV3),
    NativeCuda(PreparedNativeCudaCanonicalDiscoveryRunInputV3),
}

/// CPU-only combined research/training handoff. The exact input used by
/// Discovery stays owned and is moved into training after the research result
/// is sealed; it is never reconstructed from a path or current-generation
/// pointer.
pub struct PreparedCpuCanonicalTrendbarResearchRunV3 {
    research: crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3,
    training_input: CanonicalSearchInput,
}

impl PreparedCpuCanonicalTrendbarResearchRunV3 {
    pub const fn research_result(
        &self,
    ) -> &crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3 {
        &self.research
    }

    pub fn into_parts(
        self,
    ) -> (
        crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3,
        CanonicalSearchInput,
    ) {
        (self.research, self.training_input)
    }
}

impl PreparedCanonicalDiscoveryRunInputV3 {
    pub const fn receipt(&self) -> &CanonicalSearchInputReceiptV2 {
        match self {
            Self::Cpu(prepared) => &prepared.receipt,
            Self::NativeCuda(prepared) => &prepared.receipt,
        }
    }

    pub fn shape(&self) -> Result<(usize, usize)> {
        match self {
            Self::Cpu(prepared) => Ok((
                prepared.input.features().n_samples(),
                prepared.input.features().n_features(),
            )),
            Self::NativeCuda(prepared) => Ok((
                usize::try_from(prepared.sealed_store.contract().layout().row_count())
                    .context("resident V3 row count does not fit this process")?,
                usize::try_from(prepared.sealed_store.contract().layout().column_count())
                    .context("resident V3 column count does not fit this process")?,
            )),
        }
    }

    /// Metadata-only feature ordering used by the streaming survivor remap.
    /// The native branch copies names, never feature values, from the sealed
    /// resident-store contract.
    pub fn feature_names(&self) -> &[String] {
        match self {
            Self::Cpu(prepared) => &prepared.feature_names,
            Self::NativeCuda(prepared) => &prepared.feature_names,
        }
    }
}

/// The single cross-crate admission dispatcher. Callers may pin immutable
/// generation leases before entering this function, but only the selected arm
/// receives authority to materialize values. The CPU arm must return the same
/// opaque zero-physical-GPU admission; the native arm receives the admitted
/// full-workspace run by value.
pub fn dispatch_canonical_discovery_data_preparation_v3<
    Output,
    CpuPayload,
    CpuFactory,
    FinishCpu,
    NativePlanFactory,
    NativeFactory,
>(
    cpu_factory: CpuFactory,
    finish_cpu: FinishCpu,
    native_workspace_plan_factory: NativePlanFactory,
    native_factory: NativeFactory,
) -> Result<Output>
where
    CpuFactory: FnOnce(
        SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    ) -> Result<(CpuPayload, SealedCpuNoPhysicalGpuRunDeviceAdmissionV1)>,
    FinishCpu: FnOnce(CpuPayload, SealedCpuNoPhysicalGpuRunDeviceAdmissionV1) -> Result<Output>,
    NativePlanFactory: FnOnce() -> Result<SealedFullDiscoveryGpuWorkspacePlanV1>,
    NativeFactory: FnOnce(AdmittedNativeCudaFullDiscoveryRunV1) -> Result<Output>,
{
    let admission = acquire_discovery_run_device_admission_v1()
        .context("acquire one physical-inventory/CUDA Discovery run admission")?;
    let expected_admission_identity = admission.admission_identity_sha256();
    let counters = admission.probe_counters();
    let physical_inventory_probe_count = counters.physical_inventory_probe_count();
    let cuda_enumeration_count = counters.cuda_enumeration_count();
    let primary_context_acquisition_count = counters.primary_context_acquisition_count();
    let run_stream_creation_count = counters.run_stream_creation_count();

    match admission {
        SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(cpu_admission) => {
            ensure!(
                physical_inventory_probe_count == 1
                    && cuda_enumeration_count == 1
                    && primary_context_acquisition_count == 0
                    && run_stream_creation_count == 0,
                "CPU Discovery preparation lacks one complete physical inventory/CUDA enumeration"
            );
            let (payload, cpu_admission) = cpu_factory(cpu_admission)
                .context("materialize CPU Discovery data behind sealed physical-GPU absence")?;
            let returned = SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(cpu_admission);
            ensure!(
                returned.admission_identity_sha256() == expected_admission_identity,
                "CPU input factory returned a different physical-GPU absence authority"
            );
            let SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(cpu_admission) = returned
            else {
                unreachable!("the CPU factory result was wrapped as the CPU variant")
            };
            finish_cpu(payload, cpu_admission)
        }
        SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native_admission) => {
            ensure!(
                physical_inventory_probe_count == 1
                    && cuda_enumeration_count == 1
                    && primary_context_acquisition_count == 1
                    && run_stream_creation_count == 1,
                "native Discovery preparation lacks one context and one admitted run stream"
            );
            let workspace_plan = native_workspace_plan_factory()
                .context("seal complete native Discovery workspace plan")?;
            let admitted = bind_full_discovery_workspace_plan_v1(
                SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native_admission),
                workspace_plan,
            )
            .context("bind full Discovery workspace to the one admitted CUDA run")?;
            let admitted_native = match admitted {
                AdmittedFullDiscoveryGpuRunV1::NativeCuda(admitted_native) => admitted_native,
            };
            native_factory(admitted_native)
                .context("materialize native Discovery data on the admitted CUDA run")
        }
    }
}

pub fn prepare_canonical_discovery_run_input_v3<CpuFactory, NativePlanFactory, NativeFactory>(
    cpu_factory: CpuFactory,
    native_workspace_plan_factory: NativePlanFactory,
    native_factory: NativeFactory,
) -> Result<PreparedCanonicalDiscoveryRunInputV3>
where
    CpuFactory: FnOnce(
        SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    ) -> Result<(
        CanonicalSearchInput,
        SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    )>,
    NativePlanFactory: FnOnce() -> Result<SealedFullDiscoveryGpuWorkspacePlanV1>,
    NativeFactory: FnOnce(
        AdmittedNativeCudaFullDiscoveryRunV1,
    ) -> Result<(
        CanonicalSearchInputReceiptV2,
        SealedGpuResidentFeatureStoreV3,
    )>,
{
    dispatch_canonical_discovery_data_preparation_v3(
        cpu_factory,
        |input, cpu_admission| {
            let receipt = input
                .receipt()
                .context("seal prepared CPU canonical Search receipt")?;
            let feature_names = input.features().names.clone();
            Ok(PreparedCanonicalDiscoveryRunInputV3::Cpu(
                PreparedCpuCanonicalDiscoveryRunInputV3 {
                    input,
                    receipt,
                    feature_names,
                    no_physical_gpu_admission: cpu_admission,
                },
            ))
        },
        native_workspace_plan_factory,
        |admitted_native| {
            let (receipt, sealed_store) = native_factory(admitted_native)?;
            receipt
                .validate()
                .context("validate native canonical Search receipt")?;
            let feature_names = sealed_store
                .ordered_feature_names()
                .map(str::to_owned)
                .collect();
            Ok(PreparedCanonicalDiscoveryRunInputV3::NativeCuda(
                PreparedNativeCudaCanonicalDiscoveryRunInputV3 {
                    receipt,
                    feature_names,
                    sealed_store,
                },
            ))
        },
    )
}

pub fn run_prepared_canonical_discovery_with_holdout_and_progress_v3<F>(
    prepared: PreparedCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    match prepared {
        PreparedCanonicalDiscoveryRunInputV3::Cpu(cpu) => {
            run_cpu_prepared_discovery_v3(cpu, config, prop_firm_rules, progress_fn)
        }
        PreparedCanonicalDiscoveryRunInputV3::NativeCuda(native) => {
            let scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
                CanonicalSearchWindowRoleV1::DiscoveryInput,
                native.receipt,
            )
            .context("bind native resident parent to the canonical Search receipt")?;
            let run = bind_strict_resident_feature_store_v3_run_input(native.sealed_store, &scope)?;
            let view = seal_gpu_native_trim_prefilter_view_identity_v3(&run);
            let outcome = view.and_then(|view| {
                run_native_cuda_prepared_discovery_v3(
                    &run,
                    view,
                    config,
                    prop_firm_rules,
                    &mut progress_fn,
                )
            });
            let expected_completion_shape = (run.row_count(), run.column_count());
            let consumer_completion_lease =
                record_resident_feature_store_consumer_completion_v3(run)
                    .context("complete native resident Search consumer before release")?;
            ensure!(
                consumer_completion_lease.rows() == expected_completion_shape.0
                    && consumer_completion_lease.columns() == expected_completion_shape.1,
                "resident Search completion lease shape drifted from its consumed native run"
            );
            outcome
        }
    }
}

pub fn run_prepared_canonical_trendbar_research_with_holdout_and_progress_v3<F>(
    prepared: PreparedCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3>
where
    F: FnMut(DiscoveryProgress),
{
    contract.validate()?;
    ensure!(
        contract.input_receipt() == prepared.receipt(),
        "canonical-trendbar research contract does not match the prepared CPU/native input"
    );
    let mut research_config = config.clone();
    crate::discovery::apply_research_contract_to_discovery_config(&mut research_config, contract);
    let _research_execution =
        crate::canonical_trendbar_research::install_canonical_trendbar_research_execution_v3(
            contract,
        )?;
    let result = run_prepared_canonical_discovery_with_holdout_and_progress_v3(
        prepared,
        &research_config,
        prop_firm_rules,
        progress_fn,
    )?;
    crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3::new(
        contract.clone(),
        result,
    )
}

/// Run the combined canonical research/training CPU path while carrying the
/// exact owned Search input forward. A native input is refused before Search:
/// the native Discovery store cannot be converted back into a host feature
/// frame, and the GPU-resident training handoff is not integrated yet.
pub fn run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3<F>(
    prepared: PreparedCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    contract: &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<PreparedCpuCanonicalTrendbarResearchRunV3>
where
    F: FnMut(DiscoveryProgress),
{
    contract.validate()?;
    ensure!(
        contract.input_receipt() == prepared.receipt(),
        "canonical-trendbar research contract does not match the prepared CPU/native input"
    );
    let PreparedCanonicalDiscoveryRunInputV3::Cpu(cpu) = prepared else {
        bail!(
            "canonical combined discovery+training requires a GPU-resident training handoff; refusing to reconstruct a host FeatureFrame from the native resident Discovery store"
        );
    };
    let mut research_config = config.clone();
    crate::discovery::apply_research_contract_to_discovery_config(&mut research_config, contract);
    let _research_execution =
        crate::canonical_trendbar_research::install_canonical_trendbar_research_execution_v3(
            contract,
        )?;
    let (result, training_input) = run_cpu_prepared_discovery_v3_with_input(
        cpu,
        &research_config,
        prop_firm_rules,
        progress_fn,
    )?;
    let research =
        crate::canonical_trendbar_research::CanonicalTrendbarResearchDiscoveryResultV3::new(
            contract.clone(),
            result,
        )?;
    Ok(PreparedCpuCanonicalTrendbarResearchRunV3 {
        research,
        training_input,
    })
}

fn run_cpu_prepared_discovery_v3<F>(
    prepared: PreparedCpuCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    run_cpu_prepared_discovery_v3_with_input(prepared, config, prop_firm_rules, progress_fn)
        .map(|(result, _input)| result)
}

fn run_cpu_prepared_discovery_v3_with_input<F>(
    prepared: PreparedCpuCanonicalDiscoveryRunInputV3,
    config: &DiscoveryConfig,
    prop_firm_rules: PropFirmRiskRules,
    progress_fn: F,
) -> Result<(DiscoveryResult, CanonicalSearchInput)>
where
    F: FnMut(DiscoveryProgress),
{
    let strict_admission =
        SealedStrictDiscoveryDeviceAdmissionV1::from_no_physical_gpu_admission_v1(
            prepared.no_physical_gpu_admission,
        )
        .context("consume sealed physical-GPU absence into the CPU Discovery run")?;
    let input = prepared.input.as_run_input().map_err(anyhow::Error::new)?;
    let result = crate::discovery::run_discovery_cycle_with_prepared_cpu_admission_v3(
        &input,
        config,
        prop_firm_rules,
        strict_admission,
        progress_fn,
    )?;
    Ok((result, prepared.input))
}

struct GpuNativeTrimPrefilterViewIdentityV3 {
    parent_scope_identity_sha256: String,
    parent_rows: usize,
    parent_columns: usize,
}

fn seal_gpu_native_trim_prefilter_view_identity_v3(
    run: &StrictResidentPopulationExecutionRunV3,
) -> Result<GpuNativeTrimPrefilterViewIdentityV3> {
    let parent_scope_identity_sha256 = run
        .scope()
        .identity_sha256()
        .context("seal resident parent scope identity")?;
    ensure!(
        run.row_count() > 0 && run.column_count() > 0,
        "resident parent view is empty"
    );
    Ok(GpuNativeTrimPrefilterViewIdentityV3 {
        parent_scope_identity_sha256,
        parent_rows: run.row_count(),
        parent_columns: run.column_count(),
    })
}

fn run_native_cuda_prepared_discovery_v3<F>(
    run: &StrictResidentPopulationExecutionRunV3,
    view: GpuNativeTrimPrefilterViewIdentityV3,
    _config: &DiscoveryConfig,
    _prop_firm_rules: PropFirmRiskRules,
    progress_fn: &mut F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    ensure!(
        view.parent_scope_identity_sha256 == run.scope().identity_sha256()?
            && view.parent_rows == run.row_count()
            && view.parent_columns == run.column_count(),
        "native resident parent view identity drifted before GPU execution"
    );
    progress_fn(DiscoveryProgress::StageAdvanced {
        stage: "gpu_native_trim_prefilter",
        detail: "sealed resident input reached the GPU-native trim/prefilter boundary".to_owned(),
    });
    bail!(
        "GPU-native trim/prefilter and resident full-Discovery stage pipeline are not integrated; refusing host materialization or CPU fallback"
    )
}
