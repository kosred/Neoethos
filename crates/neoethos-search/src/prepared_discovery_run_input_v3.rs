//! Exclusive CPU-or-native input preparation for one canonical Discovery run.
//!
//! The dispatcher performs the physical inventory/CUDA admission exactly once
//! before either factory can materialize data. A physical-GPU-free machine may
//! build the owned host input. A selected CUDA device instead consumes the
//! admitted full-workspace carrier into Data's sealed resident store.

use crate::data_selection::{
    CanonicalGpuResidentSearchArtifactScopeV3, CanonicalGpuResidentSearchInputReceiptV3,
    CanonicalSearchInput, CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
};
use crate::gpu_resident_current_config_plan_v1::{
    CurrentConfigResidentSearchAdmissionFactsV1, SealedCurrentConfigResidentSearchPlanV1,
    seal_current_config_resident_search_plan_v1,
};
use crate::gpu_resident_trim_prefilter_view_v1::{
    begin_gpu_resident_trim_prefilter_view_v1, execute_gpu_resident_trim_prefilter_view_v1,
    resolve_current_config_resident_trim_prefilter_plan_v1,
    seal_gpu_resident_trim_prefilter_view_v1,
};
use crate::prefilter_schema_v1::seal_prefilter_column_classification_v1;
use crate::resident_population_auto_sizing_receipt_v2::{
    RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2, ResidentPopulationAutoSizingReceiptV2,
    evaluation_config_from_canonical_trendbar_contract_v2,
    seal_resident_population_auto_for_canonical_trendbar_research_v2,
    seal_resident_population_auto_for_canonical_trendbar_research_with_hard_cap_v2,
};
use crate::strict_discovery_device_route_v1::SealedStrictDiscoveryDeviceAdmissionV1;
use crate::strict_resident_feature_store_v3::{
    StrictResidentPopulationExecutionRunV3, bind_strict_resident_feature_store_v3_run_input,
    record_resident_feature_store_consumer_completion_v3,
    validate_strict_resident_feature_store_v3,
};
use crate::{DiscoveryConfig, DiscoveryProgress, DiscoveryResult, PropFirmRiskRules};
use anyhow::{Context, Result, bail, ensure};
use neoethos_data::SealedGpuResidentFeatureStoreV3;
use neoethos_gpu_cuda::full_discovery_workspace_plan_v1::AdmittedNativeCudaFullDiscoveryRunV1;
use neoethos_gpu_cuda::resident_feature_store_v3::ResidentTrimPrefilterSchemaUploadV1;
use neoethos_gpu_cuda::resident_trim_prefilter_v1::ResidentTrimmedPopulationSessionV1;
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
            let PreparedNativeCudaCanonicalDiscoveryRunInputV3 {
                receipt,
                feature_names,
                sealed_store,
            } = native;
            let scope = CanonicalGpuResidentSearchArtifactScopeV3::for_entire_receipt(
                CanonicalSearchWindowRoleV1::DiscoveryInput,
                receipt,
            )
            .context("bind native resident parent to the canonical Search receipt")?;
            validate_strict_resident_feature_store_v3(&sealed_store, &scope)
                .context("validate the resident parent before current-config preflight")?;
            let parent_rows = usize::try_from(sealed_store.contract().layout().row_count())
                .context("resident parent row count does not fit this process")?;
            let parent_columns =
                usize::try_from(sealed_store.contract().layout().column_count())
                    .context("resident parent column count does not fit this process")?;

            // This admission is deliberately resolved before the compact
            // schema upload or any trim allocation. Slice 1 has no authority
            // to invent archive-kNN throughput or memory-pool identity facts.
            let admission =
                require_current_config_resident_search_admission_facts_v1(&scope, &sealed_store)?;
            let runtime = crate::genetic::current_genetic_search_runtime_overrides();
            let current_config_plan = seal_current_config_resident_search_plan_v1(
                config,
                &runtime,
                parent_rows,
                parent_columns,
                admission,
            )
            .context("seal current-config resident Search plan before trim allocation")?;
            let trimmed_population = consume_native_store_into_trimmed_population_v1(
                sealed_store,
                &scope,
                feature_names,
                config,
                &current_config_plan,
            )?;
            run_native_cuda_prepared_discovery_v3(
                trimmed_population,
                current_config_plan,
                config,
                prop_firm_rules,
                &mut progress_fn,
            )
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

fn require_current_config_resident_search_admission_facts_v1(
    scope: &CanonicalGpuResidentSearchArtifactScopeV3,
    sealed_store: &SealedGpuResidentFeatureStoreV3,
) -> Result<CurrentConfigResidentSearchAdmissionFactsV1> {
    scope
        .validate()
        .context("validate current-config Search scope before admission")?;
    ensure!(
        sealed_store.admission_identity_sha256() != [0; 32]
            && sealed_store
                .device_identity()
                .primary_context_process_token()
                != [0; 32],
        "resident store lacks its sealed CUDA admission identity"
    );
    bail!(
        "current-config resident Search requires an actual run memory-pool identity, a measured exact archive-kNN popcount calibration receipt, and the native-query/calibrated resident trim workspace preflight before any allocation; Slice 1 refuses to fabricate admission facts"
    )
}

fn decode_lower_hex_sha256_v1(value: &str, field: &'static str) -> Result<[u8; 32]> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} is not a canonical lowercase SHA-256"
    );
    let mut decoded = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |byte: u8| -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => unreachable!("canonical lowercase SHA-256 was validated"),
            }
        };
        decoded[index] = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Ok(decoded)
}

fn consume_native_store_into_trimmed_population_v1(
    sealed_store: SealedGpuResidentFeatureStoreV3,
    scope: &CanonicalGpuResidentSearchArtifactScopeV3,
    feature_names: Vec<String>,
    config: &DiscoveryConfig,
    current_config_plan: &SealedCurrentConfigResidentSearchPlanV1,
) -> Result<ResidentTrimmedPopulationSessionV1> {
    validate_strict_resident_feature_store_v3(&sealed_store, scope)
        .context("revalidate resident parent before moving it into trim")?;
    let resident_feature_names = sealed_store
        .ordered_feature_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(
        feature_names == resident_feature_names,
        "prepared feature order drifted from the sealed resident store"
    );
    let classification = seal_prefilter_column_classification_v1(&feature_names)
        .context("seal the shared CPU/resident prefilter schema classification")?;
    let canonical_receipt_identity = decode_lower_hex_sha256_v1(
        &scope
            .receipt()
            .identity_sha256()
            .context("seal canonical resident input receipt identity")?,
        "canonical resident input receipt identity",
    )?;
    let schema_upload = ResidentTrimPrefilterSchemaUploadV1::new(
        canonical_receipt_identity,
        sealed_store
            .contract()
            .canonical_feature_content_merkle_sha256(),
        sealed_store.normalization_fit_sha256(),
        sealed_store.final_feature_plan_v3_sha256(),
        sealed_store.source_provenance_sha256(),
        classification.ordered_feature_schema_sha256(),
        classification.column_classification_content_sha256(),
        classification.column_class_flags().to_vec(),
        classification.timeframe_group_ids().to_vec(),
        classification.template_force_keep_flags().to_vec(),
        classification.timeframe_group_count(),
    )
    .context("seal the compact resident trim schema upload")?;
    let resident_import = sealed_store
        .into_resident_feature_store_import_v3()
        .context("move the sealed resident store into its admitted-stream import")?;
    let trim_inputs = resident_import
        .consume_into_resident_trim_prefilter_v1(schema_upload)
        .context("move the resident feature store into the trim owner")?;
    let import_identity = *trim_inputs.identity();
    let resolved_plan = resolve_current_config_resident_trim_prefilter_plan_v1(
        config,
        current_config_plan,
        &import_identity,
        &classification,
        None,
    )
    .context("bind the trim plan to the exact current-config Search admission")?;
    let (parent, schema, admission) = trim_inputs.into_parts();
    let trim_run =
        begin_gpu_resident_trim_prefilter_view_v1(parent, schema, admission, resolved_plan)
            .context("begin the admitted resident trim/prefilter run")?;
    let trim_run = execute_gpu_resident_trim_prefilter_view_v1(trim_run)
        .context("enqueue the resident trim/prefilter stages")?;
    let sealed_views = seal_gpu_resident_trim_prefilter_view_v1(trim_run)
        .context("seal the resident trim/prefilter device views")?;
    sealed_views
        .consume_into_population_session_v3()
        .context("move sealed trim views into the resident population owner")
}

fn run_native_cuda_prepared_discovery_v3<F>(
    trimmed_population: ResidentTrimmedPopulationSessionV1,
    current_config_plan: SealedCurrentConfigResidentSearchPlanV1,
    _config: &DiscoveryConfig,
    _prop_firm_rules: PropFirmRiskRules,
    progress_fn: &mut F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    ensure!(
        current_config_plan.plan_identity_sha256() != [0; 32]
            && trimmed_population.population_rows() == current_config_plan.parent_row_range().end
            && trimmed_population.parent_columns() == current_config_plan.parent_column_count()
            && trimmed_population.selected_compact_to_parent_columns_device()
            && trimmed_population.selected_column_count_device()
            && trimmed_population.same_selected_column_map_for_holdout()
            && trimmed_population.has_zero_trim_host_boundary(),
        "native resident trim/population carrier drifted before Search execution"
    );
    progress_fn(DiscoveryProgress::StageAdvanced {
        stage: "gpu_native_trim_prefilter",
        detail: "sealed resident input was move-consumed through the real GPU-native trim/prefilter owner into the population carrier".to_owned(),
    });
    bail!(
        "resident multi-generation archive-kNN Search has no consumer for the sealed trim/population carrier yet; refusing host materialization, CPU fallback, or an armed-carrier readiness claim"
    )
}
