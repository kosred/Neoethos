//! Selected, source-bound HIP feature assembly over genuine producer owners.
//!
//! The physical pack/Merkle uses the existing native feature-store kernels.
//! This layer retains canonical source provenance and the exact requested
//! ordered recipe. It does not claim a fitted feature screen, a Search view,
//! holdout authorization, support for unimplemented families, or GPU parity.
//! Enabled startup normalization uses the shared native policy and Data's
//! original canonical split. Actual device fits bind the portable fitted state.

use std::collections::BTreeSet;

use anyhow::{Context as _, Result, ensure};
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOperationTagV1, FeatureOutputV1,
    FeatureParameterV1, FeaturePlanV1, SourceArtifactBindingV1,
};
use neoethos_gpu_contracts::device::{NeoPopulationSettings, ScenarioDescriptor};
use neoethos_gpu_contracts::resident_feature_store_v3::ResidentFeatureProducerV3;
use neoethos_gpu_cuda::hip_runtime_v1::feature_store_v1::{
    HipFeatureColumnV1, HipFeatureNormalizationV3, HipFeatureStorePlanV1, HipFeatureStoreReceiptV1,
    HipPhysicalPopulationMetricsV1, HipPopulationParentV1, HipSearchEvaluationBudgetV3,
    SealedHipFeatureStoreV1,
};
use neoethos_gpu_cuda::hip_runtime_v1::{
    HipDeviceBufferV1, HipRuntimeIdentityV1, HipRuntimeMemorySnapshotV1,
};
use neoethos_gpu_cuda::resident_search_slice2_v3::{
    HipResidentSearchBoundV3, ResidentSearchExecutionInputsV3, ResidentSearchGenerationChainV3,
};
use neoethos_gpu_cuda::{
    PopulationEvaluationViewV1, PopulationGeneView, SealedResidentGenerationPlanV1,
};
use sha2::{Digest, Sha256};

use super::gpu_hip_ohlcv_v1::ResidentHipOhlcvV1;
use super::gpu_hip_smc_v1::ResidentHipSmcV1;
use super::gpu_resident_robust_normalization_v2::{
    PreparedResidentRobustNormalizationInputV2, RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2,
    fitted_state_from_device_words, prepare_resident_robust_normalization_input_v2,
    seal_canonical_robust_normalization_split_from_hip_v1, search_normalization_column_mode_v3,
};
use super::normalization::{SEARCH_NORMALIZATION_POLICY_VERSION, SearchNormalizationFittedStateV1};

/// Immutable metadata derived by a real producer, not user-supplied evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HipSelectedColumnIdentityV1 {
    name: &'static str,
    producer: ResidentFeatureProducerV3,
    local_index: usize,
    semantic_version: u32,
    source_sha256: [u8; 32],
    build_manifest_sha256: [u8; 32],
    artifact_sha256: [u8; 32],
}

impl HipSelectedColumnIdentityV1 {
    pub const fn name(&self) -> &'static str {
        self.name
    }
    pub const fn producer(&self) -> ResidentFeatureProducerV3 {
        self.producer
    }
    pub const fn local_index(&self) -> usize {
        self.local_index
    }
    pub const fn semantic_version(&self) -> u32 {
        self.semantic_version
    }
}

/// Opaque borrowed column. Only an implemented producer can create one; no
/// safe caller can replace its name, shape, formula, buffer, offset, or lease.
pub struct HipResidentFeatureColumnV1<'producer, 'lease> {
    input: &'producer ResidentHipOhlcvV1<'lease>,
    values: &'producer HipDeviceBufferV1<'lease>,
    validity: &'producer HipDeviceBufferV1<'lease>,
    offset: usize,
    identity: HipSelectedColumnIdentityV1,
}

impl<'producer, 'lease> HipResidentFeatureColumnV1<'producer, 'lease> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_producer(
        input: &'producer ResidentHipOhlcvV1<'lease>,
        values: &'producer HipDeviceBufferV1<'lease>,
        validity: &'producer HipDeviceBufferV1<'lease>,
        names: &'static [&'static str],
        index: usize,
        producer: ResidentFeatureProducerV3,
        semantic_version: u32,
        source: &[u8],
        build_manifest_sha256: [u8; 32],
        artifact_sha256: [u8; 32],
    ) -> Result<Self> {
        let name = *names
            .get(index)
            .context("HIP producer column index is out of range")?;
        let rows = input.memory_plan().row_count();
        let cells = rows
            .checked_mul(names.len())
            .context("HIP producer extent overflow")?;
        let value_bytes = cells
            .checked_mul(8)
            .context("HIP producer value extent overflow")?;
        ensure!(
            values.len_bytes() == value_bytes
                && validity.len_bytes() == cells
                && values.lease_id() == input.lease().identity().lease_id()
                && validity.lease_id() == values.lease_id()
                && semantic_version > 0
                && !source.is_empty()
                && build_manifest_sha256 != [0; 32]
                && artifact_sha256 != [0; 32],
            "HIP producer descriptor disagrees with its actual sealed owner"
        );
        Ok(Self {
            input,
            values,
            validity,
            offset: rows
                .checked_mul(index)
                .context("HIP column offset overflow")?,
            identity: HipSelectedColumnIdentityV1 {
                name,
                producer,
                local_index: index,
                semantic_version,
                source_sha256: Sha256::digest(source).into(),
                build_manifest_sha256,
                artifact_sha256,
            },
        })
    }

    pub fn identity(&self) -> &HipSelectedColumnIdentityV1 {
        &self.identity
    }
}

fn ordered_names(columns: &[HipSelectedColumnIdentityV1]) -> Result<Vec<String>> {
    ensure!(
        !columns.is_empty(),
        "HIP canonical assembly requires selected columns"
    );
    let mut seen = BTreeSet::new();
    columns
        .iter()
        .map(|column| {
            ensure!(
                !column.name.trim().is_empty() && seen.insert(column.name),
                "HIP selected feature names must be nonempty and unique"
            );
            Ok(column.name.to_owned())
        })
        .collect()
}

fn feature_contract(
    source: &SourceArtifactBindingV1,
    columns: &[HipSelectedColumnIdentityV1],
) -> Result<(FeaturePlanV1, DatasetFeatureArtifactProvenanceV1)> {
    let final_outputs = ordered_names(columns)?;
    let source_id = source.source_node_id().to_owned();
    let physical_schema = b"neoethos.ohlcv.f64-ms.v1";
    let source_node = FeatureNodeV1::source(
        source_id.clone(),
        source.dataset_identity().clone(),
        std::str::from_utf8(physical_schema)?,
        1,
        ["open", "high", "low", "close", "volume"]
            .into_iter()
            .map(|name| FeatureOutputV1::f64(format!("physical:{name}"), 1))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        Sha256::digest(physical_schema).into(),
    )?;
    let mut nodes = vec![source_node];
    for (ordinal, column) in columns.iter().enumerate() {
        // The hash binds actual implementation/build semantics. Formula review
        // is separate: zero below must not be relabeled independent math proof.
        nodes.push(FeatureNodeV1::transform(
            format!(
                "hip:selected:{ordinal}:{}:{}",
                column.producer.as_str(),
                column.local_index
            ),
            FeatureOperationTagV1::Indicator,
            column.semantic_version,
            vec![source_id.clone()],
            vec![FeatureOutputV1::f64(column.name, column.semantic_version)?],
            vec![
                FeatureParameterV1::u64(
                    "producer_column_index",
                    u64::try_from(column.local_index)?,
                )?,
                FeatureParameterV1::text("producer", column.producer.as_str())?,
                FeatureParameterV1::text("backend", "amd-hip")?,
                FeatureParameterV1::bool("normalization_enabled", false)?,
                FeatureParameterV1::hash(
                    "native_build_manifest_sha256",
                    column.build_manifest_sha256,
                )?,
                FeatureParameterV1::hash("native_artifact_sha256", column.artifact_sha256)?,
            ],
            [0; 32],
            column.source_sha256,
            None,
        )?);
    }
    let plan = FeaturePlanV1::new(nodes, final_outputs)?;
    let provenance = DatasetFeatureArtifactProvenanceV1::new(&plan, vec![source.clone()])?;
    Ok((plan, provenance))
}

fn bind_fitted_feature_contract(
    raw: FeaturePlanV1,
    source: &SourceArtifactBindingV1,
    fitted: &SearchNormalizationFittedStateV1,
) -> Result<(FeaturePlanV1, DatasetFeatureArtifactProvenanceV1)> {
    let names = raw.final_outputs().to_vec();
    let inputs = raw
        .nodes()
        .iter()
        .filter(|node| node.operation() == FeatureOperationTagV1::Indicator)
        .map(|node| node.id().to_owned())
        .collect();
    let native_source_hash = Sha256::digest(include_bytes!(
        "../../../neoethos-gpu-cuda/native/resident_robust_normalization_v2.cu"
    ))
    .into();
    // Keep every producer's formula/dependencies and validity semantics, but
    // distinguish its raw slots from the wrapping transform's final outputs.
    let mut nodes = raw
        .nodes()
        .iter()
        .map(|node| {
            if node.operation() == FeatureOperationTagV1::Indicator {
                node.with_output_names(
                    node.outputs()
                        .iter()
                        .map(|output| format!("pre-normalize:{}", output.name()))
                        .collect(),
                )
            } else {
                Ok(node.clone())
            }
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    nodes.push(FeatureNodeV1::transform(
        "normalization:robust-f64",
        FeatureOperationTagV1::Normalization,
        SEARCH_NORMALIZATION_POLICY_VERSION,
        inputs,
        names
            .iter()
            .map(|name| FeatureOutputV1::f64(name, SEARCH_NORMALIZATION_POLICY_VERSION))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        vec![
            FeatureParameterV1::text("backend", "amd-hip")?,
            FeatureParameterV1::u64("transform_semantic_version", 2)?,
        ],
        [0; 32],
        native_source_hash,
        Some(fitted.fitted_state_hash()?),
    )?);
    let plan = FeaturePlanV1::new(nodes, names)?;
    fitted.validate_plan(&plan)?;
    let provenance = DatasetFeatureArtifactProvenanceV1::new(&plan, vec![source.clone()])?;
    Ok((plan, provenance))
}

fn bind_normalization_geometry(
    physical: HipFeatureStorePlanV1,
    prepared: &PreparedResidentRobustNormalizationInputV2,
) -> Result<HipFeatureStorePlanV1> {
    let native = HipFeatureNormalizationV3::preflight(
        prepared.row_count(),
        prepared.training_rows(),
        physical
            .ordered_names()
            .iter()
            .map(|name| search_normalization_column_mode_v3(name))
            .collect(),
    )?;
    ensure!(
        prepared.enabled()
            && prepared.semantic_version() == SEARCH_NORMALIZATION_POLICY_VERSION
            && prepared.row_count() == physical.rows()
            && prepared.feature_column_count() == physical.columns()
            && prepared.normalization_scratch_bytes() == native.scratch_bytes()
            && prepared.fit_metadata_bytes() == native.fit_metadata_bytes()
            && prepared
                .padded_training_rows()
                .checked_mul(
                    physical
                        .columns()
                        .min(RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2)
                )
                .and_then(|slots| slots.checked_mul(8))
                == Some(native.scratch_bytes()),
        "HIP normalization geometry differs from the sealed shared Data preflight"
    );
    Ok(physical.with_normalization(native)?)
}

/// Requested-column preparation, not a fitted-screen selected-map receipt.
/// The SMC parent is retained even for a selection containing no SMC features,
/// because its genuine calendar and eleven-slot lanes belong to the dataset.
pub struct PreparedHipCanonicalFeatureStoreV1<'producer, 'lease> {
    parent: &'producer ResidentHipSmcV1<'lease>,
    columns: Vec<HipResidentFeatureColumnV1<'producer, 'lease>>,
    physical_plan: HipFeatureStorePlanV1,
    feature_plan: FeaturePlanV1,
    provenance: DatasetFeatureArtifactProvenanceV1,
    normalization: Option<PreparedResidentRobustNormalizationInputV2>,
    reserve: u64,
}

impl<'producer, 'lease> PreparedHipCanonicalFeatureStoreV1<'producer, 'lease> {
    pub fn preflight(
        parent: &'producer ResidentHipSmcV1<'lease>,
        columns: Vec<HipResidentFeatureColumnV1<'producer, 'lease>>,
        allocator_reserve_bytes: u64,
    ) -> Result<Self> {
        let mode = crate::sealed_data_runtime_normalization_mode_v2()?;
        let input = parent.inputs();
        let source = input.canonical_source()?;
        ensure!(
            source.artifact().identity() == source.binding().dataset_identity(),
            "HIP canonical source artifact/binding drift"
        );
        ensure!(
            columns
                .iter()
                .all(|column| input.same_upload_as(column.input)),
            "HIP selected columns must originate from the exact same shared OHLCV upload"
        );
        let identities = columns
            .iter()
            .map(|column| column.identity.clone())
            .collect::<Vec<_>>();
        let names = ordered_names(&identities)?;
        let mut physical_plan =
            HipFeatureStorePlanV1::preflight(input.memory_plan().row_count(), &names)?;
        let normalization = if mode.enabled() {
            let prepared = prepare_resident_robust_normalization_input_v2(
                seal_canonical_robust_normalization_split_from_hip_v1(input)?,
                names.len(),
            )?;
            physical_plan = bind_normalization_geometry(physical_plan, &prepared)?;
            Some(prepared)
        } else {
            // No fit and no artificial minimum-row restriction in disabled mode.
            None
        };
        let (feature_plan, provenance) = feature_contract(source.binding(), &identities)?;
        Ok(Self {
            parent,
            columns,
            physical_plan,
            feature_plan,
            provenance,
            normalization,
            reserve: allocator_reserve_bytes,
        })
    }

    pub fn physical_plan(&self) -> &HipFeatureStorePlanV1 {
        &self.physical_plan
    }
    pub fn feature_plan(&self) -> &FeaturePlanV1 {
        // Raw producer recipe only. Enabled normalization adds its fitted node
        // after genuine device completion, never during metadata preflight.
        &self.feature_plan
    }
    pub fn provenance(&self) -> &DatasetFeatureArtifactProvenanceV1 {
        &self.provenance
    }

    /// Complete one logical pack and Merkle seal. No feature arrays return to
    /// the CPU. A failure cannot produce a partially sealed Data store.
    pub fn materialize(self) -> Result<ResidentHipCanonicalFeatureStoreV1<'producer, 'lease>> {
        let input = self.parent.inputs();
        let columns = self
            .columns
            .iter()
            .map(|column| {
                HipFeatureColumnV1::new(
                    column.identity.name,
                    column.values,
                    column.offset,
                    column.validity,
                    column.offset,
                )
            })
            .collect::<Vec<_>>();
        let physical = self.physical_plan.pack_and_seal(
            input.lease(),
            input.lanes()[5],
            &columns,
            self.reserve,
        )?;
        let (feature_plan, provenance, normalization_fitted_state) =
            if let Some(prepared) = self.normalization {
                let words = physical
                    .normalization_fit_words()
                    .context("HIP enabled normalization omitted its actual fit words")?;
                let fitted = fitted_state_from_device_words(
                    physical.plan().ordered_names(),
                    prepared.training_rows(),
                    words,
                )?;
                let (plan, provenance) = bind_fitted_feature_contract(
                    self.feature_plan,
                    input.canonical_source()?.binding(),
                    &fitted,
                )?;
                (plan, provenance, Some(fitted))
            } else {
                ensure!(
                    physical.normalization_fit_words().is_none(),
                    "disabled HIP normalization unexpectedly returned fitted state"
                );
                (self.feature_plan, self.provenance, None)
            };
        let mut hash = Sha256::new();
        hash.update(b"neoethos.data.hip-canonical-assembly.v1\0");
        hash.update(feature_plan.identity().as_bytes());
        hash.update(provenance.identity().as_bytes());
        hash.update(input.identity().input_sha256());
        for parent_hash in self.parent.identity().generated_parent_sha256() {
            hash.update(parent_hash);
        }
        hash.update(physical.canonical_content_merkle_sha256());
        hash.update(input.lease().identity().lease_id().to_le_bytes());
        hash.update(input.lease().identity().stream_id().to_le_bytes());
        hash.update(self.reserve.to_le_bytes());
        Ok(ResidentHipCanonicalFeatureStoreV1 {
            physical,
            parent: self.parent,
            source_columns: self.columns,
            feature_plan,
            provenance,
            normalization_fitted_state,
            assembly_identity_sha256: hash.finalize().into(),
            allocator_reserve_bytes: self.reserve,
        })
    }
}

/// Physically sealed, source/recipe-bound selected HIP columns. No public raw
/// device handles, mutable output references, or conversion to CUDA admission.
/// Search still requires its own genuine run/view/holdout authority.
pub struct ResidentHipCanonicalFeatureStoreV1<'producer, 'lease> {
    physical: SealedHipFeatureStoreV1<'producer, 'lease>,
    parent: &'producer ResidentHipSmcV1<'lease>,
    source_columns: Vec<HipResidentFeatureColumnV1<'producer, 'lease>>,
    feature_plan: FeaturePlanV1,
    provenance: DatasetFeatureArtifactProvenanceV1,
    normalization_fitted_state: Option<SearchNormalizationFittedStateV1>,
    assembly_identity_sha256: [u8; 32],
    allocator_reserve_bytes: u64,
}

impl<'producer, 'lease> ResidentHipCanonicalFeatureStoreV1<'producer, 'lease> {
    /// Bind this exact source-backed assembly to the existing native cohort
    /// evaluator. OHLC/clock, generated calendar/SMC and packed features stay on
    /// the same lease; there is no parent re-upload or host feature readback.
    /// A caller-supplied evaluation view still needs Search's separate
    /// selection/holdout authorization before production discovery can use it.
    pub fn bind_population_v1(
        &self,
    ) -> Result<ResidentHipCanonicalPopulationV1<'_, 'producer, 'lease>> {
        let lanes = self.parent.inputs().lanes();
        let [months, days, smc] = self.parent.population_lanes_v1();
        let physical = self.physical.bind_population_parent_v1(
            lanes[3],
            lanes[1],
            lanes[2],
            months,
            days,
            smc,
            self.allocator_reserve_bytes,
        )?;
        Ok(ResidentHipCanonicalPopulationV1 {
            physical,
            store: self,
        })
    }

    /// Bind the same physical parent with explicit Search workspace in addition
    /// to the original allocator headroom. Capacity is the concurrent evaluation
    /// chunk, not a cap on the logical population. Adaptive rows are the exact
    /// retained view extent, or zero when adaptive stops are disabled.
    pub fn bind_population_for_search_v3(
        &self,
        retained_capacity: usize,
        month_capacity: u32,
        adaptive_view_rows: usize,
    ) -> Result<ResidentHipCanonicalPopulationV1<'_, 'producer, 'lease>> {
        let budget = HipSearchEvaluationBudgetV3::checked_v3(
            self.allocator_reserve_bytes,
            retained_capacity,
            month_capacity,
            adaptive_view_rows,
        )?;
        let lanes = self.parent.inputs().lanes();
        let [months, days, smc] = self.parent.population_lanes_v1();
        let physical = self.physical.bind_population_parent_for_search_v3(
            lanes[3], lanes[1], lanes[2], months, days, smc, budget,
        )?;
        Ok(ResidentHipCanonicalPopulationV1 {
            physical,
            store: self,
        })
    }

    pub fn feature_plan(&self) -> &FeaturePlanV1 {
        &self.feature_plan
    }
    pub fn provenance(&self) -> &DatasetFeatureArtifactProvenanceV1 {
        &self.provenance
    }
    pub fn normalization_fitted_state(&self) -> Option<&SearchNormalizationFittedStateV1> {
        self.normalization_fitted_state.as_ref()
    }
    pub fn physical_plan(&self) -> &HipFeatureStorePlanV1 {
        self.physical.plan()
    }
    pub fn physical_receipt(&self) -> &HipFeatureStoreReceiptV1 {
        self.physical.receipt()
    }
    pub fn canonical_content_merkle_sha256(&self) -> [u8; 32] {
        self.physical.canonical_content_merkle_sha256()
    }
    pub const fn assembly_identity_sha256(&self) -> [u8; 32] {
        self.assembly_identity_sha256
    }
    pub fn runtime_identity(&self) -> &HipRuntimeIdentityV1 {
        self.physical.runtime_identity()
    }
    pub fn memory_at_pack(&self) -> HipRuntimeMemorySnapshotV1 {
        self.physical.memory_at_pack()
    }
    pub fn selected_columns(&self) -> impl Iterator<Item = &HipSelectedColumnIdentityV1> {
        self.source_columns.iter().map(|column| &column.identity)
    }
    pub fn generated_parent_sha256(&self) -> &[[u8; 32]; 3] {
        self.parent.identity().generated_parent_sha256()
    }
    /// Producers remain borrowed until this store is dropped or closed. Their
    /// explicit cleanup follows this completed physical-store cleanup.
    pub fn try_close(self) -> Result<()> {
        self.physical.try_close().map_err(Into::into)
    }
}

/// Data keeps its genuine source/recipe/fit authority alive throughout physical
/// population ownership. Neither this guard nor its metric output asserts OOS
/// validity, financial configuration approval, or CPU/CUDA/HIP parity.
pub struct ResidentHipCanonicalPopulationV1<'store, 'producer, 'lease> {
    physical: HipPopulationParentV1<'store, 'producer, 'lease>,
    store: &'store ResidentHipCanonicalFeatureStoreV1<'producer, 'lease>,
}

impl<'store, 'producer, 'lease> ResidentHipCanonicalPopulationV1<'store, 'producer, 'lease> {
    pub fn feature_plan(&self) -> &FeaturePlanV1 {
        self.store.feature_plan()
    }
    pub fn provenance(&self) -> &DatasetFeatureArtifactProvenanceV1 {
        self.store.provenance()
    }
    pub fn normalization_fitted_state(&self) -> Option<&SearchNormalizationFittedStateV1> {
        self.store.normalization_fitted_state()
    }
    pub fn assembly_identity_sha256(&self) -> [u8; 32] {
        self.store.assembly_identity_sha256()
    }
    pub fn runtime_identity(&self) -> &HipRuntimeIdentityV1 {
        self.physical.runtime_identity()
    }

    /// Start the shared device-resident generation state machine on this exact
    /// Data parent. The returned chain borrows this guard through all phases,
    /// including terminal completion; no raw population owner can escape it.
    /// The caller must separately authorize its financial config and selected
    /// time interval. This physical bridge does not grant holdout access.
    pub fn begin_resident_search_slice2_v3(
        &mut self,
        plan: SealedResidentGenerationPlanV1,
        inputs: ResidentSearchExecutionInputsV3,
    ) -> Result<
        HipResidentSearchBoundV3<'_, 'store, 'producer, 'lease, ResidentSearchGenerationChainV3>,
    > {
        self.physical
            .begin_resident_search_slice2_v3(plan, inputs)
            .map_err(Into::into)
    }

    /// Evaluate all supplied candidates using the shared strict native engine.
    /// Only bounded genes/scenarios are uploaded and cohort metrics returned.
    pub fn evaluate_metrics_v1(
        &mut self,
        view: PopulationEvaluationViewV1,
        genes: PopulationGeneView<'_>,
        scenarios: &[ScenarioDescriptor],
        settings: &NeoPopulationSettings,
    ) -> Result<HipPhysicalPopulationMetricsV1> {
        self.physical
            .evaluate_metrics_v1(view, genes, scenarios, settings)
            .map_err(Into::into)
    }

    pub fn try_close(self) -> Result<()> {
        self.physical.try_close().map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_dataset_contracts::{
        BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
    };
    use neoethos_feature_contracts::SourceSegmentV1;

    fn column(name: &'static str, index: usize) -> HipSelectedColumnIdentityV1 {
        HipSelectedColumnIdentityV1 {
            name,
            producer: ResidentFeatureProducerV3::Smc,
            local_index: index,
            semantic_version: 3,
            source_sha256: [1; 32],
            build_manifest_sha256: [2; 32],
            artifact_sha256: [3; 32],
        }
    }

    // Host contract tests deliberately use synthetic metadata, not a native
    // owner or admission receipt. No constructor for a GPU store is bypassed.
    fn source(start: u64) -> SourceArtifactBindingV1 {
        SourceArtifactBindingV1::new(
            "source:test",
            CanonicalDatasetIdentity::external(
                "hip-contract-test",
                "EURUSD",
                CanonicalTimeframe::M1,
                BarTimestampConvention::BarOpen,
            )
            .unwrap(),
            "test-manifest",
            [7; 32],
            "test-generation",
            [8; 32],
            BarTimestampConvention::BarOpen,
            vec![
                SourceSegmentV1::new(start, start + 100, 1_700_000_000_000, 1_700_006_000_000)
                    .unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn selected_recipe_is_dynamic_exact_order_and_refuses_duplicates_or_empty() {
        assert!(ordered_names(&[]).is_err());
        assert!(ordered_names(&[column("same", 0), column("same", 1)]).is_err());
        assert!(ordered_names(&[column(" ", 0)]).is_err());
        let columns = [column("smc_fvg", 1), column("smc_ob", 0)];
        let (plan, _) = feature_contract(&source(0), &columns).unwrap();
        assert_eq!(plan.final_outputs(), ["smc_fvg", "smc_ob"]);
        let (reversed, _) =
            feature_contract(&source(0), &[columns[1].clone(), columns[0].clone()]).unwrap();
        assert_ne!(plan.identity(), reversed.identity());
        let (one, _) = feature_contract(&source(0), &columns[..1]).unwrap();
        assert_eq!(one.final_outputs().len(), 1);
        assert_ne!(one.identity(), plan.identity());
    }

    #[test]
    fn source_segment_and_producer_builds_are_not_replaced_by_content_hash() {
        let columns = vec![column("smc_ob", 0)];
        let (plan, provenance) = feature_contract(&source(0), &columns).unwrap();
        let (same_recipe, different_source) = feature_contract(&source(50), &columns).unwrap();
        assert_eq!(plan.identity(), same_recipe.identity());
        assert_ne!(provenance.identity(), different_source.identity());
        assert_eq!(different_source.bindings()[0].segments()[0].row_start(), 50);
        for field in 0..5 {
            let mut changed = columns.clone();
            match field {
                0 => changed[0].semantic_version += 1,
                1 => changed[0].local_index += 1,
                2 => changed[0].source_sha256[0] ^= 1,
                3 => changed[0].build_manifest_sha256[0] ^= 1,
                _ => changed[0].artifact_sha256[0] ^= 1,
            }
            assert_ne!(
                plan.identity(),
                feature_contract(&source(0), &changed).unwrap().0.identity()
            );
        }
    }

    #[test]
    fn disabled_normalization_has_no_fit_and_retains_short_input_support() {
        assert!(HipFeatureStorePlanV1::preflight(1, &["smc_ob".into()]).is_ok());
        let (plan, _) = feature_contract(&source(0), &[column("smc_ob", 0)]).unwrap();
        assert!(
            plan.nodes()
                .iter()
                .all(|node| node.operation() != FeatureOperationTagV1::Normalization)
        );
        assert!(
            plan.nodes()
                .iter()
                .all(|node| node.fitted_state_hash().is_none())
        );
    }

    #[test]
    fn actual_fit_words_bind_existing_portable_state_and_final_provenance() {
        // Synthetic host validation input, not a claimed device receipt.
        let names = vec!["smc_ob".to_owned(), "continuous".to_owned()];
        let words = [
            0,
            80,
            0.0f64.to_bits(),
            1.0f64.to_bits(),
            70,
            0,
            0,
            80,
            3.0f64.to_bits(),
            2.0f64.to_bits(),
            60,
            0,
        ];
        let fitted = fitted_state_from_device_words(&names, 0..80, &words).unwrap();
        assert_eq!(fitted.training_rows().unwrap(), 0..80);
        assert_eq!(fitted.fits()[1].median.to_bits(), 3.0f64.to_bits());
        let columns = [column("smc_ob", 0), column("continuous", 2)];
        let (raw, raw_provenance) = feature_contract(&source(0), &columns).unwrap();
        let raw_id = raw.identity();
        let raw_nodes = raw.nodes().to_vec();
        let (plan, provenance) = bind_fitted_feature_contract(raw, &source(0), &fitted).unwrap();
        fitted.validate_plan(&plan).unwrap();
        assert_eq!(plan.final_outputs(), names);
        assert_eq!(plan.nodes().len(), raw_nodes.len() + 1);
        // Round-trip only the renamed slots: equality proves no original
        // source, formula, parameters, edges, or validity semantics were lost.
        for original in &raw_nodes {
            let retained = plan
                .nodes()
                .iter()
                .find(|node| node.id() == original.id())
                .unwrap();
            if original.operation() == FeatureOperationTagV1::Indicator {
                assert_eq!(
                    retained
                        .outputs()
                        .iter()
                        .map(|output| output.name())
                        .collect::<Vec<_>>(),
                    original
                        .outputs()
                        .iter()
                        .map(|output| format!("pre-normalize:{}", output.name()))
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    &retained
                        .with_output_names(
                            original
                                .outputs()
                                .iter()
                                .map(|output| output.name().to_owned())
                                .collect()
                        )
                        .unwrap(),
                    original
                );
            } else {
                assert_eq!(retained, original);
            }
        }
        assert_ne!(raw_id, plan.identity());
        // Provenance binds the unchanged source generation and row segments;
        // the transform and fitted values are bound by the separate plan ID.
        assert_eq!(raw_provenance, provenance);
        let mut changed_words = words;
        changed_words[8] = 4.0f64.to_bits();
        let changed_fit = fitted_state_from_device_words(&names, 0..80, &changed_words).unwrap();
        let (raw_again, _) = feature_contract(&source(0), &columns).unwrap();
        let (changed_plan, same_source) =
            bind_fitted_feature_contract(raw_again, &source(0), &changed_fit).unwrap();
        assert_ne!(
            fitted.fitted_state_hash().unwrap(),
            changed_fit.fitted_state_hash().unwrap()
        );
        assert_ne!(plan.identity(), changed_plan.identity());
        assert_eq!(provenance, same_source);
        assert!(fitted.validate_plan(&changed_plan).is_err());
        assert!(changed_fit.validate_plan(&plan).is_err());
        for slot in [0, 1, 2, 3, 4, 5] {
            let mut bad = words;
            bad[slot] = match slot {
                0 => 1,
                1 => 81,
                2 => 1.0f64.to_bits(),
                3 => 2.0f64.to_bits(),
                4 => 0,
                _ => 2,
            };
            assert!(
                fitted_state_from_device_words(&names, 0..80, &bad).is_err(),
                "slot {slot}"
            );
        }
        assert!(fitted_state_from_device_words(&names, 0..80, &words[..11]).is_err());
        assert!(fitted_state_from_device_words(&names, 0..79, &words).is_err());
    }
}
