//! Move-only, metadata-only Data preflight for a future complete native
//! Discovery workspace plan.
//!
//! This seam owns the exact pinned generation leases and the complete
//! crate-owned resident producer census. It does not decode OHLCV values,
//! materialize feature values, or consume the one-shot CUDA run carrier.

use neoethos_dataset_contracts::CanonicalTimeframe;
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentProducerCapabilityManifestV3,
};

use super::features::FeatureProfile;
use super::gpu_resident_feature_recipe_v4::PreparedResidentFeatureRecipeAssemblyV4;
use super::gpu_resident_feature_store_v3::{
    GpuOnlyFeatureMaterializationErrorV3, seal_current_resident_producer_capability_manifest_v3,
};
use super::gpu_resident_robust_normalization_v2::{
    SealedCanonicalRobustNormalizationSplitV2,
    seal_canonical_robust_normalization_split_from_pinned_v2,
};
use super::pinned_canonical_series_v1::{
    PinnedCanonicalSeriesV1, PinnedResidentCanonicalSourceDescriptorV1,
};

/// Exact ordered producer frontier on the current production path. This is
/// descriptive backlog evidence, not capability authority.
pub const CURRENT_PENDING_RESIDENT_PRODUCERS_V3: [ResidentFeatureProducerV3; 0] = [];

/// Typed names for Data-owned receipts that remain after the producer census
/// is complete. These variants carry no bytes, hashes, handles, or authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOnlyFeatureWorkspaceReceiptBacklogV3 {
    DatasetRecipeIdentity,
    FeaturePlanSchemaIdentity,
    RoutePlanIdentity,
    OrderedResidentFeatureRoutes,
    ExactResidentProducerBatchLedger,
    NormalizationScratchExtent,
    NormalizationFitMetadataExtent,
    FinalFeaturePlanIdentity,
    NormalizationFitIdentity,
    SourceProvenanceIdentity,
}

/// Every Data-owned recipe, extent, final-plan, normalization and provenance
/// receipt and resident producer now has a concrete sealer. Empty arrays are
/// retained as explicit release-census evidence, not as runtime authority.
pub const CURRENT_PENDING_FEATURE_WORKSPACE_RECEIPTS_V3: [GpuOnlyFeatureWorkspaceReceiptBacklogV3;
    0] = [];

/// Opaque phase-zero Data authority. The exact pin and producer manifest move
/// together until a later Data-owned workspace component sealer consumes them.
/// There is deliberately no `Clone`, serialization, default, or evidence-based
/// constructor.
#[must_use = "the exact pinned Data preflight must be consumed by native workspace assembly"]
#[derive(Debug)]
pub struct PreparedGpuOnlyFeatureWorkspacePreflightV3 {
    source_descriptor: PinnedResidentCanonicalSourceDescriptorV1,
    base_timeframe: CanonicalTimeframe,
    profile: FeatureProfile,
    row_count: usize,
    budget_rows: usize,
    producer_capabilities: ResidentProducerCapabilityManifestV3,
    robust_normalization_split: SealedCanonicalRobustNormalizationSplitV2,
}

impl PreparedGpuOnlyFeatureWorkspacePreflightV3 {
    pub const fn base_timeframe(&self) -> CanonicalTimeframe {
        self.base_timeframe
    }

    pub const fn profile(&self) -> FeatureProfile {
        self.profile
    }

    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    pub const fn budget_rows(&self) -> usize {
        self.budget_rows
    }

    pub fn producer_capability_count(&self) -> usize {
        self.producer_capabilities.capabilities().len()
    }

    pub fn pinned_series_receipt(&self) -> &crate::CanonicalDatasetSeriesReceiptV1 {
        self.source_descriptor.receipt()
    }

    /// Descriptive integrity check only. The move-only split itself remains
    /// private and can be consumed only by the future complete Data component
    /// sealer once the full producer/identity frontier is closed.
    pub fn has_intact_robust_normalization_split_v2(&self) -> bool {
        self.robust_normalization_split
            .is_intact_for_row_count(self.row_count)
    }

    /// Move every phase-zero authority into the V4 local-draft assembler.
    /// The conversion consumes the source leases and split exactly once and
    /// accepts no caller-supplied identities or byte counts.
    pub(crate) fn into_resident_feature_recipe_assembly_v4(
        self,
    ) -> Result<PreparedResidentFeatureRecipeAssemblyV4, GpuOnlyFeatureMaterializationErrorV3> {
        let Self {
            source_descriptor,
            base_timeframe,
            profile,
            row_count,
            budget_rows,
            producer_capabilities,
            robust_normalization_split,
        } = self;
        PreparedResidentFeatureRecipeAssemblyV4::from_workspace_preflight(
            source_descriptor,
            base_timeframe,
            profile,
            row_count,
            budget_rows,
            producer_capabilities,
            robust_normalization_split,
        )
        .map_err(|error| GpuOnlyFeatureMaterializationErrorV3::Other(error.into()))
    }
}

/// Consume one exact series pin and seal the complete crate-owned resident
/// producer census using manifest metadata only. Any future capability drift
/// still fails before a CUDA context, stream, allocation, or feature value is
/// touched.
pub fn preflight_gpu_only_feature_workspace_v3(
    pinned_series: PinnedCanonicalSeriesV1,
    base_timeframe: CanonicalTimeframe,
    profile: FeatureProfile,
    budget_rows: usize,
) -> Result<PreparedGpuOnlyFeatureWorkspacePreflightV3, GpuOnlyFeatureMaterializationErrorV3> {
    let row_count = pinned_series
        .row_count(base_timeframe)
        .map_err(GpuOnlyFeatureMaterializationErrorV3::Other)?;
    if row_count == 0 || budget_rows < row_count {
        return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
            anyhow::anyhow!(
                "strict resident workspace preflight requires a nonempty pinned base timeframe within its frozen row budget"
            ),
        ));
    }
    let robust_normalization_split =
        seal_canonical_robust_normalization_split_from_pinned_v2(&pinned_series, base_timeframe)
            .map_err(GpuOnlyFeatureMaterializationErrorV3::Other)?;
    let source_descriptor = pinned_series
        .into_resident_source_descriptor_v1()
        .map_err(GpuOnlyFeatureMaterializationErrorV3::Other)?;
    let producer_capabilities = seal_current_resident_producer_capability_manifest_v3()?;
    Ok(PreparedGpuOnlyFeatureWorkspacePreflightV3 {
        source_descriptor,
        base_timeframe,
        profile,
        row_count,
        budget_rows,
        producer_capabilities,
        robust_normalization_split,
    })
}
