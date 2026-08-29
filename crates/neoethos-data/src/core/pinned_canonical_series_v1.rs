//! Move-only immutable canonical-series pinning without value materialization.

use crate::SymbolDataset;
use crate::core::canonical_ohlcv::materialize_pinned_canonical_timeframe_v1;
#[cfg(feature = "gpu-cuda")]
use crate::core::canonical_ohlcv::{CanonicalDatasetArtifactV1, CanonicalOhlcvFrame};
use crate::core::dataset_generation_lease::DatasetGenerationLease;
use crate::core::dataset_manifest::{
    CanonicalDatasetSeriesReceiptV1, DatasetManifestV1, open_exact_dataset_generation,
};
use anyhow::{Context, Result, ensure};
use neoethos_dataset_contracts::CanonicalTimeframe;
#[cfg(feature = "gpu-cuda")]
use neoethos_feature_contracts::SourceArtifactBindingV1;
#[cfg(feature = "gpu-cuda")]
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

struct PinnedCanonicalGenerationV1 {
    manifest: DatasetManifestV1,
    lease: Arc<DatasetGenerationLease>,
}

#[cfg(feature = "gpu-cuda")]
#[derive(Debug)]
struct PinnedResidentCanonicalSourceV1 {
    artifact: CanonicalDatasetArtifactV1,
    binding: SourceArtifactBindingV1,
}

/// One decoded full generation that still owns the exact artifact lease and
/// its independently sealed source binding. This type is crate-private and
/// move-only; it cannot be reconstructed from hashes or caller row ranges.
#[cfg(feature = "gpu-cuda")]
#[must_use = "the materialized pinned source must move into resident assembly"]
#[derive(Debug)]
pub(crate) struct MaterializedPinnedResidentCanonicalSourceV1 {
    timeframe: CanonicalTimeframe,
    frame: CanonicalOhlcvFrame,
    binding: SourceArtifactBindingV1,
    source_binding_sha256: [u8; 32],
}

#[cfg(feature = "gpu-cuda")]
impl MaterializedPinnedResidentCanonicalSourceV1 {
    pub(crate) const fn timeframe(&self) -> CanonicalTimeframe {
        self.timeframe
    }

    pub(crate) const fn frame(&self) -> &CanonicalOhlcvFrame {
        &self.frame
    }

    pub(crate) const fn binding(&self) -> &SourceArtifactBindingV1 {
        &self.binding
    }

    pub(crate) const fn source_binding_sha256(&self) -> [u8; 32] {
        self.source_binding_sha256
    }
}

/// Exact base plus selected higher direct generations, in the canonical order
/// sealed by the series receipt. The contained frames retain every reader
/// lease through resident recipe, runtime, and final store ownership.
#[cfg(feature = "gpu-cuda")]
#[must_use = "the materialized pinned source set must move into resident assembly"]
#[derive(Debug)]
pub(crate) struct MaterializedPinnedResidentCanonicalSourcesV1 {
    receipt: CanonicalDatasetSeriesReceiptV1,
    base: MaterializedPinnedResidentCanonicalSourceV1,
    direct_parents: Vec<MaterializedPinnedResidentCanonicalSourceV1>,
}

#[cfg(feature = "gpu-cuda")]
impl MaterializedPinnedResidentCanonicalSourcesV1 {
    pub(crate) const fn receipt(&self) -> &CanonicalDatasetSeriesReceiptV1 {
        &self.receipt
    }

    pub(crate) const fn base(&self) -> &MaterializedPinnedResidentCanonicalSourceV1 {
        &self.base
    }

    pub(crate) fn direct_parents(&self) -> &[MaterializedPinnedResidentCanonicalSourceV1] {
        &self.direct_parents
    }

    pub(crate) fn all_sources(
        &self,
    ) -> impl Iterator<Item = &MaterializedPinnedResidentCanonicalSourceV1> {
        std::iter::once(&self.base).chain(self.direct_parents.iter())
    }

    pub(crate) const fn source_count(&self) -> usize {
        1 + self.direct_parents.len()
    }
}

/// Move-only full-generation source authority for resident feature assembly.
/// The retained artifacts own every exact reader lease; callers can inspect
/// bindings but cannot choose segments, hashes, node ids, or lease lifetime.
#[cfg(feature = "gpu-cuda")]
#[must_use = "the pinned resident source descriptor must outlive the resident feature store"]
#[derive(Debug)]
pub(crate) struct PinnedResidentCanonicalSourceDescriptorV1 {
    receipt: CanonicalDatasetSeriesReceiptV1,
    sources: Vec<PinnedResidentCanonicalSourceV1>,
}

#[cfg(feature = "gpu-cuda")]
impl PinnedResidentCanonicalSourceDescriptorV1 {
    pub(crate) const fn receipt(&self) -> &CanonicalDatasetSeriesReceiptV1 {
        &self.receipt
    }

    pub(crate) fn generation_count(&self) -> usize {
        self.sources.len()
    }

    pub(crate) fn source_binding(
        &self,
        timeframe: CanonicalTimeframe,
    ) -> Result<&SourceArtifactBindingV1> {
        self.sources
            .iter()
            .find(|source| source.artifact.frame_timeframe() == timeframe)
            .map(|source| &source.binding)
            .with_context(|| format!("pinned resident source has no direct {timeframe} binding"))
    }

    /// Decode every exact full generation once, retaining its lease and sealed
    /// source binding. No current-generation lookup, row window, or caller
    /// segment may enter this path.
    pub(crate) fn into_materialized_resident_sources_v1(
        self,
        base_timeframe: CanonicalTimeframe,
    ) -> Result<MaterializedPinnedResidentCanonicalSourcesV1> {
        let Self { receipt, sources } = self;
        ensure!(
            receipt.direct_timeframes().len() == sources.len(),
            "pinned resident source receipt/generation count drifted before materialization"
        );
        let mut base = None;
        let mut direct_parents = Vec::with_capacity(sources.len().saturating_sub(1));
        for (selected, source) in receipt.direct_timeframes().iter().zip(sources) {
            let PinnedResidentCanonicalSourceV1 { artifact, binding } = source;
            let timeframe = artifact.frame_timeframe();
            ensure!(
                selected.identity() == artifact.identity()
                    && selected.generation_id() == artifact.generation_id()
                    && binding.dataset_identity() == artifact.identity(),
                "pinned resident generation disagrees with its receipt or source binding"
            );
            let array = artifact.lease().reopen_verified().with_context(|| {
                format!(
                    "reopen pinned resident generation {}",
                    artifact.lease().path().display()
                )
            })?;
            let ohlcv = crate::vortex_array_to_ohlcv(array)
                .with_context(|| format!("decode pinned resident {timeframe} full generation"))?;
            let frame = CanonicalOhlcvFrame::from_parts(ohlcv, artifact)?;
            let source_binding_sha256 = source_binding_sha256_v1(&binding)?;
            let materialized = MaterializedPinnedResidentCanonicalSourceV1 {
                timeframe,
                frame,
                binding,
                source_binding_sha256,
            };
            if timeframe == base_timeframe {
                ensure!(
                    base.is_none(),
                    "pinned resident source repeats its base timeframe"
                );
                base = Some(materialized);
            } else {
                ensure!(
                    timeframe > base_timeframe,
                    "pinned resident direct parent {timeframe} is not above base {base_timeframe}"
                );
                direct_parents.push(materialized);
            }
        }
        let base = base.context("pinned resident source omitted its exact base timeframe")?;
        Ok(MaterializedPinnedResidentCanonicalSourcesV1 {
            receipt,
            base,
            direct_parents,
        })
    }
}

#[cfg(feature = "gpu-cuda")]
fn source_binding_sha256_v1(binding: &SourceArtifactBindingV1) -> Result<[u8; 32]> {
    fn update_bytes(hash: &mut Sha256, bytes: &[u8], field: &'static str) -> Result<()> {
        let len = u64::try_from(bytes.len()).with_context(|| format!("{field} length overflow"))?;
        hash.update(len.to_le_bytes());
        hash.update(bytes);
        Ok(())
    }

    let mut hash = Sha256::new();
    hash.update(b"neoethos.data.full-source-artifact-binding.v1\0");
    update_bytes(
        &mut hash,
        binding.source_node_id().as_bytes(),
        "source node id",
    )?;
    update_bytes(
        &mut hash,
        &binding.dataset_identity().canonical_bytes(),
        "dataset identity",
    )?;
    update_bytes(
        &mut hash,
        binding.manifest_schema_id().as_bytes(),
        "manifest schema id",
    )?;
    hash.update(binding.manifest_hash());
    update_bytes(
        &mut hash,
        binding.generation_id().as_bytes(),
        "generation id",
    )?;
    hash.update(binding.vortex_hash());
    hash.update([binding.bar_timestamp_convention().identity_tag()]);
    hash.update(
        u64::try_from(binding.segments().len())
            .context("source binding segment count overflow")?
            .to_le_bytes(),
    );
    for segment in binding.segments() {
        hash.update(segment.row_start().to_le_bytes());
        hash.update(segment.row_end().to_le_bytes());
        hash.update(segment.timestamp_start_ms().to_le_bytes());
        hash.update(segment.timestamp_end_ms().to_le_bytes());
    }
    Ok(hash.finalize().into())
}

/// Exact generation manifests and shared reader leases for one canonical
/// source/account/symbol series. This type deliberately does not implement
/// `Clone`; the selected CPU/native materialization factory consumes it once.
pub struct PinnedCanonicalSeriesV1 {
    receipt: CanonicalDatasetSeriesReceiptV1,
    generations: Vec<PinnedCanonicalGenerationV1>,
}

impl std::fmt::Debug for PinnedCanonicalSeriesV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnedCanonicalSeriesV1")
            .field("receipt", &self.receipt)
            .field("generation_count", &self.generations.len())
            .finish_non_exhaustive()
    }
}

impl PinnedCanonicalSeriesV1 {
    pub const fn receipt(&self) -> &CanonicalDatasetSeriesReceiptV1 {
        &self.receipt
    }

    pub fn row_count(&self, timeframe: CanonicalTimeframe) -> Result<usize> {
        let generation = self
            .generations
            .iter()
            .find(|generation| generation.manifest.identity().timeframe() == timeframe)
            .with_context(|| format!("pinned series has no direct {timeframe} generation"))?;
        usize::try_from(generation.manifest.row_count())
            .context("pinned generation row count does not fit this process")
    }

    /// Consume every exact manifest+lease into a resident-only descriptor.
    /// This path derives full-generation source bindings from the pinned
    /// manifests and never decodes an OHLCV value.
    #[cfg(feature = "gpu-cuda")]
    pub(crate) fn into_resident_source_descriptor_v1(
        self,
    ) -> Result<PinnedResidentCanonicalSourceDescriptorV1> {
        let PinnedCanonicalSeriesV1 {
            receipt,
            generations,
        } = self;
        ensure!(
            generations.len() == receipt.direct_timeframes().len(),
            "pinned resident source generation count disagrees with its sealed series receipt"
        );
        let mut sources = Vec::with_capacity(generations.len());
        for (selected, generation) in receipt.direct_timeframes().iter().zip(generations) {
            ensure!(
                generation.manifest.identity() == selected.identity()
                    && generation.manifest.generation_id() == selected.generation_id()
                    && generation.manifest.manifest_binding_sha256()
                        == selected.manifest_binding_sha256(),
                "pinned resident source generation disagrees with its selected receipt"
            );
            let artifact =
                CanonicalDatasetArtifactV1::from_manifest(&generation.manifest, generation.lease)?;
            let binding = artifact
                .source_binding(resident_source_node_id_v1(artifact.identity().timeframe()))?;
            sources.push(PinnedResidentCanonicalSourceV1 { artifact, binding });
        }
        ensure!(
            sources.len() == receipt.direct_timeframes().len(),
            "pinned resident source conversion omitted a direct generation"
        );
        Ok(PinnedResidentCanonicalSourceDescriptorV1 { receipt, sources })
    }

    /// Decode only after the selected prepared CPU factory proves that the
    /// complete physical inventory contains no GPU.
    #[cfg(feature = "gpu-cuda")]
    pub fn into_cpu_dataset_after_no_physical_gpu_v1(
        self,
        _authority: &neoethos_gpu_cuda::run_device_admission_v1::SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    ) -> Result<SymbolDataset> {
        self.materialize_pinned_canonical_series_v1()
    }

    /// Toolchain-free compatibility path. CUDA-enabled binaries cannot call
    /// this method; they must supply the sealed cross-vendor absence authority
    /// above.
    #[cfg(not(feature = "gpu-cuda"))]
    pub fn into_cpu_dataset_without_native_adapter_v1(self) -> Result<SymbolDataset> {
        self.materialize_pinned_canonical_series_v1()
    }

    fn materialize_pinned_canonical_series_v1(self) -> Result<SymbolDataset> {
        let symbol = self.receipt.anchor().identity().symbol_name().to_owned();
        let mut frames = HashMap::with_capacity(self.generations.len());
        let mut source_artifacts = HashMap::with_capacity(self.generations.len());
        for generation in self.generations {
            let timeframe = generation
                .manifest
                .identity()
                .timeframe()
                .as_str()
                .to_owned();
            ensure!(
                !frames.contains_key(&timeframe),
                "pinned canonical series repeats timeframe {timeframe}"
            );
            let frame =
                materialize_pinned_canonical_timeframe_v1(generation.manifest, generation.lease)?;
            frames.insert(timeframe.clone(), frame.ohlcv().clone());
            source_artifacts.insert(timeframe, frame.artifact().clone());
        }
        Ok(SymbolDataset {
            symbol,
            frames,
            source_artifacts,
        })
    }
}

#[cfg(feature = "gpu-cuda")]
fn resident_source_node_id_v1(timeframe: CanonicalTimeframe) -> String {
    format!(
        "neoethos.data.pinned-resident-source.v1:{}",
        timeframe.as_str()
    )
}

/// Acquire every exact generation lease without decoding any OHLCV value.
pub fn pin_exact_canonical_series_v1(
    root: impl AsRef<Path>,
    receipt: CanonicalDatasetSeriesReceiptV1,
) -> Result<PinnedCanonicalSeriesV1> {
    receipt.validate()?;
    let root = root.as_ref();
    let mut generations = Vec::with_capacity(receipt.direct_timeframes().len());
    for selected in receipt.direct_timeframes() {
        let (manifest, lease) =
            open_exact_dataset_generation(root, selected).with_context(|| {
                format!(
                    "pin exact canonical generation {} {}",
                    selected.identity().timeframe(),
                    selected.generation_id()
                )
            })?;
        ensure!(
            manifest.identity() == selected.identity()
                && manifest.generation_id() == selected.generation_id()
                && manifest.manifest_binding_sha256() == selected.manifest_binding_sha256(),
            "pinned generation manifest disagrees with its selected receipt"
        );
        generations.push(PinnedCanonicalGenerationV1 {
            manifest,
            lease: Arc::new(lease),
        });
    }
    Ok(PinnedCanonicalSeriesV1 {
        receipt,
        generations,
    })
}
