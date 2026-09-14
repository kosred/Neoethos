use crate::core::normalization::normalization_fit_hash;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::{PrimitiveArray, StructArray};
use vortex_array::dtype::{DType, NativePType, PType};

// ─── Data-layer runtime overrides (config-consolidation S3-data) ───────────
// Config-driven replacement for `NEOETHOS_BOT_NORMALIZE_FEATURES`. The binary
// installs it from `Settings.models.data_runtime` once at startup (as a plain
// bool, so this foundation crate keeps NOT depending on neoethos-core).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataRuntimeOverrides {
    pub normalize_features: bool,
}

impl Default for DataRuntimeOverrides {
    fn default() -> Self {
        Self {
            normalize_features: false,
        }
    }
}

static DATA_RUNTIME_OVERRIDES: std::sync::OnceLock<DataRuntimeOverrides> =
    std::sync::OnceLock::new();

/// Install process-wide data-layer runtime overrides from config. The
/// binaries call this once at startup with `settings.models.data_runtime.*`.
/// Idempotent — the first install wins.
pub fn install_data_runtime_overrides(normalize_features: bool) {
    let _ = DATA_RUNTIME_OVERRIDES.set(DataRuntimeOverrides { normalize_features });
}

/// Current data-layer runtime override, or the deterministic OFF default when
/// no install has happened.
pub fn current_data_runtime_overrides() -> DataRuntimeOverrides {
    DATA_RUNTIME_OVERRIDES.get().copied().unwrap_or_default()
}

/// Opaque proof that the process-wide normalization mode came from the
/// startup-installed Data configuration. Resident GPU planning deliberately
/// refuses the legacy implicit default: a native workspace must be bound to
/// the configuration that was actually admitted for this process.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip-smc"))]
#[derive(Debug)]
pub(crate) struct SealedDataRuntimeNormalizationModeV2 {
    enabled: bool,
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip-smc"))]
impl SealedDataRuntimeNormalizationModeV2 {
    pub(crate) const fn enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip-smc"))]
pub(crate) fn sealed_data_runtime_normalization_mode_v2()
-> Result<SealedDataRuntimeNormalizationModeV2> {
    let overrides = DATA_RUNTIME_OVERRIDES.get().ok_or_else(|| {
        anyhow::anyhow!(
            "resident normalization requires the startup-installed Data runtime configuration"
        )
    })?;
    Ok(SealedDataRuntimeNormalizationModeV2 {
        enabled: overrides.normalize_features,
    })
}

// ─── Feature-cube assembly policy (2026-08-10, env→config wave 2) ──────────
// The config recipient for the retired `NEOETHOS_FEATURE_CUBE_MODE`.
//
// It lives in its OWN slot rather than being added to `DataRuntimeOverrides`
// because that struct's installer, `install_data_runtime_overrides`, is called
// from `neoethos-app` and `neoethos-cli` — crates this change does not open.
// Widening its arity would break their build with no one able to repair it in
// the same change. This slot is installed from
// `neoethos_search::install_search_runtime_overrides_from_settings`, which
// every production binary already calls with the resolved `Settings`, so the
// field reaches production without a second edit anywhere.

static FEATURE_CUBE_POLICY: std::sync::OnceLock<neoethos_core::config::FeatureCubeMode> =
    std::sync::OnceLock::new();

/// Install the operator's `models.data_runtime.feature_cube_mode`.
/// Idempotent — the first install wins.
pub fn install_feature_cube_policy(mode: neoethos_core::config::FeatureCubeMode) {
    let _ = FEATURE_CUBE_POLICY.set(mode);
}

/// The installed feature-cube policy, or `Auto` when nothing installed one.
/// `Auto` is exactly what every run got while the env var was unset, so an
/// un-installed process behaves as it always did.
pub fn current_feature_cube_policy() -> neoethos_core::config::FeatureCubeMode {
    FEATURE_CUBE_POLICY
        .get()
        .copied()
        .unwrap_or(neoethos_core::config::FeatureCubeMode::Auto)
}

#[cfg(test)]
mod data_runtime_overrides_tests {
    use super::*;

    #[test]
    fn data_runtime_overrides_default_is_off() {
        // Behavior-preservation: env-unset normalization defaulted OFF.
        let d = DataRuntimeOverrides::default();
        assert!(!d.normalize_features);
    }
}

pub mod core;
mod feature_build_control;
pub use feature_build_control::{FeatureBuildCancelled, FeatureBuildControl, FeatureBuildProgress};
pub mod test_fixtures;
pub use crate::core::{initialize_source_seal_before_runtime, source_seal_slot_limit};
// Re-export the canonical timeframe list so callers using neoethos-data
// can grab it without pulling in neoethos-core directly.
pub use crate::core::canonical_ohlcv::{
    CanonicalDatasetArtifactV1, CanonicalOhlcvFrame, load_canonical_timeframe,
    load_exact_canonical_timeframe,
};
pub use crate::core::canonical_ohlcv_stream::{
    CanonicalOhlcvChunk, CanonicalOhlcvReverseSpool, CanonicalOhlcvReverseSpoolIter,
    CanonicalOhlcvStreamPublishRequest, CanonicalVolumeChunk, publish_canonical_ohlcv_stream,
};
pub use crate::core::dataset_manifest::{
    CanonicalDatasetSeriesReceiptV1, ExactDatasetGenerationConflict, SelectedDatasetGenerationV1,
    open_exact_dataset_generation,
};
pub use crate::core::direct_timeframes::*;
pub use crate::core::discover::{
    DataFileEntry, DataFormat, DataVerificationStatus, DatasetDiscovery, MAX_FILE_SIZE_BYTES,
    MAX_WALK_DEPTH, SkipReason, SkippedFile,
};
pub use crate::core::feature_registry::*;
pub use crate::core::features::*;
pub use crate::core::footprint_features::*;
#[cfg(feature = "gpu-cuda")]
pub use crate::core::gpu_only_feature_workspace_preflight_v3::{
    CURRENT_PENDING_FEATURE_WORKSPACE_RECEIPTS_V3, CURRENT_PENDING_RESIDENT_PRODUCERS_V3,
    GpuOnlyFeatureWorkspaceReceiptBacklogV3, PreparedGpuOnlyFeatureWorkspacePreflightV3,
    preflight_gpu_only_feature_workspace_v3,
};
#[cfg(feature = "gpu-cuda")]
pub use crate::core::gpu_resident_feature_store_v3::{
    AdmittedCompactSelectedStoreV2, CanonicalGpuResidentFeatureExecutionSemanticV1,
    GpuOnlyFeatureMaterializationAdmissionV3, GpuOnlyFeatureMaterializationErrorV3,
    PreparedCompactSelectedStoreV2, PreparedGpuOnlyFeatureMaterializationV3,
    PreparedGpuOnlyFeatureTwoPassContinuationV2, ScoredGpuOnlyFeatureTwoPassContinuationV2,
    SealedGpuResidentFeatureStoreV3, SealedResidentSelectedMapReceiptV2,
    ValidatedGpuResidentFeatureExecutionAuthorityV1, begin_prepared_gpu_only_feature_two_pass_v2,
    materialize_compact_selected_store_v2, materialize_gpu_only_feature_store_v3,
    materialize_prepared_gpu_only_feature_store_for_data_population_v3,
    materialize_prepared_gpu_only_feature_store_v3, prepare_compact_selected_store_v2,
    prepare_gpu_only_feature_materialization_v3, seal_selected_map_v2, stream_score_batches_v2,
};
pub use crate::core::hpc_ta::*;
pub use crate::core::import_discover::{
    ImportDiscovery, ImportSkipReason, ImportSourceEntry, MAX_IMPORT_SOURCE_BYTES,
    MAX_IMPORT_WALK_DEPTH, SkippedImportSource,
};
pub use crate::core::indicators::*;
pub use crate::core::normalization::{
    NORMALIZATION_TRANSFORM_SEMANTIC_VERSION, RobustNormalizationFitF64,
    SEARCH_NORMALIZATION_POLICY_VERSION, SearchNormalizationFittedStateV1,
};
pub use crate::core::pinned_canonical_series_v1::{
    PinnedCanonicalSeriesV1, pin_exact_canonical_series_v1,
};
pub use crate::core::pinned_source_projection_v1::{
    CANONICAL_PINNED_SOURCE_PROJECTION_SCHEMA_VERSION_V1, CanonicalPinnedSourceBindingFactsV1,
    CanonicalPinnedSourceProjectionErrorV1, CanonicalPinnedSourceProjectionV1,
    CanonicalPinnedSourceSegmentFactsV1,
};
pub use crate::core::quant_exact_math_v3::{
    QUANT_LOG_OPERATION_SCHEDULE_V3, QUANT_OPENLIBM_COMMIT_V3, QUANT_OPENLIBM_E_LOG_RECEIPT_V3,
    QUANT_OPENLIBM_E_LOG_SOURCE_SHA256_V3, QUANT_OPENLIBM_E_LOG_SOURCE_V3,
    quant_log_positive_f64_v3,
};
pub use crate::core::quant_features::*;
pub use crate::core::regime_detection::*;
pub use crate::core::session_features::*;
pub use crate::core::slicing::{slice_ohlcv, slice_ohlcv_by_date_range_ms};
pub use crate::core::smc::*;
pub use crate::core::timestamps::*;
pub use crate::core::vortex_io::*;
pub use neoethos_core::is_canonical_timeframe;
pub use neoethos_dataset_contracts::{
    BarTimestampConvention, CANONICAL_TIMEFRAMES, CTraderEnvironment, CanonicalDatasetIdentity,
    CanonicalDatasetScope, CanonicalTimeframe,
};

#[derive(Debug, Clone)]
pub struct Ohlcv {
    pub timestamp: Option<Vec<i64>>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Copy)]
pub enum CanonicalVolumeRef<'a> {
    Absent,
    Float64(&'a [f64]),
    UInt64(&'a [u64]),
    Int64(&'a [i64]),
}

pub struct CanonicalOhlcvPublishRequest<'a> {
    pub configured_root: &'a Path,
    pub identity: &'a CanonicalDatasetIdentity,
    pub expected_generation: Option<&'a str>,
    pub provenance: &'a crate::core::dataset_manifest::ProducerProvenanceEnvelopeV1,
    pub ohlcv: &'a Ohlcv,
    pub volume: CanonicalVolumeRef<'a>,
    pub rows_per_chunk: usize,
}

/// Publish validated OHLCV values into the one canonical immutable-generation
/// protocol while retaining the physical broker/source volume type.
pub fn publish_canonical_ohlcv_generation(
    request: CanonicalOhlcvPublishRequest<'_>,
) -> Result<crate::core::dataset_manifest::PublishResult> {
    if request.rows_per_chunk == 0 {
        bail!("Vortex rows_per_chunk must be greater than zero");
    }
    let normalized = normalize_ohlcv(request.ohlcv)?;
    if normalized.is_empty() {
        bail!("cannot publish an empty canonical OHLCV generation");
    }
    let timestamps = normalized
        .timestamp
        .as_deref()
        .context("canonical OHLCV has no timestamp_ms")?;
    let timestamp_range = crate::core::dataset_manifest::DatasetTimestampRange::new(
        timestamps[0],
        timestamps[timestamps.len() - 1],
    )?;
    let row_count = normalized.len();
    let rows_per_chunk = request.rows_per_chunk;
    let volume = request.volume;

    crate::core::dataset_manifest::publish_vortex_generation_streaming(
        crate::core::dataset_manifest::PublishMetadataRequest {
            configured_root: request.configured_root,
            identity: request.identity,
            expected_generation: request.expected_generation,
            provenance: request.provenance,
        },
        move |candidate_path| {
            let chunks = (0..row_count).step_by(rows_per_chunk).map(|start| {
                let end = (start + rows_per_chunk).min(row_count);
                let chunk_volume_values = match volume {
                    CanonicalVolumeRef::Float64(values) => Some(values[start..end].to_vec()),
                    CanonicalVolumeRef::Absent
                    | CanonicalVolumeRef::UInt64(_)
                    | CanonicalVolumeRef::Int64(_) => None,
                };
                let chunk = Ohlcv {
                    timestamp: Some(timestamps[start..end].to_vec()),
                    open: normalized.open[start..end].to_vec(),
                    high: normalized.high[start..end].to_vec(),
                    low: normalized.low[start..end].to_vec(),
                    close: normalized.close[start..end].to_vec(),
                    volume: chunk_volume_values,
                };
                let chunk_volume = match volume {
                    CanonicalVolumeRef::Absent => CanonicalVolumeRef::Absent,
                    CanonicalVolumeRef::Float64(_) => CanonicalVolumeRef::Float64(
                        chunk
                            .volume
                            .as_deref()
                            .expect("Float64 volume chunk was constructed"),
                    ),
                    CanonicalVolumeRef::UInt64(values) => {
                        CanonicalVolumeRef::UInt64(&values[start..end])
                    }
                    CanonicalVolumeRef::Int64(values) => {
                        CanonicalVolumeRef::Int64(&values[start..end])
                    }
                };
                ohlcv_to_vortex_array_with_canonical_volume(&chunk, chunk_volume)
            });
            let write_stats =
                crate::core::vortex_io::write_vortex_chunks_fallible(candidate_path, chunks)?;
            Ok(crate::core::dataset_manifest::CandidateWriteOutcome {
                write_stats,
                timestamp_range,
            })
        },
    )
}

impl Ohlcv {
    pub fn len(&self) -> usize {
        self.close.len()
    }
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct SymbolDataset {
    pub symbol: String,
    pub frames: HashMap<String, Ohlcv>,
    /// Exact immutable source artifact for each materialized timeframe.
    /// Derived/live frames need their own typed origin and may not fabricate an
    /// entry here; feature construction fails closed when one is absent.
    pub source_artifacts: HashMap<String, CanonicalDatasetArtifactV1>,
}

impl SymbolDataset {
    pub fn timeframe(&self, tf: &str) -> Option<&Ohlcv> {
        self.frames.get(tf)
    }

    pub fn canonical_frame(&self, tf: &str) -> Result<CanonicalOhlcvFrame> {
        let (ohlcv, artifact) = self.canonical_parts(tf)?;
        CanonicalOhlcvFrame::from_parts(ohlcv.clone(), artifact.clone())
    }

    /// Transfer a complete timeframe and its generation lease to the final
    /// consumer after borrowed feature computation is finished. Other input
    /// timeframes are released; derived features retain their own reader leases.
    pub fn into_canonical_frame(mut self, tf: &str) -> Result<CanonicalOhlcvFrame> {
        let ohlcv = self
            .frames
            .remove(tf)
            .with_context(|| format!("dataset {} has no timeframe {tf}", self.symbol))?;
        let artifact = self.source_artifacts.remove(tf).with_context(|| {
            format!(
                "dataset {} timeframe {tf} has no verified immutable source artifact",
                self.symbol
            )
        })?;
        CanonicalOhlcvFrame::from_parts(ohlcv, artifact)
    }

    fn canonical_frame_before_timestamp_ms(
        &self,
        tf: &str,
        end_exclusive_ms: i64,
    ) -> Result<CanonicalOhlcvFrame> {
        let (ohlcv, artifact) = self.canonical_parts(tf)?;
        CanonicalOhlcvFrame::copy_full_prefix_before_timestamp_ms(ohlcv, artifact, end_exclusive_ms)
    }

    fn canonical_parts(&self, tf: &str) -> Result<(&Ohlcv, &CanonicalDatasetArtifactV1)> {
        let ohlcv = self
            .frames
            .get(tf)
            .with_context(|| format!("dataset {} has no timeframe {tf}", self.symbol))?;
        let artifact = self.source_artifacts.get(tf).with_context(|| {
            format!(
                "dataset {} timeframe {tf} has no verified immutable source artifact",
                self.symbol
            )
        })?;
        Ok((ohlcv, artifact))
    }

    pub fn timeframes(&self) -> Vec<String> {
        let mut out: Vec<String> = self.frames.keys().cloned().collect();
        out.sort();
        out
    }
}

pub fn discover_symbols(root: impl AsRef<Path>) -> Result<Vec<String>> {
    let mut symbols = discover_all_canonical_dataset_identities(root)?
        .into_iter()
        .map(|identity| identity.symbol_name().to_owned())
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();
    Ok(symbols)
}

pub fn discover_timeframes(root: impl AsRef<Path>, symbol: &str) -> Result<Vec<String>> {
    let identities = discover_canonical_dataset_identities(root, symbol)?;
    let mut out = identities
        .into_iter()
        .map(|identity| identity.timeframe().as_str().to_owned())
        .collect::<Vec<_>>();
    out.dedup();
    Ok(out)
}

pub fn load_symbol_timeframe(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
) -> Result<Ohlcv> {
    let canonical_timeframe = timeframe
        .parse::<CanonicalTimeframe>()
        .with_context(|| format!("unsupported canonical timeframe {timeframe}"))?;
    let identities = discover_canonical_dataset_identities(&root, symbol)?;
    let matching = identities
        .iter()
        .filter(|identity| identity.timeframe() == canonical_timeframe)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        matching.len() == 1,
        "expected exactly one verified canonical Vortex generation for {symbol} {timeframe}, found {}; raw source files require explicit import and retired symbol=/timeframe= layouts require explicit offline migration",
        matching.len()
    );
    let (ohlcv, _artifact) = load_canonical_timeframe(root, matching[0])?.into_full_parts()?;
    Ok(ohlcv)
}

/// Load only the trailing `tail_n` rows for a symbol/timeframe.
///
/// #155: the full `load_symbol_timeframe` path materialises every row in
/// the Vortex file into an `Ohlcv` even when the caller only wants the
/// last 200 candles for a chart. For a multi-year M1 dataset that's
/// ~1 M rows × (5 × 8 bytes) ≈ 40 MB allocated per request, plus a
/// timestamp normalisation pass that walks every value.
///
/// This helper loads the same file but trims the in-memory `Ohlcv` down
/// to its last `tail_n` rows BEFORE returning it. The on-disk read still
/// materialises the whole stream — Vortex doesn't expose a cheap "skip
/// to row N" primitive at the layout level we use today — but the
/// caller-visible allocation and downstream iteration drops to
/// O(tail_n). When Vortex grows a true row-range scan API, this is the
/// function to upgrade; the surrounding contract stays the same.
pub fn load_symbol_timeframe_tail(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
    tail_n: usize,
) -> Result<Ohlcv> {
    let mut ohlcv = load_symbol_timeframe(root, symbol, timeframe)?;
    let total = ohlcv.len();
    if tail_n >= total {
        return Ok(ohlcv);
    }
    let drop = total - tail_n;
    ohlcv.open.drain(..drop);
    ohlcv.high.drain(..drop);
    ohlcv.low.drain(..drop);
    ohlcv.close.drain(..drop);
    if let Some(ts) = ohlcv.timestamp.as_mut() {
        ts.drain(..drop);
    }
    if let Some(v) = ohlcv.volume.as_mut() {
        v.drain(..drop);
    }
    Ok(ohlcv)
}

pub fn load_symbol_dataset(root: impl AsRef<Path>, symbol: &str) -> Result<SymbolDataset> {
    let identities = discover_canonical_dataset_identities(&root, symbol)?;
    let mut frames = HashMap::new();
    let mut source_artifacts = HashMap::new();
    for identity in identities {
        let tf = identity.timeframe().as_str().to_owned();
        anyhow::ensure!(
            !frames.contains_key(&tf),
            "multiple canonical dataset identities match {symbol} {tf}; select an exact source/account identity"
        );
        let loaded = load_canonical_timeframe(&root, &identity)
            .with_context(|| format!("failed to load canonical dataset timeframe {symbol} {tf}"))?;
        let (ohlcv, artifact) = loaded.into_full_parts()?;
        frames.insert(tf.clone(), ohlcv);
        source_artifacts.insert(tf, artifact);
    }
    anyhow::ensure!(
        !frames.is_empty(),
        "no canonical versioned Vortex datasets found for symbol {symbol}; legacy symbol=/timeframe= layouts require explicit offline migration"
    );
    Ok(SymbolDataset {
        symbol: symbol.to_string(),
        frames,
        source_artifacts,
    })
}

/// Open one immutable canonical series from the exact generation receipts
/// selected by the caller.
///
/// Every timeframe is reopened through its generation id and manifest-binding
/// hash. This function never inventories a root, follows a current pointer, or
/// derives one timeframe from another.
pub fn load_exact_dataset_series_receipt(
    root: impl AsRef<Path>,
    series: &CanonicalDatasetSeriesReceiptV1,
) -> Result<SymbolDataset> {
    series.validate()?;
    let root = root.as_ref();
    let symbol = series.anchor().identity().symbol_name();
    let mut frames = HashMap::with_capacity(series.direct_timeframes().len());
    let mut source_artifacts = HashMap::with_capacity(series.direct_timeframes().len());

    for selected in series.direct_timeframes() {
        anyhow::ensure!(
            selected.identity().symbol_name() == symbol,
            "selected canonical series contains foreign symbol {} under anchor {symbol}",
            selected.identity().symbol_name()
        );
        let timeframe = selected.identity().timeframe().as_str().to_owned();
        anyhow::ensure!(
            !frames.contains_key(&timeframe),
            "selected canonical series repeats timeframe {timeframe}"
        );
        let loaded = load_exact_canonical_timeframe(root, selected).with_context(|| {
            format!(
                "failed to reopen exact selected generation {} for {symbol} {timeframe}",
                selected.generation_id()
            )
        })?;
        anyhow::ensure!(
            loaded.artifact().identity() == selected.identity(),
            "exact selected generation reopened with a different dataset identity"
        );
        let (ohlcv, artifact) = loaded.into_full_parts()?;
        frames.insert(timeframe.clone(), ohlcv);
        source_artifacts.insert(timeframe, artifact);
    }

    anyhow::ensure!(
        frames.contains_key(series.anchor().identity().timeframe().as_str()),
        "selected canonical series lost its anchor timeframe"
    );
    Ok(SymbolDataset {
        symbol: symbol.to_owned(),
        frames,
        source_artifacts,
    })
}

/// Load the exact source/account series selected by one canonical identity.
///
/// The selected identity acts as an anchor: every returned timeframe must
/// have the same scope (external namespace or exact cTrader
/// environment/server/account/symbol id), exact symbol name, and bar timestamp
/// convention. This is the production-safe alternative to symbol-only loading
/// when multiple legitimate datasets exist for the same pair.
pub fn load_dataset_for_identity(
    root: impl AsRef<Path>,
    selected: &CanonicalDatasetIdentity,
) -> Result<SymbolDataset> {
    let identities = discover_all_canonical_dataset_identities(&root)?;
    anyhow::ensure!(
        identities.iter().any(|identity| identity == selected),
        "selected canonical dataset identity has no current versioned Vortex generation"
    );
    let matching = identities
        .into_iter()
        .filter(|identity| same_dataset_series(identity, selected))
        .collect::<Vec<_>>();
    load_exact_dataset_identities(root, selected.symbol_name(), matching)
}

pub fn load_dataset_for_identity_with_timeframes(
    root: impl AsRef<Path>,
    selected: &CanonicalDatasetIdentity,
    target_tfs: &[&str],
) -> Result<SymbolDataset> {
    let requested = target_tfs
        .iter()
        .map(|timeframe| {
            timeframe
                .parse::<CanonicalTimeframe>()
                .with_context(|| format!("unsupported canonical timeframe {timeframe}"))
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    let identities = discover_all_canonical_dataset_identities(&root)?;
    anyhow::ensure!(
        identities.iter().any(|identity| identity == selected),
        "selected canonical dataset identity has no current versioned Vortex generation"
    );
    let matching = identities
        .into_iter()
        .filter(|identity| {
            same_dataset_series(identity, selected) && requested.contains(&identity.timeframe())
        })
        .collect::<Vec<_>>();
    anyhow::ensure!(
        matching.len() == requested.len(),
        "selected canonical dataset series provides {} of {} requested timeframes",
        matching.len(),
        requested.len()
    );
    load_exact_dataset_identities(root, selected.symbol_name(), matching)
}

fn same_dataset_series(
    candidate: &CanonicalDatasetIdentity,
    selected: &CanonicalDatasetIdentity,
) -> bool {
    candidate.scope() == selected.scope()
        && candidate.symbol_name() == selected.symbol_name()
        && candidate.bar_timestamp_convention() == selected.bar_timestamp_convention()
}

fn load_exact_dataset_identities(
    root: impl AsRef<Path>,
    symbol: &str,
    identities: Vec<CanonicalDatasetIdentity>,
) -> Result<SymbolDataset> {
    anyhow::ensure!(
        !identities.is_empty(),
        "selected canonical dataset series has no loadable timeframes"
    );
    let mut frames = HashMap::new();
    let mut source_artifacts = HashMap::new();
    for identity in identities {
        let timeframe = identity.timeframe().as_str().to_owned();
        anyhow::ensure!(
            !frames.contains_key(&timeframe),
            "selected canonical dataset series contains duplicate timeframe {timeframe}"
        );
        let loaded = load_canonical_timeframe(&root, &identity).with_context(|| {
            format!(
                "failed to load selected canonical dataset {}",
                identity.to_path_component()
            )
        })?;
        let (ohlcv, artifact) = loaded.into_full_parts()?;
        frames.insert(timeframe.clone(), ohlcv);
        source_artifacts.insert(timeframe, artifact);
    }
    Ok(SymbolDataset {
        symbol: symbol.to_owned(),
        frames,
        source_artifacts,
    })
}

pub fn load_symbol_dataset_with_timeframes(
    root: impl AsRef<Path>,
    symbol: &str,
    target_tfs: &[&str],
) -> Result<SymbolDataset> {
    let identities = discover_canonical_dataset_identities(&root, symbol)?;
    let mut frames = HashMap::new();
    let mut source_artifacts = HashMap::new();
    for tf in target_tfs {
        let canonical_tf = tf
            .parse::<CanonicalTimeframe>()
            .with_context(|| format!("unsupported canonical timeframe {tf}"))?;
        let matching = identities
            .iter()
            .filter(|identity| identity.timeframe() == canonical_tf)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            matching.len() == 1,
            "expected exactly one canonical dataset identity for {symbol} {tf}, found {}",
            matching.len()
        );
        let loaded = load_canonical_timeframe(&root, matching[0]).with_context(|| {
            format!("failed to load requested canonical dataset timeframe {symbol} {tf}")
        })?;
        let (ohlcv, artifact) = loaded.into_full_parts()?;
        frames.insert(tf.to_string(), ohlcv);
        source_artifacts.insert(tf.to_string(), artifact);
    }
    Ok(SymbolDataset {
        symbol: symbol.to_string(),
        frames,
        source_artifacts,
    })
}

/// Discover versioned canonical identities without interpreting the retired
/// human `symbol=/timeframe=` layout. Ambiguous identities are retained here
/// and rejected by callers that failed to request an exact account/source.
pub fn discover_canonical_dataset_identities(
    root: impl AsRef<Path>,
    symbol: &str,
) -> Result<Vec<CanonicalDatasetIdentity>> {
    let mut identities = discover_all_canonical_dataset_identities(root)?
        .into_iter()
        .filter(|identity| identity.symbol_name().eq_ignore_ascii_case(symbol))
        .collect::<Vec<_>>();
    identities.sort_by(|left, right| {
        left.timeframe()
            .ctrader_protocol_code()
            .cmp(&right.timeframe().ctrader_protocol_code())
            .then_with(|| left.to_path_component().cmp(&right.to_path_component()))
    });
    Ok(identities)
}

fn discover_all_canonical_dataset_identities(
    root: impl AsRef<Path>,
) -> Result<Vec<CanonicalDatasetIdentity>> {
    let mut identities = Vec::new();
    let root = root.as_ref();
    // Identity selection is metadata-only. The exact selected generation is
    // fully hashed/decoded by `load_canonical_timeframe`; hashing every other
    // multi-gigabyte dataset here made one-symbol loads scale with the entire
    // store and made UI inventory continuously saturate disk.
    let discovery = crate::core::discover::DatasetDiscovery::scan_metadata(root)?;
    for entry in discovery.entries {
        let dataset_root = entry
            .path
            .parent()
            .context("canonical generation has no dataset root")?;
        let component = dataset_root
            .file_name()
            .and_then(|name| name.to_str())
            .context("canonical dataset identity path is not UTF-8")?;
        identities.push(
            CanonicalDatasetIdentity::from_path_component(component)
                .with_context(|| format!("invalid verified canonical dataset {component:?}"))?,
        );
    }
    identities.sort_by(|left, right| {
        left.symbol_name()
            .cmp(right.symbol_name())
            .then_with(|| {
                left.timeframe()
                    .ctrader_protocol_code()
                    .cmp(&right.timeframe().ctrader_protocol_code())
            })
            .then_with(|| left.to_path_component().cmp(&right.to_path_component()))
    });
    Ok(identities)
}

pub fn write_ohlcv_vortex(path: impl AsRef<Path>, ohlcv: &Ohlcv) -> Result<()> {
    let normalized = normalize_ohlcv(ohlcv)?;
    let array = ohlcv_to_vortex_array(&normalized)?;
    write_vortex_array(path, array)
}

pub fn write_ohlcv_vortex_with_volume(
    path: impl AsRef<Path>,
    ohlcv: &Ohlcv,
    volume: CanonicalVolumeRef<'_>,
) -> Result<()> {
    let array = ohlcv_to_vortex_array_with_canonical_volume(ohlcv, volume)?;
    write_vortex_array(path, array)
}

/// Convert an OHLCV value into bounded same-schema Vortex chunks.
///
/// This is the bridge used by the verified publisher and, later, the shared
/// streaming importer. It intentionally never creates a full-file encoded
/// buffer. `normalize_ohlcv` is deliberately validation-only: canonical data
/// is never unit-inferred, reordered, deduplicated, or otherwise repaired.
pub fn ohlcv_to_vortex_chunks(
    ohlcv: &Ohlcv,
    rows_per_chunk: usize,
) -> Result<Vec<vortex_array::ArrayRef>> {
    if rows_per_chunk == 0 {
        bail!("Vortex rows_per_chunk must be greater than zero");
    }
    let normalized = normalize_ohlcv(ohlcv)?;
    if normalized.is_empty() {
        bail!("cannot encode an empty OHLCV dataset");
    }
    let timestamps = normalized
        .timestamp
        .as_ref()
        .context("OHLCV dataset has no timestamps")?;
    let mut chunks = Vec::with_capacity(normalized.len().div_ceil(rows_per_chunk));
    for start in (0..normalized.len()).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(normalized.len());
        let chunk = Ohlcv {
            timestamp: Some(timestamps[start..end].to_vec()),
            open: normalized.open[start..end].to_vec(),
            high: normalized.high[start..end].to_vec(),
            low: normalized.low[start..end].to_vec(),
            close: normalized.close[start..end].to_vec(),
            volume: normalized
                .volume
                .as_ref()
                .map(|values| values[start..end].to_vec()),
        };
        chunks.push(ohlcv_to_vortex_array(&chunk)?);
    }
    Ok(chunks)
}

pub fn load_vortex(path: impl AsRef<Path>) -> Result<Ohlcv> {
    let path = path.as_ref();
    let array = read_vortex_array(path)?;
    vortex_array_to_ohlcv(array)
}

pub fn normalize_ohlcv(ohlcv: &Ohlcv) -> Result<Ohlcv> {
    validate_canonical_ohlcv(ohlcv)?;
    Ok(ohlcv.clone())
}

fn validate_canonical_ohlcv(ohlcv: &Ohlcv) -> Result<()> {
    let timestamps = ohlcv
        .timestamp
        .as_ref()
        .context("OHLCV dataset has no timestamps")?;
    crate::core::timestamps::validate_canonical_millisecond_timestamps(timestamps)
        .context("validate canonical OHLCV timestamps")?;
    let volume = ohlcv.volume.as_ref();
    let expected_len = timestamps.len();

    if ohlcv.open.len() != expected_len
        || ohlcv.high.len() != expected_len
        || ohlcv.low.len() != expected_len
        || ohlcv.close.len() != expected_len
        || volume.is_some_and(|values| values.len() != expected_len)
    {
        bail!(
            "OHLCV column length mismatch: timestamps={} open={} high={} low={} close={} — \
             the file may be corrupted; re-import it.",
            expected_len,
            ohlcv.open.len(),
            ohlcv.high.len(),
            ohlcv.low.len(),
            ohlcv.close.len()
        );
    }

    for (idx, &timestamp) in timestamps.iter().enumerate() {
        let volume_value = volume.and_then(|values| values.get(idx).copied());
        let row = OhlcvRow {
            timestamp,
            open: ohlcv.open[idx],
            high: ohlcv.high[idx],
            low: ohlcv.low[idx],
            close: ohlcv.close[idx],
            volume: volume_value,
        };
        validate_ohlcv_row(&row).with_context(|| format!("validate canonical OHLCV row {idx}"))?;
    }
    Ok(())
}

fn ohlcv_to_vortex_array(ohlcv: &Ohlcv) -> Result<vortex_array::ArrayRef> {
    let volume = ohlcv
        .volume
        .as_deref()
        .map_or(CanonicalVolumeRef::Absent, CanonicalVolumeRef::Float64);
    ohlcv_to_vortex_array_with_canonical_volume(ohlcv, volume)
}

pub(crate) fn ohlcv_to_vortex_array_with_canonical_volume(
    ohlcv: &Ohlcv,
    volume: CanonicalVolumeRef<'_>,
) -> Result<vortex_array::ArrayRef> {
    match volume {
        CanonicalVolumeRef::Absent => {
            if ohlcv.volume.is_some() {
                bail!("canonical volume contract says absent but OHLCV carries Float64 volume");
            }
        }
        CanonicalVolumeRef::Float64(values) => {
            if ohlcv.volume.as_deref() != Some(values) {
                bail!("canonical Float64 volume must be the validated OHLCV volume column");
            }
        }
        CanonicalVolumeRef::UInt64(values) => {
            if ohlcv.volume.is_some() {
                bail!("canonical UInt64 volume cannot coexist with an OHLCV Float64 column");
            }
            if values.len() != ohlcv.len() {
                bail!(
                    "canonical UInt64 volume length {} disagrees with {} market rows",
                    values.len(),
                    ohlcv.len()
                );
            }
        }
        CanonicalVolumeRef::Int64(values) => {
            if ohlcv.volume.is_some() {
                bail!("canonical Int64 volume cannot coexist with an OHLCV Float64 column");
            }
            if values.len() != ohlcv.len() {
                bail!(
                    "canonical Int64 volume length {} disagrees with {} market rows",
                    values.len(),
                    ohlcv.len()
                );
            }
            if let Some(value) = values.iter().find(|value| **value < 0) {
                bail!("raw Int64 volume {value} is negative");
            }
        }
    }
    validate_canonical_ohlcv(ohlcv)?;
    let timestamps = ohlcv
        .timestamp
        .as_ref()
        .context("OHLCV dataset has no timestamps")?;
    let mut fields = vec![
        (
            "timestamp",
            PrimitiveArray::from_iter(timestamps.iter().copied()).into_array(),
        ),
        (
            "open",
            PrimitiveArray::from_iter(ohlcv.open.iter().copied()).into_array(),
        ),
        (
            "high",
            PrimitiveArray::from_iter(ohlcv.high.iter().copied()).into_array(),
        ),
        (
            "low",
            PrimitiveArray::from_iter(ohlcv.low.iter().copied()).into_array(),
        ),
        (
            "close",
            PrimitiveArray::from_iter(ohlcv.close.iter().copied()).into_array(),
        ),
    ];

    match volume {
        CanonicalVolumeRef::Absent => {}
        CanonicalVolumeRef::Float64(values) => fields.push((
            "volume",
            PrimitiveArray::from_iter(values.iter().copied()).into_array(),
        )),
        CanonicalVolumeRef::UInt64(values) => fields.push((
            "volume",
            PrimitiveArray::from_iter(values.iter().copied()).into_array(),
        )),
        CanonicalVolumeRef::Int64(values) => fields.push((
            "volume",
            PrimitiveArray::from_iter(values.iter().copied()).into_array(),
        )),
    }

    Ok(StructArray::from_fields(&fields)
        .context("failed to build OHLCV vortex struct array")?
        .into_array())
}

pub(crate) fn vortex_array_to_ohlcv(array: vortex_array::ArrayRef) -> Result<Ohlcv> {
    let struct_array = array.to_struct();

    let timestamp = extract_non_null_primitive_vec::<i64>(
        struct_array
            .unmasked_field_by_name("timestamp")
            .context("timestamp field missing")?,
        "timestamp",
    )?;

    let get_col = |names: &[&str]| -> Result<Vec<f64>> {
        for name in names {
            if let Some(field) = struct_array.unmasked_field_by_name_opt(name) {
                return extract_non_null_primitive_vec::<f64>(field, name);
            }
        }
        bail!(
            "Missing OHLCV column(s) {:?} — re-import the source; the file may use an older schema.",
            names
        )
    };

    let open = get_col(&["open", "o"])?;
    let high = get_col(&["high", "h"])?;
    let low = get_col(&["low", "l"])?;
    let close = get_col(&["close", "c"])?;
    let volume = ["volume", "vol", "v"]
        .into_iter()
        .find_map(|name| {
            struct_array
                .unmasked_field_by_name_opt(name)
                .map(|field| extract_canonical_volume_as_f64(field, name))
        })
        .transpose()?;

    // Read-path structural check: a corrupt/truncated file can decode into
    // columns of different lengths, after which positional indexing
    // downstream (`ohlcv.open[i]`, chart rendering, feature extraction)
    // would panic out of bounds. The write path validates every row in
    // `normalize_ohlcv`; mirror a cheap length check here so a bad file
    // fails with a clear error instead of a later panic.
    let n = close.len();
    let ts_len = timestamp.len();
    if open.len() != n
        || high.len() != n
        || low.len() != n
        || ts_len != n
        || volume.as_ref().is_some_and(|v| v.len() != n)
    {
        bail!(
            "Vortex column length mismatch (timestamp={ts_len} open={} high={} low={} close={n}) \
             — the file is corrupt or truncated; re-import it.",
            open.len(),
            high.len(),
            low.len(),
        );
    }

    let ohlcv = Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume,
    };
    validate_canonical_ohlcv(&ohlcv).context("validate canonical Vortex OHLCV")?;
    Ok(ohlcv)
}

fn extract_canonical_volume_as_f64(
    array: &vortex_array::ArrayRef,
    label: &str,
) -> Result<Vec<f64>> {
    match array.dtype() {
        DType::Primitive(PType::F64, _) => extract_non_null_primitive_vec::<f64>(array, label),
        DType::Primitive(PType::U64, _) => {
            let raw = extract_non_null_primitive_vec::<u64>(array, label)?;
            raw.into_iter()
                .map(|value| {
                    if !u64_has_exact_f64_mapping(value) {
                        bail!(
                            "raw UInt64 volume {value} has no exact f64 mapping; volume-dependent plans are unsupported"
                        );
                    }
                    Ok(value as f64)
                })
                .collect()
        }
        DType::Primitive(PType::I64, _) => {
            let raw = extract_non_null_primitive_vec::<i64>(array, label)?;
            raw.into_iter()
                .map(|value| {
                    if value < 0 {
                        bail!("raw Int64 volume {value} is negative");
                    }
                    if !u64_has_exact_f64_mapping(value as u64) {
                        bail!(
                            "raw Int64 volume {value} has no exact f64 mapping; volume-dependent plans are unsupported"
                        );
                    }
                    Ok(value as f64)
                })
                .collect()
        }
        DType::Primitive(PType::F32, _) => bail!(
            "Vortex volume is Float32 and precision-unrecoverable; implicit widening is forbidden"
        ),
        other => bail!("Vortex volume must be non-nullable f64/u64/i64, got {other}"),
    }
}

pub(crate) fn u64_has_exact_f64_mapping(value: u64) -> bool {
    if value == 0 {
        return true;
    }
    let significant_bits = u64::BITS - value.leading_zeros();
    significant_bits <= f64::MANTISSA_DIGITS
        || value.trailing_zeros() >= significant_bits - f64::MANTISSA_DIGITS
}

fn extract_non_null_primitive_vec<T: NativePType>(
    array: &vortex_array::ArrayRef,
    label: &str,
) -> Result<Vec<T>> {
    if !array
        .all_valid()
        .with_context(|| format!("failed to inspect {label} validity"))?
    {
        bail!(
            "Column '{label}' has null values — the source data has gaps; \
             re-import after filling/trimming them."
        );
    }

    Ok(array.to_primitive().as_slice::<T>().to_vec())
}

fn validate_ohlcv_row(row: &OhlcvRow) -> Result<()> {
    if !row.open.is_finite()
        || !row.high.is_finite()
        || !row.low.is_finite()
        || !row.close.is_finite()
        || row.volume.is_some_and(|value| !value.is_finite())
    {
        bail!(
            "NaN/Inf in OHLCV at timestamp {} (open={} high={} low={} close={}) — \
             re-import and verify the price data is clean.",
            row.timestamp,
            row.open,
            row.high,
            row.low,
            row.close
        );
    }
    // A zero or negative price passes every structural check below —
    // `open=0, high=0.837, low=0, close=0.834` has high ≥ low and both open and
    // close inside the range — while being economically impossible. Canonical
    // reads and writes both reject it; no runtime row dropping is permitted.
    if row.open <= 0.0 || row.high <= 0.0 || row.low <= 0.0 || row.close <= 0.0 {
        bail!(
            "non-positive price in OHLCV at timestamp {} (open={} high={} low={} close={})              — a zero price is the absence of a price, not a cheap one; re-import from a              source that has this bar.",
            row.timestamp,
            row.open,
            row.high,
            row.low,
            row.close
        );
    }
    if row.high < row.low
        || row.open < row.low
        || row.open > row.high
        || row.close < row.low
        || row.close > row.high
    {
        bail!(
            "Invalid OHLC row at timestamp {} (open={} high={} low={} close={}) — \
             source has bad candles; re-import or trim them.",
            row.timestamp,
            row.open,
            row.high,
            row.low,
            row.close
        );
    }
    if row.volume.is_some_and(|value| value < 0.0) {
        bail!("negative volume detected");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct OhlcvRow {
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: Option<f64>,
}

pub fn compute_hpc_features(source: &CanonicalOhlcvFrame) -> Result<FeatureFrame> {
    compute_hpc_feature_frame(source, FeatureProfile::Standard)
}

pub fn compute_hpc_feature_frame(
    source: &CanonicalOhlcvFrame,
    profile: FeatureProfile,
) -> Result<FeatureFrame> {
    compute_hpc_feature_frame_sized(source, profile, source.len())
}

type ProductionFeatureColumns = Vec<FeatureColumnF64>;

/// Compiler-checked dispatcher for every top-level scalar producer in the
/// production feature plan. The public manifest and the runtime both use the
/// same typed ids, so a family cannot remain reachable while being invisible
/// to provenance/validity review (the previous Footprint defect).
fn compute_production_feature_columns(
    feature_math_authority: MultiTimeframeFeatureMathAuthorityV3,
    producer: ProductionFeatureProducerId,
    source: &CanonicalOhlcvFrame,
    budget_rows: usize,
    classic_run_plan: Option<&crate::core::hpc_ta::ClassicTaRunPlan>,
    control: &FeatureBuildControl,
) -> Result<ProductionFeatureColumns> {
    control.report("feature_family", production_feature_node_id(producer), 0, 1)?;
    let ohlcv = source.ohlcv();
    let columns = match producer {
        ProductionFeatureProducerId::SmartMoneyConcept => compute_smc_feature_columns_f64(ohlcv),
        ProductionFeatureProducerId::ClassicVectorTa => match classic_run_plan {
            Some(run_plan) => {
                crate::core::hpc_ta::compute_classic_ta_feature_columns_f64_with_control(
                    ohlcv, run_plan, control,
                )
            }
            None => {
                let plan = crate::core::hpc_ta::prepare_classic_ta_run_plan(
                    budget_rows.max(ohlcv.len()),
                    control.resolved_indicator_compute_policy(),
                )?;
                crate::core::hpc_ta::compute_classic_ta_feature_columns_f64_with_control(
                    ohlcv, &plan, control,
                )
            }
        },
        ProductionFeatureProducerId::Quantitative => match feature_math_authority {
            MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy => {
                compute_quant_feature_columns_f64(ohlcv)
            }
            #[cfg(feature = "gpu-cuda")]
            MultiTimeframeFeatureMathAuthorityV3::ResidentGpuExactParityCpuReferenceV3 => {
                compute_quant_feature_columns_v4_f64(
                    source.ohlcv(),
                    source.artifact().frame_timeframe(),
                )
            }
        },
        ProductionFeatureProducerId::Session => compute_session_feature_columns_f64(ohlcv),
        ProductionFeatureProducerId::Regime => compute_regime_feature_columns_f64(ohlcv),
        ProductionFeatureProducerId::Footprint => compute_footprint_feature_columns_f64(ohlcv),
    }?;
    control.report("feature_family", production_feature_node_id(producer), 1, 1)?;
    Ok(columns)
}

fn production_feature_manifest_row(
    feature_math_authority: MultiTimeframeFeatureMathAuthorityV3,
    producer: ProductionFeatureProducerId,
) -> Result<&'static ProductionFeatureProducerManifestRowV1> {
    #[cfg(not(feature = "gpu-cuda"))]
    let _ = feature_math_authority;
    #[cfg(feature = "gpu-cuda")]
    if feature_math_authority
        == MultiTimeframeFeatureMathAuthorityV3::ResidentGpuExactParityCpuReferenceV3
        && producer == ProductionFeatureProducerId::Quantitative
    {
        return Ok(quantitative_feature_producer_manifest_v4()?);
    }

    production_feature_producer_manifest_v1()?
        .iter()
        .find(|row| row.producer() == producer)
        .ok_or_else(|| anyhow::anyhow!("production producer {producer:?} has no manifest"))
}

fn production_feature_node_id(producer: ProductionFeatureProducerId) -> &'static str {
    match producer {
        ProductionFeatureProducerId::SmartMoneyConcept => "producer:smart-money-concept",
        ProductionFeatureProducerId::ClassicVectorTa => "producer:classic-vector-ta",
        ProductionFeatureProducerId::Quantitative => "producer:quantitative",
        ProductionFeatureProducerId::Session => "producer:session",
        ProductionFeatureProducerId::Regime => "producer:regime",
        ProductionFeatureProducerId::Footprint => "producer:footprint",
    }
}

fn direct_frame_source_node(
    source: &CanonicalOhlcvFrame,
    source_node_id: &str,
    source_token: &str,
) -> Result<(String, String)> {
    let frame_timeframe = source.artifact().frame_timeframe();
    let frame_token = format!("{source_token}:{}", frame_timeframe.as_str());
    anyhow::ensure!(
        source.artifact().identity().timeframe() == frame_timeframe,
        "direct canonical frame timeframe disagrees with its dataset identity"
    );
    Ok((source_node_id.to_owned(), frame_token))
}

fn build_production_feature_contract(
    source: &CanonicalOhlcvFrame,
    groups: &[(ProductionFeatureProducerId, ProductionFeatureColumns)],
    feature_math_authority: MultiTimeframeFeatureMathAuthorityV3,
) -> Result<(
    neoethos_feature_contracts::FeaturePlanV1,
    neoethos_feature_contracts::DatasetFeatureArtifactProvenanceV1,
)> {
    use neoethos_feature_contracts::{
        DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOperationTagV1, FeatureOutputV1,
        FeaturePlanV1,
    };

    const PHYSICAL_SCHEMA_ID: &str = "neoethos.ohlcv.f64-ms.v1";
    // Formula review is a separate Task-16B gate. A zero formula-evidence hash
    // is explicit and cannot be mistaken for independently reviewed evidence;
    // exact implementation semantics are still bound by SemanticSourceSetV1.
    const UNREVIEWED_FORMULA_EVIDENCE: [u8; 32] = [0; 32];

    let source_node_id = format!(
        "source:{}",
        source.artifact().identity().to_path_component()
    );
    let source_semantic_hash: [u8; 32] = Sha256::digest(PHYSICAL_SCHEMA_ID.as_bytes()).into();
    let mut physical_outputs = vec!["open", "high", "low", "close"]
        .into_iter()
        .map(|name| FeatureOutputV1::f64(format!("physical:{name}"), 1))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if source.ohlcv().volume.is_some() {
        physical_outputs.push(FeatureOutputV1::f64("physical:volume", 1)?);
    }
    let source_node = FeatureNodeV1::source(
        source_node_id.clone(),
        source.artifact().identity().clone(),
        PHYSICAL_SCHEMA_ID,
        1,
        physical_outputs,
        source_semantic_hash,
    )?;

    let mut nodes = Vec::with_capacity(groups.len() + 1);
    nodes.push(source_node);
    let (producer_input_node_id, _) = direct_frame_source_node(source, &source_node_id, "single")?;
    let mut final_outputs = Vec::new();
    for (producer, columns) in groups {
        let manifest = production_feature_manifest_row(feature_math_authority, *producer)?;
        let outputs = columns
            .iter()
            .map(|column| FeatureOutputV1::f64(column.name.clone(), manifest.semantic_version()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        final_outputs.extend(columns.iter().map(|column| column.name.clone()));
        nodes.push(FeatureNodeV1::transform(
            production_feature_node_id(*producer),
            FeatureOperationTagV1::Indicator,
            manifest.semantic_version(),
            vec![producer_input_node_id.clone()],
            outputs,
            Vec::new(),
            UNREVIEWED_FORMULA_EVIDENCE,
            *manifest.semantic_source_set().identity().as_bytes(),
            None,
        )?);
    }
    let plan = FeaturePlanV1::new(nodes, final_outputs)?;
    let provenance = DatasetFeatureArtifactProvenanceV1::new(
        &plan,
        vec![source.source_binding(source_node_id)?],
    )?;
    Ok((plan, provenance))
}

/// Same as [`compute_hpc_feature_frame`], but with the indicator vocabulary
/// budget sized against `budget_rows` instead of against THIS frame.
///
/// # Why a multi-timeframe build MUST use this
///
/// `hpc_ta` turns free RAM into a maximum column count and then admits the
/// prefix of `ALL_INDICATORS` that fits. Size that from each frame's own row
/// count and the admitted ID SET becomes a function of the frame — and every
/// timeframe has a different row count. Measured on the operator's box
/// (20.6 GB free): base M5 at 1,054,320 bars admits 269 ids, H1 at 70,288
/// admits all 342, H4 at 17,572 admits all 342. The per-timeframe blocks then
/// differ in width by ~140 columns, `try_assemble_cube_in_ram`'s width
/// invariant refuses (correctly) to assemble, and every run on a box under
/// roughly 40 GB free silently falls through to the slower streaming disk path.
///
/// So the cube builder sizes ONE budget from the run's WIDEST frame — the base
/// timeframe — and passes that row count to every timeframe. Conservative by
/// construction: the higher timeframes are charged the base frame's per-column
/// price, so the plan can only over-reserve, never under-reserve.
pub fn compute_hpc_feature_frame_sized(
    source: &CanonicalOhlcvFrame,
    profile: FeatureProfile,
    budget_rows: usize,
) -> Result<FeatureFrame> {
    let classic_run_plan = crate::core::hpc_ta::prepare_classic_ta_run_plan(
        budget_rows.max(source.len()),
        crate::core::hpc_ta::resolved_indicator_compute_policy(),
    )?;
    compute_hpc_feature_frame_sized_with_classic_plan(
        source,
        profile,
        budget_rows,
        &classic_run_plan,
        MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy,
        &FeatureBuildControl::default(),
    )
}

fn compute_hpc_feature_frame_sized_with_classic_plan(
    source: &CanonicalOhlcvFrame,
    _profile: FeatureProfile,
    budget_rows: usize,
    classic_run_plan: &crate::core::hpc_ta::ClassicTaRunPlan,
    feature_math_authority: MultiTimeframeFeatureMathAuthorityV3,
    control: &FeatureBuildControl,
) -> Result<FeatureFrame> {
    control.checkpoint()?;
    // Check the retained Classic output BEFORE any of the concurrent families
    // starts. VocabularyBudget's historical floor and an older frozen plan
    // must not authorize an allocation that cannot fit now. This is a necessary
    // output bound, not a complete estimate of all six producers' scratch.
    require_source_feature_output_headroom(
        source.len(),
        &classic_run_plan.admission_report(),
        neoethos_core::allocation_headroom_bytes(),
    )
    .with_context(|| {
        format!(
            "admitting direct {} source before feature producers start",
            source.artifact().frame_timeframe()
        )
    })?;
    let ohlcv = source.ohlcv();

    // Perf (2026-07-02, operator: ">24h on dense TFs, one core pinned"): the
    // five indicator families used to run one-after-another and four of them
    // are internally single-threaded, so on M1/M3 (millions of rows) this
    // phase held ONE core for hours. Run the families CONCURRENTLY via a
    // rayon::join nest (classic_ta's internal par_iter work-steals within the
    // same pool). Memory: peak is unchanged — all five column sets existed
    // simultaneously before this change too (they were pushed into `columns`).
    //
    // PARITY-CRITICAL: column ORDER feeds effective_feature_names and every
    // discovery artifact — it must stay EXACTLY smc → classic → quant →
    // session → regime → footprint. rayon::join returns results by POSITION (not by
    // completion), so the chained collection below is deterministic.
    let [
        smc_id,
        classic_id,
        quant_id,
        session_id,
        regime_id,
        footprint_id,
    ] = PRODUCTION_FEATURE_PRODUCER_ORDER;
    let (smc, (classic, (quant, (session, (regime, footprint))))) = rayon::join(
        || {
            compute_production_feature_columns(
                feature_math_authority,
                smc_id,
                source,
                budget_rows,
                Some(classic_run_plan),
                control,
            )
        },
        || {
            rayon::join(
                || {
                    compute_production_feature_columns(
                        feature_math_authority,
                        classic_id,
                        source,
                        budget_rows,
                        Some(classic_run_plan),
                        control,
                    )
                },
                || {
                    rayon::join(
                        || {
                            compute_production_feature_columns(
                                feature_math_authority,
                                quant_id,
                                source,
                                budget_rows,
                                Some(classic_run_plan),
                                control,
                            )
                        },
                        || {
                            rayon::join(
                                || {
                                    compute_production_feature_columns(
                                        feature_math_authority,
                                        session_id,
                                        source,
                                        budget_rows,
                                        Some(classic_run_plan),
                                        control,
                                    )
                                },
                                || {
                                    rayon::join(
                                        || {
                                            compute_production_feature_columns(
                                                feature_math_authority,
                                                regime_id,
                                                source,
                                                budget_rows,
                                                Some(classic_run_plan),
                                                control,
                                            )
                                        },
                                        // Footprint family (2026-07-02): bar-level
                                        // effort-vs-result order-flow proxies.
                                        // APPENDED LAST so every pre-existing
                                        // portfolio's column order is unchanged
                                        // (projection matches by name).
                                        || {
                                            compute_production_feature_columns(
                                                feature_math_authority,
                                                footprint_id,
                                                source,
                                                budget_rows,
                                                Some(classic_run_plan),
                                                control,
                                            )
                                        },
                                    )
                                },
                            )
                        },
                    )
                },
            )
        },
    );

    // `compute_classic_ta_columns` now returns a Result: it hard-errors when
    // the indicator vocabulary collapses below its measured floor (the
    // 341-silent-drop regression). Propagated here rather than logged, because
    // a feature frame built on 66 of 800 columns is not a degraded frame — it
    // is a different search, and every artifact from it would be labelled as
    // though it were the same.
    let smc = smc?;
    let classic = classic?;
    let quant = quant?;
    let session = session?;
    let regime = regime?;
    let footprint = footprint?;
    control.checkpoint()?;

    let groups = vec![
        (smc_id, smc),
        (classic_id, classic),
        (quant_id, quant),
        (session_id, session),
        (regime_id, regime),
        (footprint_id, footprint),
    ];
    let (plan, provenance) =
        build_production_feature_contract(source, &groups, feature_math_authority)?;
    let columns = groups
        .into_iter()
        .flat_map(|(_, columns)| columns)
        .collect::<Vec<_>>();

    let n_rows = ohlcv.len();
    for column in &columns {
        anyhow::ensure!(
            column.len() == n_rows,
            "feature column '{}' has {} values but the frame has {n_rows} bars — refusing to \
             zero-pad a feature (a padded zero is indistinguishable from a real reading)",
            column.name,
            column.len()
        );
    }

    // HARD FAIL: a FeatureFrame without timestamps cannot be joined with
    // labels, so a silent empty-Vec fallback masks an upstream loader bug.
    let timestamps = ohlcv
        .timestamp
        .clone()
        .ok_or_else(|| anyhow::anyhow!("compute_hpc_feature_frame: OHLCV is missing timestamps"))?;
    FeatureFrame::from_canonical_columns(
        timestamps,
        columns,
        plan,
        provenance,
        vec![std::sync::Arc::clone(source.artifact().lease())],
    )
}

/// Bytes required by the frozen Classic output on THIS source grid. The plan
/// was sized against the largest direct frame, but a shorter higher timeframe
/// must not be charged that frame's rows again. Include base, historical and
/// extended sweeps even when the old vocabulary floor exceeded its budget.
/// The other feature families, producer scratch and allocator overhead are
/// additional; passing this lower-bound check is not whole-run RAM admission.
fn require_source_feature_output_headroom(
    source_rows: usize,
    report: &crate::core::hpc_ta::ClassicTaExecutionReport,
    available_bytes: u64,
) -> Result<u64> {
    anyhow::ensure!(
        source_rows <= report.budget_rows,
        "source feature output has {source_rows} rows, exceeding its frozen {}-row plan",
        report.budget_rows
    );
    let planned_columns = report
        .admitted_base_columns
        .checked_add(report.historical_sweep_reserved_columns)
        .and_then(|columns| columns.checked_add(report.extended_planned_columns))
        .context("source feature output column count overflowed")?;
    let required_bytes = u64::try_from(source_rows)?
        .checked_mul(u64::try_from(planned_columns)?)
        .and_then(|cells| {
            cells.checked_mul(
                (std::mem::size_of::<f64>() + std::mem::size_of::<FeatureCellValidity>()) as u64,
            )
        })
        .context("source feature output byte count overflowed")?;
    anyhow::ensure!(
        available_bytes > 0 && required_bytes <= available_bytes,
        "source feature allocation refused before computation: the frozen Classic output alone requires {required_bytes} bytes ({source_rows} rows, {planned_columns} columns), but only {available_bytes} bytes of host allocation headroom were measured; scratch and other families need additional memory; no history or indicators were removed"
    );
    Ok(required_bytes)
}

pub fn prepare_multitimeframe_features(
    ds: &SymbolDataset,
    base_tf: &str,
    higher_tfs: &[&str],
) -> Result<FeatureFrame> {
    let opts = FeatureBuildOptions {
        higher_tfs: higher_tfs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    prepare_multitimeframe_features_with_options(ds, base_tf, &opts)
}

/// One canonical root for disposable Vortex-backed feature runs. Capacity
/// accounting and writers must resolve the same path through this API.
pub fn vortex_feature_run_root() -> PathBuf {
    std::env::temp_dir().join("neoethos_vortex_feature_runs")
}

/// [`prepare_multitimeframe_features`] for one streaming batch.
///
/// `batch = None` is byte-identical to `prepare_multitimeframe_features`.
pub fn prepare_multitimeframe_features_batch(
    ds: &SymbolDataset,
    base_tf: &str,
    higher_tfs: &[&str],
    batch: Option<std::sync::Arc<crate::core::hpc_ta::SweepBatch>>,
) -> Result<FeatureFrame> {
    let opts = FeatureBuildOptions {
        higher_tfs: higher_tfs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    prepare_multitimeframe_features_batch_with_options(ds, base_tf, &opts, batch)
}

/// [`prepare_multitimeframe_features_with_options`] for one streaming batch.
///
/// This is the research-safe variant: normalization training rows, base-prefix
/// policy, and every other typed feature-build option remain identical across
/// batches while only the bounded `(indicator, period)` working set advances.
/// `batch = None` is byte-identical to the ordinary options entry point.
pub fn prepare_multitimeframe_features_batch_with_options(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    batch: Option<std::sync::Arc<crate::core::hpc_ta::SweepBatch>>,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_batch_with_options_and_control(
        ds,
        base_tf,
        opts,
        batch,
        &FeatureBuildControl::default(),
    )
}

/// One streaming working set through the common cancellable producer. The
/// resulting frame owns the exact replay selection in its feature options.
pub fn prepare_multitimeframe_features_batch_with_options_and_control(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    batch: Option<std::sync::Arc<crate::core::hpc_ta::SweepBatch>>,
    control: &FeatureBuildControl,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_with_feature_math_authority_v3(
        ds,
        base_tf,
        opts,
        None,
        MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy,
        batch,
        control,
        None,
        None,
    )
}

/// [`compute_hpc_feature_frame_sized`] for one streaming batch.
///
/// `batch = None` is byte-identical to `compute_hpc_feature_frame_sized`.
pub fn compute_hpc_feature_frame_batch(
    source: &CanonicalOhlcvFrame,
    profile: FeatureProfile,
    budget_rows: usize,
    batch: Option<std::sync::Arc<crate::core::hpc_ta::SweepBatch>>,
) -> Result<FeatureFrame> {
    let classic_run_plan = crate::core::hpc_ta::prepare_classic_ta_run_plan_with_working_set(
        budget_rows.max(source.len()),
        crate::core::hpc_ta::resolved_indicator_compute_policy(),
        batch,
    )?;
    compute_hpc_feature_frame_sized_with_classic_plan(
        source,
        profile,
        budget_rows,
        &classic_run_plan,
        MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy,
        &FeatureBuildControl::default(),
    )
}

#[derive(Debug)]
struct MultiTimeframeFeatureContractBlock {
    timeframe: String,
    source: CanonicalOhlcvFrame,
    original_names: Vec<String>,
    projected_names: Vec<String>,
    higher_timeframe: bool,
    availability_rule: String,
    availability_lag_ms: Option<i64>,
    max_age_ms: Option<i64>,
}

struct CompletedHigherTimeframeFeatureBlock {
    contract: Option<MultiTimeframeFeatureContractBlock>,
    sink: MultiTimeframeFeatureSink,
    normalization_fits: Vec<crate::core::normalization::RobustNormalizationFitF64>,
    dropped: Vec<String>,
}

#[derive(Debug)]
struct HigherTimeframeBuildTask {
    timeframe: String,
    source: CanonicalOhlcvFrame,
    estimated_peak_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HigherTimeframeBuildWave {
    task_range: std::ops::Range<usize>,
    estimated_peak_bytes: u64,
    task_memory: Vec<HigherTimeframeTaskMemoryPlan>,
}

const HIGHER_TIMEFRAME_SYSTEM_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const HIGHER_TIMEFRAME_TASK_FIXED_BYTES: u64 = 64 * 1024 * 1024;
const HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES: u64 = 16;
// Two f64 training buffers plus conservative stable-sort/allocator scratch.
// Parallelism is across columns; the floating-point operations inside each
// column remain exactly those of normalize_feature_column_f64.
const NORMALIZATION_SCRATCH_BYTES_PER_ROW: u64 = 32;

fn normalization_scratch_bytes(rows: usize) -> u64 {
    (rows as u64).saturating_mul(NORMALIZATION_SCRATCH_BYTES_PER_ROW)
}

fn admit_normalization_workers(
    rows: usize,
    columns: usize,
    scratch_budget_bytes: u64,
    max_workers: usize,
) -> Result<usize> {
    if columns == 0 {
        return Ok(0);
    }
    anyhow::ensure!(
        rows > 0 && max_workers > 0,
        "normalization requires rows and workers"
    );
    let per_worker = normalization_scratch_bytes(rows);
    let affordable = usize::try_from(scratch_budget_bytes / per_worker).unwrap_or(usize::MAX);
    let workers = columns.min(max_workers).min(affordable);
    anyhow::ensure!(
        workers > 0,
        "normalization scratch needs at least {per_worker} bytes, but only {scratch_budget_bytes} bytes are admitted; no training rows or indicators were removed"
    );
    Ok(workers)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HigherTimeframeTaskMemoryPlan {
    alignment_batch_columns: usize,
    normalization_workers: usize,
    estimated_peak_bytes: u64,
}

fn higher_timeframe_fixed_work_bytes(
    source_rows: usize,
    base_rows: usize,
    feature_columns: usize,
    normalize: bool,
) -> u64 {
    let source_cells = (source_rows as u64).saturating_mul(feature_columns as u64);
    source_cells
        // The complete source-grid producer still runs once, using the frozen
        // vocabulary. Column batching only changes alignment/storage, not math.
        .saturating_mul(HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES + 8)
        .saturating_add(
            (source_rows as u64).saturating_mul(std::mem::size_of::<Option<i64>>() as u64),
        )
        // The minimum includes ONE normalizer. The task planner separately
        // reserves every additional concurrent column before admitting it.
        .saturating_add(if normalize {
            normalization_scratch_bytes(base_rows)
        } else {
            0
        })
        .saturating_add(HIGHER_TIMEFRAME_TASK_FIXED_BYTES)
}

fn estimate_higher_timeframe_task_peak_bytes(
    source_rows: usize,
    base_rows: usize,
    feature_columns: usize,
    normalize: bool,
    max_normalization_workers: usize,
) -> u64 {
    let aligned_cells = (base_rows as u64).saturating_mul(feature_columns as u64);
    higher_timeframe_fixed_work_bytes(source_rows, base_rows, feature_columns, normalize)
        .saturating_add(aligned_cells.saturating_mul(HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES))
        .saturating_add(if normalize {
            normalization_scratch_bytes(base_rows).saturating_mul(
                max_normalization_workers
                    .max(1)
                    .min(feature_columns)
                    .saturating_sub(1) as u64,
            )
        } else {
            0
        })
}

fn plan_higher_timeframe_task_memory(
    source_rows: usize,
    base_rows: usize,
    feature_columns: usize,
    normalize: bool,
    memory_budget_bytes: u64,
    stream_alignment: bool,
    max_normalization_workers: usize,
) -> Result<HigherTimeframeTaskMemoryPlan> {
    anyhow::ensure!(
        source_rows > 0 && base_rows > 0 && feature_columns > 0,
        "higher-timeframe memory admission requires nonempty source, base and feature columns"
    );
    let fixed =
        higher_timeframe_fixed_work_bytes(source_rows, base_rows, feature_columns, normalize);
    let per_column = (base_rows as u64).saturating_mul(HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES);
    anyhow::ensure!(
        !normalize || max_normalization_workers > 0,
        "normalization requires an admitted worker limit"
    );
    let maximum = if normalize {
        max_normalization_workers.min(feature_columns)
    } else {
        1
    };
    // Prefer the available CPU width, reducing only simultaneous scratch when
    // RAM is tight. Even a one-column budget must retain the full vocabulary.
    for workers in (1..=maximum).rev() {
        let additional_scratch = if normalize {
            normalization_scratch_bytes(base_rows).saturating_mul(workers.saturating_sub(1) as u64)
        } else {
            0
        };
        let Some(fixed_with_scratch) = fixed.checked_add(additional_scratch) else {
            continue;
        };
        let Some(remaining) = memory_budget_bytes.checked_sub(fixed_with_scratch) else {
            continue;
        };
        let affordable = usize::try_from(remaining / per_column).unwrap_or(usize::MAX);
        let alignment_batch_columns = feature_columns.min(affordable);
        if alignment_batch_columns < workers
            || (!stream_alignment && alignment_batch_columns != feature_columns)
        {
            continue;
        }
        return Ok(HigherTimeframeTaskMemoryPlan {
            alignment_batch_columns,
            normalization_workers: if normalize { workers } else { 0 },
            estimated_peak_bytes: fixed_with_scratch + per_column * alignment_batch_columns as u64,
        });
    }
    anyhow::bail!(
        "higher-timeframe RAM admission refused: complete source producer plus {} aligned columns and normalization scratch do not fit within {memory_budget_bytes} bytes; spill retained output to Vortex before admission when necessary; no history or indicators were removed",
        if stream_alignment { 1 } else { feature_columns }
    )
}

fn higher_timeframe_parallel_budget_bytes(
    in_ram: bool,
    available_bytes: u64,
    final_cube_bytes: u64,
    resident_base_block_bytes: u64,
) -> u64 {
    let after_system_reserve =
        available_bytes.saturating_sub(HIGHER_TIMEFRAME_SYSTEM_RESERVE_BYTES);
    if in_ram {
        // Every not-yet-built output block must remain available for the final
        // in-memory cube. Only the remaining headroom may fund concurrent
        // producer/alignment scratch.
        after_system_reserve
            .saturating_sub(final_cube_bytes.saturating_sub(resident_base_block_bytes))
    } else {
        // Vortex releases every completed wave. Keep one third of the free
        // post-reserve memory outside this scheduler for input frames, allocator
        // fragmentation, the runtime and unrelated process work.
        after_system_reserve.saturating_mul(2) / 3
    }
}

fn plan_higher_timeframe_build_waves(
    source_rows: &[usize],
    base_rows: usize,
    feature_columns: usize,
    normalize: bool,
    memory_budget_bytes: u64,
    stream_alignment: bool,
    worker_threads: usize,
) -> Result<Vec<HigherTimeframeBuildWave>> {
    if source_rows.is_empty() {
        return Ok(Vec::new());
    }
    anyhow::ensure!(
        memory_budget_bytes > 0 && worker_threads > 0,
        "higher-timeframe wave admission requires measured headroom and CPU workers"
    );
    let per_column = u64::try_from(base_rows)?
        .checked_mul(HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES)
        .context("higher-timeframe aligned column size overflowed")?;
    let minimum_alignment = per_column
        .checked_mul(u64::try_from(if stream_alignment {
            1
        } else {
            feature_columns
        })?)
        .context("higher-timeframe minimum alignment size overflowed")?;
    // Reserve each complete source producer and minimum alignment BEFORE
    // distributing spare memory. Giving every task the whole wave budget first
    // maximized each alignment buffer and made the later packing serial.
    let mut minima = Vec::with_capacity(source_rows.len());
    for (index, &rows) in source_rows.iter().enumerate() {
        let minimum =
            higher_timeframe_fixed_work_bytes(rows, base_rows, feature_columns, normalize)
                .checked_add(minimum_alignment)
                .context("higher-timeframe minimum task size overflowed")?;
        anyhow::ensure!(
            minimum <= memory_budget_bytes,
            "higher-timeframe task {index} needs at least {minimum} bytes, exceeding its {memory_budget_bytes}-byte wave budget; an oversized task must be streamed or refused, not run alone"
        );
        let memory = plan_higher_timeframe_task_memory(
            rows,
            base_rows,
            feature_columns,
            normalize,
            minimum,
            stream_alignment,
            1,
        )?;
        minima.push(memory.estimated_peak_bytes);
    }
    let mut waves = Vec::new();
    let mut start = 0usize;
    while start < source_rows.len() {
        // Parallelism must not turn a wide block into hundreds of tiny files.
        // This throughput heuristic targets at most one alignment shard per
        // global CPU worker per TF, not a mathematical memory requirement.
        // When a pair cannot meet it, the single-task fallback still receives
        // the whole budget and may stream smaller batches without losing data.
        let granularity = if stream_alignment {
            feature_columns.div_ceil(worker_threads)
        } else {
            feature_columns
        };
        let mut end = start + 1;
        let mut reserved = minima[start];
        let mut wave_minima = vec![minima[start]];
        for candidate_end in start + 2..=source_rows.len().min(start.saturating_add(worker_threads))
        {
            let count = candidate_end - start;
            let mut candidate_minima = Vec::with_capacity(count);
            let mut total = 0u64;
            for (offset, index) in (start..candidate_end).enumerate() {
                let workers = if normalize {
                    (worker_threads / count + usize::from(offset < worker_threads % count))
                        .min(feature_columns)
                } else {
                    1
                };
                let extra_scratch = if normalize {
                    normalization_scratch_bytes(base_rows)
                        .checked_mul(workers.saturating_sub(1) as u64)
                } else {
                    Some(0)
                };
                let minimum = extra_scratch
                    .and_then(|scratch| {
                        higher_timeframe_fixed_work_bytes(
                            source_rows[index],
                            base_rows,
                            feature_columns,
                            normalize,
                        )
                        .checked_add(scratch)
                    })
                    .and_then(|fixed| {
                        per_column
                            .checked_mul(granularity.max(workers) as u64)
                            .and_then(|aligned| fixed.checked_add(aligned))
                    });
                let Some((minimum, combined)) = minimum
                    .and_then(|minimum| total.checked_add(minimum).map(|sum| (minimum, sum)))
                else {
                    break;
                };
                if combined > memory_budget_bytes {
                    break;
                }
                total = combined;
                candidate_minima.push(minimum);
            }
            if candidate_minima.len() != count {
                break;
            }
            end = candidate_end;
            reserved = total;
            wave_minima = candidate_minima;
        }
        let count = end - start;
        let mut remaining = memory_budget_bytes - reserved;
        let mut estimated_peak_bytes = 0u64;
        let mut task_memory = Vec::with_capacity(count);
        for (offset, index) in (start..end).enumerate() {
            // Unequal source minima stay intact. Only the remaining alignment
            // budget is shared, with unused space carried to subsequent tasks.
            let task_budget = wave_minima[offset] + remaining / (count - offset) as u64;
            let workers = worker_threads / count + usize::from(offset < worker_threads % count);
            let memory = plan_higher_timeframe_task_memory(
                source_rows[index],
                base_rows,
                feature_columns,
                normalize,
                task_budget,
                stream_alignment,
                workers,
            )?;
            remaining -= memory.estimated_peak_bytes - wave_minima[offset];
            estimated_peak_bytes += memory.estimated_peak_bytes;
            task_memory.push(memory);
        }
        waves.push(HigherTimeframeBuildWave {
            task_range: start..end,
            estimated_peak_bytes,
            task_memory,
        });
        start = end;
    }
    Ok(waves)
}

fn prepare_multitimeframe_feature_columns(
    columns: &mut Vec<FeatureColumnF64>,
    normalize: bool,
    normalization_training_rows: Option<std::ops::Range<usize>>,
    drop_columns_without_normalization_training_support: bool,
    normalization_workers: usize,
    control: &FeatureBuildControl,
    fitted_normalization: Option<&SearchNormalizationFittedStateV1>,
) -> Result<(
    Vec<crate::core::normalization::RobustNormalizationFitF64>,
    Vec<String>,
)> {
    let mut dropped = Vec::new();
    let mut normalization_fits = Vec::new();
    let fitted_by_name = fitted_normalization.map(|state| {
        state
            .column_names()
            .iter()
            .zip(state.fits())
            .collect::<HashMap<_, _>>()
    });
    control.checkpoint()?;
    if normalize {
        let training_rows = normalization_training_rows
            .context("normalization is enabled without an explicit training row range")?;
        if let Some(fits) = &fitted_by_name {
            // This is schema replay, never a support estimate on OOS/live.
            columns.retain(|column| fits.contains_key(&column.name));
        } else if drop_columns_without_normalization_training_support {
            dropped =
                retain_columns_with_normalization_training_support(columns, training_rows.clone())?;
        }
        anyhow::ensure!(
            columns.is_empty() || normalization_workers > 0,
            "normalization requires admitted scratch workers"
        );
        let total = columns.len();
        control.report("normalization", "admitted column batch", 0, total)?;
        let completed = std::sync::Mutex::new(0usize);
        // Bound scratch explicitly even when called inside a higher-TF Rayon
        // task. Nested work uses the caller's installed CPU pool, not a new
        // pool. Indexed collection preserves column/fit/hash order, while the
        // unchanged scalar normalizer preserves the per-column reduction order.
        for batch in columns.chunks_mut(normalization_workers.max(1)) {
            let fits = batch
                .par_iter_mut()
                .map(|column| {
                    control.checkpoint()?;
                    let fit = match &fitted_by_name {
                        Some(fits) => {
                            let fit = fits
                                .get(&column.name)
                                .context("persisted normalization column disappeared")?;
                            crate::core::normalization::apply_search_normalization_fit(
                                column, fit,
                            )?;
                            (*fit).clone()
                        }
                        None => crate::core::normalization::normalize_search_feature_column_f64(
                            column,
                            training_rows.clone(),
                        )?,
                    };
                    let mut completed = completed
                        .lock()
                        .map_err(|_| anyhow::anyhow!("normalization progress lock poisoned"))?;
                    *completed += 1;
                    control.report("normalization", &column.name, *completed, total)?;
                    Ok(fit)
                })
                .collect::<Result<Vec<_>>>()?;
            normalization_fits.extend(fits);
        }
    }
    Ok((normalization_fits, dropped))
}

fn retain_columns_with_normalization_training_support(
    columns: &mut Vec<FeatureColumnF64>,
    training_rows: std::ops::Range<usize>,
) -> Result<Vec<String>> {
    for column in columns.iter() {
        anyhow::ensure!(
            training_rows.start < training_rows.end && training_rows.end <= column.len(),
            "feature column `{}` normalization support range {:?} is outside 0..{}",
            column.name,
            training_rows,
            column.len()
        );
    }

    let mut dropped = Vec::new();
    columns.retain(|column| {
        let supported = column.validity[training_rows.clone()]
            .iter()
            .any(|validity| validity.is_valid());
        if !supported {
            dropped.push(column.name.clone());
        }
        supported
    });
    Ok(dropped)
}

fn take_in_memory_columns(frame: FeatureFrame) -> Result<Vec<FeatureColumnF64>> {
    match frame.data {
        FeatureData::InMemory(columns) => Ok(columns),
        FeatureData::Vortex(_)
        | FeatureData::VortexSet(_)
        | FeatureData::VortexWindow(_)
        | FeatureData::View(_) => {
            anyhow::bail!("scalar feature computation unexpectedly returned a Vortex-backed frame")
        }
    }
}

/// Compute and causally align one immutable higher-timeframe source onto the
/// canonical millisecond base grid. No timestamp-unit inference or f32 bridge
/// exists on this path.
fn compute_aligned_higher_block(
    source: CanonicalOhlcvFrame,
    base_tf: &str,
    base_ns: &[i64],
    h_tf: &str,
    profile: FeatureProfile,
    budget_rows: usize,
    classic_run_plan: &crate::core::hpc_ta::ClassicTaRunPlan,
    feature_math_authority: MultiTimeframeFeatureMathAuthorityV3,
    task_memory_budget_bytes: u64,
    max_normalization_workers: usize,
    normalize: bool,
    stream_alignment: bool,
    control: &FeatureBuildControl,
    mut consume_columns: impl FnMut(Vec<FeatureColumnF64>, usize) -> Result<Vec<String>>,
) -> Result<Option<MultiTimeframeFeatureContractBlock>> {
    control.checkpoint()?;
    if h_tf == base_tf {
        return Ok(None);
    }
    let h_ohlcv = source.ohlcv();
    // Every independently downloaded timeframe uses the same run-wide budget
    // sized from the widest direct frame. Otherwise the admitted indicator set
    // (and therefore block width) could vary between timeframes.
    let h_feats = compute_hpc_feature_frame_sized_with_classic_plan(
        &source,
        profile,
        budget_rows,
        classic_run_plan,
        feature_math_authority,
        control,
    )?;
    let h_ns = h_ohlcv
        .timestamp
        .as_ref()
        .context("higher tf has no timestamps")?;
    // Higher-TF bars are OPEN-stamped, so final feature values must remain
    // hidden until the bar is closed. Fixed frames use the exact typed period.
    // Calendar frames never invent 24h/7d/30d durations: row N becomes
    // available only at the observed open of direct broker row N+1, while the
    // final row remains unavailable because its close is not evidenced.
    let h_timeframe = h_tf.parse::<CanonicalTimeframe>().map_err(|_| {
        anyhow::anyhow!(
            "cannot resolve canonical higher timeframe '{h_tf}' — refusing \
                 to align its features without close-availability (that would reintroduce \
                 up to one period of lookahead into the feature cube)"
        )
    })?;
    let fixed_period_ms = h_timeframe.fixed_duration_ms();
    let max_age_ms = fixed_period_ms.map(|period_ms| period_ms.saturating_mul(2));
    if let (Some(base_last), Some(h_last), Some(max_age)) =
        (base_ns.last().copied(), h_ns.last().copied(), max_age_ms)
    {
        if base_last > h_last {
            let staleness = base_last - h_last;
            if staleness > max_age {
                tracing::warn!(
                    target: "neoethos_data::prepare_multitimeframe_features",
                    base_tf = base_tf,
                    higher_tf = h_tf,
                    staleness_ms = staleness,
                    max_age_ms = max_age,
                    "higher-TF last bar is older than 2× period — feature columns past max_age will be NaN. Re-run --bootstrap-data for this (symbol, timeframe) to refresh."
                );
            }
        }
    }
    let original_names = h_feats.names.clone();
    let h_columns = take_in_memory_columns(h_feats)?;
    // Recheck the ACTUAL source output width before allocating an aligned
    // base-grid block. Never assume all producers emitted the base TF's width.
    let memory = plan_higher_timeframe_task_memory(
        source.len(),
        base_ns.len(),
        h_columns.len(),
        normalize,
        task_memory_budget_bytes,
        stream_alignment,
        max_normalization_workers,
    )?;
    tracing::info!(
        target: "neoethos_data::prepare_multitimeframe_features",
        higher_tf = h_tf,
        source_columns = h_columns.len(),
        alignment_batch_columns = memory.alignment_batch_columns,
        normalization_workers = memory.normalization_workers,
        estimated_peak_bytes = memory.estimated_peak_bytes,
        task_memory_budget_bytes,
        "higher-timeframe alignment width admitted without changing the source vocabulary"
    );
    let (availability_rule, availability_lag_ms) = if let Some(period_ms) = fixed_period_ms {
        ("fixed_open_plus_period_v1", Some(period_ms))
    } else {
        ("next_direct_bar_open_observed_span_expiry_v1", None)
    };
    let mut projected_names = Vec::with_capacity(h_columns.len());
    let mut pending_columns = h_columns.into_iter();
    loop {
        control.report(
            "alignment",
            h_tf,
            projected_names.len(),
            original_names.len(),
        )?;
        let batch = pending_columns
            .by_ref()
            .take(memory.alignment_batch_columns)
            .collect::<Vec<_>>();
        if batch.is_empty() {
            break;
        }
        let mut aligned = if let Some(period_ms) = fixed_period_ms {
            align_feature_columns_by_ms(base_ns, h_ns, &batch, true, max_age_ms, period_ms)?
        } else {
            align_calendar_feature_columns_by_observed_next_open_ms(base_ns, h_ns, &batch, true)?
        };
        drop(batch);
        for column in &mut aligned {
            column.name = format!("{}_{}", h_tf, column.name);
        }
        // The Vortex consumer persists/releases this batch before the next
        // allocation. The contract still has ONE full, ordered block per TF.
        projected_names.extend(consume_columns(aligned, memory.normalization_workers)?);
    }
    if projected_names.is_empty() {
        return Ok(None);
    }
    Ok(Some(MultiTimeframeFeatureContractBlock {
        timeframe: h_tf.to_owned(),
        source,
        original_names,
        projected_names,
        higher_timeframe: true,
        availability_rule: availability_rule.to_owned(),
        availability_lag_ms,
        max_age_ms,
    }))
}

fn producer_for_feature(name: &str) -> Result<ProductionFeatureProducerId> {
    let source = feature_column_metadata(name)
        .with_context(|| format!("feature `{name}` has no registered production source"))?
        .source;
    Ok(match source {
        FeatureSource::SmartMoneyConcept => ProductionFeatureProducerId::SmartMoneyConcept,
        FeatureSource::ClassicTechnicalAnalysis => ProductionFeatureProducerId::ClassicVectorTa,
        FeatureSource::Quantitative => ProductionFeatureProducerId::Quantitative,
        FeatureSource::Session => ProductionFeatureProducerId::Session,
        FeatureSource::Regime => ProductionFeatureProducerId::Regime,
        FeatureSource::Footprint => ProductionFeatureProducerId::Footprint,
    })
}

fn semantic_source_hash(parts: &[&[u8]]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_be_bytes());
        hash.update(part);
    }
    hash.finalize().into()
}

fn build_multitimeframe_feature_contract(
    blocks: &[MultiTimeframeFeatureContractBlock],
    feature_math_authority: MultiTimeframeFeatureMathAuthorityV3,
    normalization_fits: Option<&[crate::core::normalization::RobustNormalizationFitF64]>,
) -> Result<(
    neoethos_feature_contracts::FeaturePlanV1,
    neoethos_feature_contracts::DatasetFeatureArtifactProvenanceV1,
    Vec<std::sync::Arc<crate::core::dataset_generation_lease::DatasetGenerationLease>>,
)> {
    use neoethos_feature_contracts::{
        DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOperationTagV1, FeatureOutputV1,
        FeatureParameterV1, FeaturePlanV1,
    };

    anyhow::ensure!(
        !blocks.is_empty(),
        "multi-timeframe plan has no source blocks"
    );
    let physical_schema_hash = semantic_source_hash(&[b"neoethos.ohlcv.f64-ms.v1"]);
    let projection_source_hash = semantic_source_hash(&[include_bytes!("lib.rs")]);
    let alignment_source_hash =
        semantic_source_hash(&[include_bytes!("core/features.rs"), include_bytes!("lib.rs")]);
    let normalization_source_hash =
        semantic_source_hash(&[include_bytes!("core/normalization.rs")]);
    let normalized = normalization_fits.is_some();
    let mut nodes = Vec::new();
    let mut bindings = Vec::new();
    let mut projection_node_ids = Vec::new();
    let mut final_outputs = Vec::new();
    let mut source_leases = Vec::new();
    let mut registered_sources = std::collections::BTreeSet::new();

    for (block_index, block) in blocks.iter().enumerate() {
        anyhow::ensure!(
            !block.original_names.is_empty() && !block.projected_names.is_empty(),
            "{} feature block is empty",
            block.timeframe
        );
        let source_token = block.source.artifact().identity().to_path_component();
        let source_node_id = format!("source:{source_token}");
        if registered_sources.insert(source_node_id.clone()) {
            let mut physical_outputs = ["open", "high", "low", "close"]
                .into_iter()
                .map(|name| FeatureOutputV1::f64(format!("physical:{source_token}:{name}"), 1))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if block.source.ohlcv().volume.is_some() {
                physical_outputs.push(FeatureOutputV1::f64(
                    format!("physical:{source_token}:volume"),
                    1,
                )?);
            }
            nodes.push(FeatureNodeV1::source(
                source_node_id.clone(),
                block.source.artifact().identity().clone(),
                "neoethos.ohlcv.f64-ms.v1",
                1,
                physical_outputs,
                physical_schema_hash,
            )?);
            bindings.push(block.source.source_binding(source_node_id.clone())?);
            source_leases.push(std::sync::Arc::clone(block.source.artifact().lease()));
        }
        let (frame_input_node_id, frame_token) =
            direct_frame_source_node(&block.source, &source_node_id, &source_token)?;

        let mut producer_node_ids = Vec::new();
        for producer in PRODUCTION_FEATURE_PRODUCER_ORDER {
            let selected = block
                .original_names
                .iter()
                .enumerate()
                .filter_map(|(index, name)| {
                    producer_for_feature(name)
                        .map(|actual| (actual == producer).then_some((index, name)))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            anyhow::ensure!(
                !selected.is_empty(),
                "{} feature block emitted no columns for production producer {producer:?}",
                block.timeframe
            );
            let manifest = production_feature_manifest_row(feature_math_authority, producer)?;
            let node_id = format!("{}:{frame_token}", production_feature_node_id(producer));
            let outputs = selected
                .iter()
                .map(|(_, name)| {
                    FeatureOutputV1::f64(
                        format!("raw:{frame_token}:{name}"),
                        manifest.semantic_version(),
                    )
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            nodes.push(FeatureNodeV1::transform(
                node_id.clone(),
                FeatureOperationTagV1::Indicator,
                manifest.semantic_version(),
                vec![frame_input_node_id.clone()],
                outputs,
                Vec::new(),
                [0; 32],
                *manifest.semantic_source_set().identity().as_bytes(),
                None,
            )?);
            producer_node_ids.push(node_id);
        }

        let projection_node_id = format!("projection:{block_index}:{frame_token}");
        let projected_outputs = block
            .projected_names
            .iter()
            .map(|name| {
                let output_name = if normalized {
                    format!("pre-normalize:{block_index}:{name}")
                } else {
                    name.clone()
                };
                FeatureOutputV1::f64(
                    output_name,
                    if block.higher_timeframe {
                        HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION
                    } else {
                        1
                    },
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        nodes.push(FeatureNodeV1::transform(
            projection_node_id.clone(),
            if block.higher_timeframe {
                FeatureOperationTagV1::HigherTimeframeAlignment
            } else {
                FeatureOperationTagV1::Derived
            },
            if block.higher_timeframe {
                HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION
            } else {
                1
            },
            producer_node_ids,
            projected_outputs,
            vec![
                FeatureParameterV1::text("timeframe", block.timeframe.clone())?,
                FeatureParameterV1::bool("higher_timeframe", block.higher_timeframe)?,
                FeatureParameterV1::text("availability_rule", block.availability_rule.clone())?,
                FeatureParameterV1::i64(
                    "availability_lag_ms",
                    block.availability_lag_ms.unwrap_or(-1),
                )?,
                FeatureParameterV1::i64("max_age_ms", block.max_age_ms.unwrap_or(-1))?,
            ],
            if block.higher_timeframe {
                alignment_source_hash
            } else {
                projection_source_hash
            },
            if block.higher_timeframe {
                alignment_source_hash
            } else {
                projection_source_hash
            },
            None,
        )?);
        projection_node_ids.push(projection_node_id);
        final_outputs.extend(block.projected_names.iter().cloned());
    }

    if let Some(fits) = normalization_fits {
        let all_names = blocks
            .iter()
            .flat_map(|block| block.projected_names.iter().cloned())
            .collect::<Vec<_>>();
        let fitted_state_hash = normalization_fit_hash(&all_names, fits)?;
        let outputs = final_outputs
            .iter()
            .map(|name| {
                FeatureOutputV1::f64(
                    name.clone(),
                    crate::core::normalization::SEARCH_NORMALIZATION_POLICY_VERSION,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        nodes.push(FeatureNodeV1::transform(
            "normalization:robust-f64",
            FeatureOperationTagV1::Normalization,
            crate::core::normalization::SEARCH_NORMALIZATION_POLICY_VERSION,
            projection_node_ids,
            outputs,
            Vec::new(),
            normalization_source_hash,
            normalization_source_hash,
            Some(fitted_state_hash),
        )?);
    }

    let plan = FeaturePlanV1::new(nodes, final_outputs)?;
    let provenance = DatasetFeatureArtifactProvenanceV1::new(&plan, bindings)?;
    Ok((plan, provenance, source_leases))
}

/// Decide whether the multi-TF feature cube stays in RAM or is persisted as
/// independent Vortex shards. The decision is made after the base block is
/// known and before any higher-timeframe block is allocated. On the disk path
/// each bounded parallel wave is written and released before the next wave is
/// computed.
///
/// `models.data_runtime.feature_cube_mode=disk` can lower memory use. `auto`
/// and the historical `ram` policy remain bounded by the live-memory probe;
/// no configuration can bypass the live-memory guard. This estimate is not an
/// OS memory reservation; the higher-TF loop rechecks it before each wave.
/// Test-only seam replacing the deleted `NEOETHOS_FEATURE_CUBE_MODE`.
/// `0` = derive (production), `1` = force RAM, `2` = force disk.
///
/// The RAM and disk assemblies must produce BIT-IDENTICAL cubes — if they
/// diverge, discovery results depend on how much free RAM the machine happened
/// to have, which is the worst kind of non-determinism. Proving that needs a
/// way to run both paths on the same input, so the seam survives the env
/// deletion as something a test can reach and a shell cannot.
#[cfg(test)]
pub(crate) static TEST_CUBE_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Test-only seam for exercising normalization through both physical sinks.
/// `0` = configured production value, `1` = enabled, `2` = disabled.
#[cfg(test)]
static TEST_NORMALIZATION_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

// Private test seam, protected by the existing cube-mode test lock. Production
// always refreshes the existing memory probe before each higher-TF wave.
#[cfg(test)]
static TEST_HIGHER_TIMEFRAME_AVAILABLE_MEMORY: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

fn higher_timeframe_available_memory_bytes() -> u64 {
    #[cfg(test)]
    {
        let available =
            TEST_HIGHER_TIMEFRAME_AVAILABLE_MEMORY.load(std::sync::atomic::Ordering::Relaxed);
        if available != u64::MAX {
            return available;
        }
    }
    neoethos_core::allocation_headroom_bytes()
}

fn feature_normalization_enabled() -> bool {
    #[cfg(test)]
    {
        match TEST_NORMALIZATION_MODE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return true,
            2 => return false,
            _ => {}
        }
    }
    current_data_runtime_overrides().normalize_features
}

/// Peak RAM the in-memory assembly needs for a cube of `cube_bytes`: the cube
/// plus ONE timeframe block (~1.1x, demanded as 1.5x for margin) plus a 2 GB
/// floor for the OS and the GA's working buffers.
///
/// Split out of [`should_build_cube_in_ram`] on 2026-08-10. The rule and the
/// PROBE are two different things, and fusing them made the rule untestable:
/// `in_ram_budget_tracks_available_memory` read `available_memory_bytes()` to
/// compute a cube size that should just fail, and `should_build_cube_in_ram`
/// then read the probe AGAIN. Free RAM moves between two calls on any busy
/// machine — it moved during this very build — so the test failed on a
/// correct implementation. A guard that fails when nothing is wrong gets
/// deleted by the third person who sees it, and then the real regression ships.
fn cube_ram_requirement_bytes(cube_bytes: u64) -> f64 {
    (cube_bytes as f64) * 1.5 + 2.0e9
}

/// The whole decision, against a SUPPLIED reading. Deterministic: same inputs,
/// same answer, on any machine and at any load.
fn cube_fits_in(cube_bytes: u64, available_bytes: u64) -> bool {
    available_bytes != 0 && cube_ram_requirement_bytes(cube_bytes) < available_bytes as f64
}

fn should_build_cube_in_ram(cube_bytes: u64) -> bool {
    #[cfg(test)]
    {
        match TEST_CUBE_MODE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return true,
            2 => return false,
            _ => {}
        }
    }
    report_retired_env_vars();
    let configured = current_feature_cube_policy();
    let cube_gb = format!("{:.1}", cube_bytes as f64 / 1e9);

    // `disk` is always honoured: it can only LOWER peak memory, so there is
    // no conflict for the probe to resolve.
    if configured == neoethos_core::config::FeatureCubeMode::Disk {
        tracing::info!(
            target: "neoethos_data::feature_cube",
            cube_gb = %cube_gb,
            configured = "disk",
            in_ram = false,
            "feature-cube assembly forced to Vortex scratch shards by \
             models.data_runtime.feature_cube_mode"
        );
        return false;
    }

    let available = neoethos_core::allocation_headroom_bytes();
    if available == 0 {
        tracing::warn!(
            target: "neoethos_data::feature_cube",
            cube_gb = %cube_gb,
            configured = %configured.as_str(),
            "no host allocation headroom was measured — the RAM cube is not admitted; \
             Vortex storage still requires producer and wave admission"
        );
        return false;
    }
    let needed = cube_ram_requirement_bytes(cube_bytes);
    let fits = cube_fits_in(cube_bytes, available);

    tracing::info!(
        target: "neoethos_data::feature_cube",
        cube_gb = %cube_gb,
        needed_gb = format!("{:.1}", needed / 1e9),
        available_gb = format!("{:.1}", available as f64 / 1e9),
        configured = %configured.as_str(),
        in_ram = fits,
        "feature-cube assembly path derived from the free-RAM probe"
    );
    fits
}

/// Environment variables this crate used to honour and no longer does.
/// `(name, what decides it now)`.
const RETIRED_ENV_VARS: &[(&str, &str)] = &[
    (
        "NEOETHOS_FEATURE_CUBE_MODE",
        "models.data_runtime.feature_cube_mode, clamped by the free-RAM probe in \
         should_build_cube_in_ram (never-OOM invariant)",
    ),
    (
        "NEOETHOS_REQUIRE_GPU",
        "the indicator compute policy installed from Settings \
         (hpc_ta::set_indicator_compute_policy, called by \
         neoethos_search::backend::install_evaluation_backend_from_settings)",
    ),
];

/// Report, once per process and at ERROR, every retired environment variable
/// of this crate that is still exported. A stale export that is silently
/// ignored is the failure mode being removed; being ignored LOUDLY is not.
pub fn report_retired_env_vars() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        for (name, replacement) in RETIRED_ENV_VARS {
            let Ok(value) = std::env::var(name) else {
                continue;
            };
            if value.trim().is_empty() {
                continue;
            }
            tracing::error!(
                target: "neoethos_data::retired_env",
                env_var = %name,
                value_found = %value,
                decided_by = %replacement,
                "RETIRED ENVIRONMENT VARIABLE IS SET AND WAS IGNORED — this value did NOT \
                 reach the run."
            );
        }
    });
}

static FEATURE_RUN_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

enum MultiTimeframeFeatureSink {
    InMemory {
        columns: Vec<FeatureColumnF64>,
    },
    Vortex {
        scratch_root: PathBuf,
        run_id_prefix: String,
        shards: Vec<std::sync::Arc<crate::core::vortex_feature_store::VortexFeatureStore>>,
    },
}

impl MultiTimeframeFeatureSink {
    fn is_in_memory(&self) -> bool {
        matches!(self, Self::InMemory { .. })
    }

    fn resident_column_bytes(&self) -> u64 {
        match self {
            Self::InMemory { columns } => {
                columns.iter().fold(0u64, |total, column| {
                    total
                        .saturating_add((column.values.capacity() as u64).saturating_mul(8))
                        .saturating_add(
                            (column.validity.capacity() as u64)
                                .saturating_mul(std::mem::size_of::<FeatureCellValidity>() as u64),
                        )
                })
            }
            Self::Vortex { .. } => 0,
        }
    }

    /// Independent task outputs share the storage policy, not a lock or a
    /// mutable writer. The ordered wave merge below moves them into this sink.
    fn empty_peer(&self) -> Self {
        match self {
            Self::InMemory { .. } => Self::InMemory {
                columns: Vec::new(),
            },
            Self::Vortex {
                scratch_root,
                run_id_prefix,
                ..
            } => Self::Vortex {
                scratch_root: scratch_root.clone(),
                run_id_prefix: format!(
                    "{run_id_prefix}-task-{}",
                    FEATURE_RUN_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                ),
                shards: Vec::new(),
            },
        }
    }

    fn append(&mut self, other: Self) -> Result<()> {
        match (self, other) {
            (Self::InMemory { columns }, Self::InMemory { columns: completed }) => {
                columns.extend(completed)
            }
            (
                Self::Vortex { shards, .. },
                Self::Vortex {
                    shards: completed, ..
                },
            ) => shards.extend(completed),
            _ => anyhow::bail!("higher-timeframe task returned a different feature storage policy"),
        }
        Ok(())
    }

    fn spill_to_vortex(
        &mut self,
        symbol: &str,
        base_tf: &str,
        timestamps: &[i64],
        available_bytes: u64,
        control: &FeatureBuildControl,
    ) -> Result<()> {
        let Self::InMemory { columns } = self else {
            return Ok(());
        };
        let writer_budget = higher_timeframe_parallel_budget_bytes(false, available_bytes, 0, 0)
            .saturating_sub(HIGHER_TIMEFRAME_TASK_FIXED_BYTES);
        let column_bytes = (timestamps.len() as u64)
            .saturating_mul(HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES)
            .max(1);
        let batch_columns = usize::try_from(writer_budget / column_bytes).unwrap_or(usize::MAX);
        anyhow::ensure!(
            columns.is_empty() || batch_columns > 0,
            "higher-timeframe RAM admission cannot spill retained output: no measured scratch headroom for one Vortex column"
        );
        let mut replacement = Self::new(false, symbol, base_tf, 1)?;
        let mut pending = std::mem::take(columns).into_iter();
        loop {
            let batch = pending.by_ref().take(batch_columns).collect::<Vec<_>>();
            if batch.is_empty() {
                break;
            }
            replacement.push(base_tf, timestamps, batch, control)?;
        }
        *self = replacement;
        Ok(())
    }

    fn new(in_ram: bool, symbol: &str, base_tf: &str, timeframe_count: usize) -> Result<Self> {
        if in_ram {
            return Ok(Self::InMemory {
                columns: Vec::new(),
            });
        }

        let scratch_root = vortex_feature_run_root();
        let removed = crate::core::feature_run_lease::sweep_orphan_feature_runs(&scratch_root)?;
        if !removed.is_empty() {
            tracing::info!(
                target: "neoethos_data::vortex_feature_store",
                removed_runs = removed.len(),
                "removed crashed Vortex feature scratch runs after acquiring their OS leases"
            );
        }
        let sanitize = |value: &str| {
            value
                .chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .take(24)
                .collect::<String>()
        };
        let run_id_prefix = format!(
            "{}-{}-{}-{}",
            sanitize(symbol),
            sanitize(base_tf),
            std::process::id(),
            FEATURE_RUN_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        Ok(Self::Vortex {
            scratch_root,
            run_id_prefix,
            shards: Vec::with_capacity(timeframe_count),
        })
    }

    fn push(
        &mut self,
        timeframe: &str,
        timestamps: &[i64],
        columns: Vec<FeatureColumnF64>,
        control: &FeatureBuildControl,
    ) -> Result<()> {
        control.checkpoint()?;
        anyhow::ensure!(
            !columns.is_empty(),
            "cannot append an empty {timeframe} feature block"
        );
        match self {
            Self::InMemory { columns: all } => {
                all.extend(columns);
            }
            Self::Vortex {
                scratch_root,
                run_id_prefix,
                shards,
            } => {
                let shard_index = shards.len();
                let timeframe = timeframe
                    .chars()
                    .filter(|character| character.is_ascii_alphanumeric())
                    .take(16)
                    .collect::<String>();
                let run_id = format!("{run_id_prefix}-{shard_index}-{timeframe}");
                let feature_run = std::sync::Arc::new(
                    crate::core::feature_run_lease::FeatureRunLease::create(scratch_root, &run_id)?,
                );
                let store =
                    crate::core::vortex_feature_store::VortexFeatureStore::create_with_control(
                        feature_run,
                        timestamps,
                        &columns,
                        crate::core::vortex_feature_store::VortexFeatureStoreOptions::default(),
                        control,
                    )?;
                shards.push(store);
                // Release this column batch before the worker aligns its next
                // batch. A wave never retains its complete aligned disk output.
            }
        }
        Ok(())
    }
}

/// Build the multi-timeframe feature cube.
///
/// This used to take a fourth argument, `_cache: Option<&FeatureCache>`. The
/// underscore was accurate — nothing in the body read it — while four call
/// sites constructed a `FeatureCache` and passed it in. Both the parameter and
/// the type are gone; see the note at the top of `core/loader.rs` for why the
/// cache could not have been wired as designed, and what a correct one would
/// have to key on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MultiTimeframeFeatureMathAuthorityV3 {
    CurrentProcessPolicy,
    #[cfg(feature = "gpu-cuda")]
    ResidentGpuExactParityCpuReferenceV3,
}

pub fn prepare_multitimeframe_features_with_options(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_with_optional_cutoff(ds, base_tf, opts, None)
}

/// Build the raw producer representation explicitly, regardless of the
/// process-wide training normalization setting. Live consumers must then
/// apply the artifact's persisted fit; this API never fits on live rows.
pub fn prepare_multitimeframe_features_raw_with_options(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_raw_with_options_and_control(
        ds,
        base_tf,
        opts,
        &FeatureBuildControl::default(),
    )
}

/// Explicit raw replay with the caller's cancellation/progress control.
pub fn prepare_multitimeframe_features_raw_with_options_and_control(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    control: &FeatureBuildControl,
) -> Result<FeatureFrame> {
    control.checkpoint()?;
    // Preserve the original recipe, including historical fit flags that were
    // inert when raw search was admitted. The explicit false below prevents
    // both fitting and support projection even if ambient training is enabled.
    prepare_multitimeframe_features_with_feature_math_authority_v3(
        ds,
        base_tf,
        opts,
        None,
        MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy,
        None,
        control,
        Some(false),
        None,
    )
}

/// Replay the exact search training fit on freshly computed direct-source
/// columns. Historical fit coordinates are validated against their recorded
/// training scope, never against this possibly much shorter live snapshot.
/// The canonical producer, causal alignment, bounded sink and cancellation
/// path are shared with training; only fitting is replaced by transformation.
pub fn prepare_multitimeframe_features_with_fitted_normalization(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    state: &SearchNormalizationFittedStateV1,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_with_fitted_normalization_and_control(
        ds,
        base_tf,
        opts,
        state,
        &FeatureBuildControl::default(),
    )
}

/// Frozen fitted replay with cancellation carried through the shared producer.
pub fn prepare_multitimeframe_features_with_fitted_normalization_and_control(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    state: &SearchNormalizationFittedStateV1,
    control: &FeatureBuildControl,
) -> Result<FeatureFrame> {
    control.checkpoint()?;
    state.validate()?;
    if let Some(recorded_rows) = &opts.normalization_training_rows {
        anyhow::ensure!(
            *recorded_rows == state.training_rows()?,
            "fitted normalization replay recipe declares a different training scope"
        );
    }
    prepare_multitimeframe_features_with_feature_math_authority_v3(
        ds,
        base_tf,
        opts,
        None,
        MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy,
        None,
        control,
        Some(true),
        Some(state),
    )
}

/// The UI job's Stop flag follows the actual producer/writer workers, never a
/// process-global flag. The ordinary API remains the uncontrolled math oracle.
pub fn prepare_multitimeframe_features_with_control(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    control: &FeatureBuildControl,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_with_feature_math_authority_v3(
        ds,
        base_tf,
        opts,
        None,
        MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy,
        None,
        control,
        None,
        None,
    )
}

/// Build the canonical multi-timeframe CPU exact-parity feature cube with the
/// resident GPU V3 Quant math and ordered Classic subset.
///
/// This is an explicit contract-authoring boundary, not an adaptive fallback:
/// it retains every non-Classic production family and all source/provenance
/// checks while replacing only the ordinary complete Classic graph with the
/// versioned exact-parity subset. The process feature authority must already
/// resolve to CPU/Auto; the underlying planner refuses a GpuOnly process so
/// CPU bits can never be sealed as CUDA output.
#[cfg(feature = "gpu-cuda")]
pub fn prepare_multitimeframe_features_gpu_exact_parity_cpu_reference_v3(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
) -> Result<FeatureFrame> {
    anyhow::ensure!(
        matches!(
            opts.profile,
            FeatureProfile::Standard | FeatureProfile::HPC | FeatureProfile::Adaptive
        ),
        "the resident GPU V3 exact-parity CPU reference supports only Standard, HPC, or \
         Adaptive; Full must retain the complete fail-closed Classic graph"
    );
    prepare_multitimeframe_features_with_feature_math_authority_v3(
        ds,
        base_tf,
        opts,
        None,
        MultiTimeframeFeatureMathAuthorityV3::ResidentGpuExactParityCpuReferenceV3,
        None,
        &FeatureBuildControl::default(),
        None,
        None,
    )
}

/// Build the multi-timeframe feature cube from the independently downloaded
/// direct rows strictly before one shared half-open timestamp cutoff.
///
/// The cutoff is applied to each canonical timeframe frame before feature
/// computation. The returned feature provenance therefore keeps every full
/// immutable generation identity/hash/lease while binding only the exact rows
/// consumed from that generation. No timeframe is derived from another one.
pub fn prepare_multitimeframe_features_before_with_options(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    end_exclusive_ms: i64,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_with_optional_cutoff(ds, base_tf, opts, Some(end_exclusive_ms))
}

/// Build one exact pre-cutoff multi-timeframe streaming batch without ambient
/// process state. The immutable working set is captured into the run-wide
/// Classic plan before any per-timeframe producer or Rayon worker starts.
pub fn prepare_multitimeframe_features_before_batch_with_options(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    end_exclusive_ms: i64,
    batch: Option<std::sync::Arc<crate::core::hpc_ta::SweepBatch>>,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_with_feature_math_authority_v3(
        ds,
        base_tf,
        opts,
        Some(end_exclusive_ms),
        MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy,
        batch,
        &FeatureBuildControl::default(),
        None,
        None,
    )
}

fn prepare_multitimeframe_features_with_optional_cutoff(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    end_exclusive_ms: Option<i64>,
) -> Result<FeatureFrame> {
    prepare_multitimeframe_features_with_feature_math_authority_v3(
        ds,
        base_tf,
        opts,
        end_exclusive_ms,
        MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy,
        None,
        &FeatureBuildControl::default(),
        None,
        None,
    )
}

fn resolve_feature_working_set(
    opts: &FeatureBuildOptions,
    requested: Option<std::sync::Arc<crate::core::hpc_ta::SweepBatch>>,
) -> Result<Option<std::sync::Arc<crate::core::hpc_ta::SweepBatch>>> {
    if let Some(batch) = &requested {
        batch.validate()?;
    }
    let Some(recorded) = &opts.classic_ta_working_set else {
        return Ok(requested);
    };
    recorded.validate()?;
    anyhow::ensure!(
        recorded.replace_base_vocabulary,
        "recorded Classic working set must specify the complete base selection"
    );
    if let Some(batch) = &requested {
        anyhow::ensure!(
            batch.as_ref() == recorded,
            "requested Classic working set conflicts with the recorded exact selection"
        );
    }
    Ok(Some(std::sync::Arc::new(recorded.clone())))
}

fn prepare_multitimeframe_features_with_feature_math_authority_v3(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    end_exclusive_ms: Option<i64>,
    feature_math_authority: MultiTimeframeFeatureMathAuthorityV3,
    working_set: Option<std::sync::Arc<crate::core::hpc_ta::SweepBatch>>,
    control: &FeatureBuildControl,
    normalization_override: Option<bool>,
    fitted_normalization: Option<&SearchNormalizationFittedStateV1>,
) -> Result<FeatureFrame> {
    control.checkpoint()?;
    let working_set = resolve_feature_working_set(opts, working_set)?;
    let base_timeframe = base_tf
        .parse::<CanonicalTimeframe>()
        .map_err(|error| anyhow::anyhow!(error))?;
    let selected_identity = ds
        .source_artifacts
        .get(base_tf)
        .with_context(|| format!("base timeframe {base_tf} has no direct canonical artifact"))?
        .identity()
        .clone();
    let mut required = vec![base_timeframe];
    let mut active_higher = Vec::new();
    for timeframe in &opts.higher_tfs {
        let parsed = timeframe
            .parse::<CanonicalTimeframe>()
            .map_err(|error| anyhow::anyhow!(error))?;
        if parsed == base_timeframe {
            continue;
        }
        anyhow::ensure!(
            !required.contains(&parsed),
            "duplicate requested direct timeframe {parsed}"
        );
        required.push(parsed);
        active_higher.push(parsed.as_str().to_owned());
    }
    require_direct_timeframes(ds, &selected_identity, &required)?;
    let mut direct_sources = std::collections::HashMap::with_capacity(required.len());
    for timeframe in &required {
        control.checkpoint()?;
        let source = match end_exclusive_ms {
            Some(cutoff) => ds.canonical_frame_before_timestamp_ms(timeframe.as_str(), cutoff).with_context(|| {
                format!(
                    "clipping direct canonical timeframe {timeframe} before the shared half-open cutoff {cutoff} ms"
                )
            })?,
            None => ds.canonical_frame(timeframe.as_str())?,
        };
        direct_sources.insert(timeframe.as_str().to_owned(), source);
    }
    let budget_rows = direct_sources
        .values()
        .map(CanonicalOhlcvFrame::len)
        .max()
        .context("direct timeframe request is empty")?;

    // Capture one machine/admission decision for the complete direct-TF cube.
    // In GpuOnly this also resolves the entire admitted graph before any
    // producer starts, so a missing CUDA route cannot leave partial CPU/SMC
    // feature allocations behind. Every frame below borrows this exact plan;
    // falling available RAM during the build cannot narrow later timeframes.
    let classic_run_plan = match feature_math_authority {
        MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy => {
            crate::core::hpc_ta::prepare_classic_ta_run_plan_with_working_set(
                budget_rows,
                control.resolved_indicator_compute_policy(),
                working_set,
            )?
        }
        #[cfg(feature = "gpu-cuda")]
        MultiTimeframeFeatureMathAuthorityV3::ResidentGpuExactParityCpuReferenceV3 => {
            anyhow::ensure!(
                working_set.is_none(),
                "resident GPU exact-parity CPU reference does not accept a streaming working set"
            );
            crate::core::hpc_ta::prepare_classic_ta_gpu_exact_parity_cpu_reference_run_plan_v3(
                budget_rows,
            )?
        }
    };

    let base_source = direct_sources
        .remove(base_tf)
        .context("validated base direct timeframe disappeared")?;
    let base_ns = base_source
        .ohlcv()
        .timestamp
        .as_ref()
        .context("base has no timestamps")?
        .clone();
    let n_samples = base_ns.len();
    let normalize = normalization_override.unwrap_or_else(feature_normalization_enabled);
    if normalize && fitted_normalization.is_none() {
        anyhow::ensure!(
            opts.normalization_training_rows.is_some(),
            "normalization is enabled but no explicit in-sample training row range was supplied"
        );
    }
    let normalization_training_rows = match fitted_normalization {
        Some(state) => Some(state.training_rows()?),
        None => opts
            .normalization_training_rows
            .clone()
            .filter(|_| normalize),
    };
    if let Some(training_rows) = &normalization_training_rows {
        anyhow::ensure!(
            training_rows.start < training_rows.end
                && (fitted_normalization.is_some() || training_rows.end <= n_samples),
            "normalization training rows {:?} are outside 0..{n_samples}",
            training_rows
        );
    }

    let base_control = control.for_timeframe(base_tf);
    let base_frame = compute_hpc_feature_frame_sized_with_classic_plan(
        &base_source,
        opts.profile,
        budget_rows,
        &classic_run_plan,
        feature_math_authority,
        &base_control,
    )?;
    let original_names = base_frame.names.clone();
    let feature_columns_per_timeframe = original_names.len();
    let mut base_columns = take_in_memory_columns(base_frame)?;
    if opts.prefix_base_features {
        for column in &mut base_columns {
            column.name = format!("{}_{}", base_tf, column.name);
        }
    }
    let estimated_features = base_columns.len().saturating_mul(1 + active_higher.len());
    let cube_bytes = (n_samples as u64)
        .saturating_mul(estimated_features as u64)
        .saturating_mul(
            (std::mem::size_of::<f64>() + std::mem::size_of::<FeatureCellValidity>()) as u64,
        );
    // The sink decision is deliberately made while only the base block is
    // live. The disk path below persists and releases it before any higher-TF
    // alignment can allocate another full base-grid block.
    let in_ram = should_build_cube_in_ram(cube_bytes);
    tracing::info!(
        target: "neoethos_data::prepare_multitimeframe_features",
        symbol = %ds.symbol,
        base_tf = base_tf,
        rows = n_samples,
        features = estimated_features,
        timeframes = 1 + active_higher.len(),
        est_cube_gb = format!("{:.2}", cube_bytes as f64 / 1e9),
        available_ram_gb = format!("{:.1}", neoethos_core::available_memory_bytes() as f64 / 1e9),
        sink = if in_ram { "RAM (f64 + validity)" } else { "Vortex scratch shards" },
        "feature cube build plan"
    );
    let mut sink =
        MultiTimeframeFeatureSink::new(in_ram, &ds.symbol, base_tf, 1 + active_higher.len())?;
    let mut contract_blocks = Vec::with_capacity(1 + active_higher.len());
    let mut retained_columns = 0usize;

    let normalization_workers = if normalize {
        // The base columns already exist: charge only new scratch against the
        // current free-memory reading, keeping the same runtime/system reserve.
        admit_normalization_workers(
            n_samples,
            base_columns.len(),
            higher_timeframe_parallel_budget_bytes(
                false,
                neoethos_core::allocation_headroom_bytes(),
                0,
                0,
            ),
            rayon::current_num_threads(),
        )?
    } else {
        0
    };
    tracing::info!(
        target: "neoethos_data::prepare_multitimeframe_features",
        timeframe = base_tf,
        normalization_workers,
        normalization_scratch_bytes = normalization_scratch_bytes(n_samples)
            .saturating_mul(normalization_workers as u64),
        "base normalization concurrency admitted against current RAM"
    );
    let (mut normalization_fits, mut dropped) = prepare_multitimeframe_feature_columns(
        &mut base_columns,
        normalize,
        normalization_training_rows.clone(),
        opts.drop_columns_without_normalization_training_support,
        normalization_workers,
        &base_control,
        fitted_normalization,
    )?;
    if !base_columns.is_empty() {
        retained_columns = base_columns.len();
        let projected_names = base_columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        sink.push(base_tf, &base_ns, base_columns, &base_control)?;
        contract_blocks.push(MultiTimeframeFeatureContractBlock {
            timeframe: base_tf.to_owned(),
            source: base_source,
            original_names,
            projected_names,
            higher_timeframe: false,
            availability_rule: "base_bar_open_v1".to_owned(),
            availability_lag_ms: Some(0),
            max_age_ms: None,
        });
    }

    let mut higher_tasks = Vec::with_capacity(active_higher.len());
    for higher_tf in &active_higher {
        let source = direct_sources
            .remove(higher_tf)
            .with_context(|| format!("required direct timeframe {higher_tf} disappeared"))?;
        higher_tasks.push(HigherTimeframeBuildTask {
            timeframe: higher_tf.clone(),
            estimated_peak_bytes: estimate_higher_timeframe_task_peak_bytes(
                source.len(),
                n_samples,
                feature_columns_per_timeframe,
                normalize,
                rayon::current_num_threads(),
            ),
            source,
        });
    }
    let mut pending_tasks = std::collections::VecDeque::from(higher_tasks);
    let higher_started = Instant::now();
    let mut wave_index = 0usize;
    while !pending_tasks.is_empty() {
        control.checkpoint()?;
        // The probe is a snapshot, not a process-wide reservation. Refresh it
        // after every completed wave, including bytes already retained by RAM.
        let mut available_for_higher = higher_timeframe_available_memory_bytes();
        let mut parallel_memory_budget = higher_timeframe_parallel_budget_bytes(
            sink.is_in_memory(),
            available_for_higher,
            cube_bytes,
            sink.resident_column_bytes(),
        );
        if sink.is_in_memory()
            && pending_tasks
                .iter()
                .any(|task| task.estimated_peak_bytes > parallel_memory_budget)
        {
            tracing::info!(
                target: "neoethos_data::prepare_multitimeframe_features",
                available_bytes = available_for_higher,
                parallel_memory_budget,
                "higher-timeframe scratch no longer fits beside the RAM cube; spilling retained columns to Vortex without changing the feature plan"
            );
            sink.spill_to_vortex(&ds.symbol, base_tf, &base_ns, available_for_higher, control)?;
            available_for_higher = higher_timeframe_available_memory_bytes();
            parallel_memory_budget =
                higher_timeframe_parallel_budget_bytes(false, available_for_higher, cube_bytes, 0);
        }
        let source_rows = pending_tasks
            .iter()
            .map(|task| task.source.len())
            .collect::<Vec<_>>();
        let waves = plan_higher_timeframe_build_waves(
            &source_rows,
            n_samples,
            feature_columns_per_timeframe,
            normalize,
            parallel_memory_budget,
            !sink.is_in_memory(),
            rayon::current_num_threads(),
        )?;
        let remaining_waves = waves.len();
        let wave = waves
            .into_iter()
            .next()
            .context("nonempty higher-timeframe request has no admitted wave")?;
        let wave_len = wave.task_range.len();
        let wave_tasks = pending_tasks.drain(..wave_len).collect::<Vec<_>>();
        let timeframes = wave_tasks
            .iter()
            .map(|task| task.timeframe.as_str())
            .collect::<Vec<_>>()
            .join(",");
        tracing::info!(
            target: "neoethos_data::prepare_multitimeframe_features",
            wave = wave_index + 1,
            remaining_waves,
            tasks = wave_len,
            worker_threads = rayon::current_num_threads(),
            timeframe = %timeframes,
            estimated_peak_gb = format!("{:.2}", wave.estimated_peak_bytes as f64 / 1e9),
            parallel_memory_budget_gb = format!("{:.2}", parallel_memory_budget as f64 / 1e9),
            available_ram_gb = format!("{:.2}", available_for_higher as f64 / 1e9),
            sink = if sink.is_in_memory() { "RAM" } else { "Vortex" },
            "higher-timeframe parallel wave admitted before allocation"
        );
        let wave_sink = &sink;
        let prepared_wave = wave_tasks
            .into_par_iter()
            .zip(wave.task_memory.into_par_iter())
            .map(|(task, memory)| -> Result<CompletedHigherTimeframeFeatureBlock> {
                let task_control = control.for_timeframe(&task.timeframe);
                task_control.checkpoint()?;
                let task_started = Instant::now();
                tracing::info!(
                    target: "neoethos_data::prepare_multitimeframe_features",
                    higher_tf = %task.timeframe,
                    source_rows = task.source.len(),
                    estimated_peak_gb = format!("{:.2}", memory.estimated_peak_bytes as f64 / 1e9),
                    "higher-timeframe feature task started"
                );
                let mut task_sink = wave_sink.empty_peer();
                let mut task_normalization_fits = Vec::new();
                let mut task_dropped = Vec::new();
                let contract = compute_aligned_higher_block(
                    task.source,
                    base_tf,
                    &base_ns,
                    &task.timeframe,
                    opts.profile,
                    budget_rows,
                    &classic_run_plan,
                    feature_math_authority,
                    memory.estimated_peak_bytes,
                    memory.normalization_workers.max(1),
                    normalize,
                    !wave_sink.is_in_memory(),
                    &task_control,
                    |mut columns, normalization_workers| {
                        let (fits, removed) = prepare_multitimeframe_feature_columns(
                            &mut columns,
                            normalize,
                            normalization_training_rows.clone(),
                            opts.drop_columns_without_normalization_training_support,
                            normalization_workers,
                            &task_control,
                            fitted_normalization,
                        )?;
                        task_normalization_fits.extend(fits);
                        task_dropped.extend(removed);
                        let names = columns.iter().map(|column| column.name.clone()).collect();
                        if !columns.is_empty() {
                            task_sink.push(&task.timeframe, &base_ns, columns, &task_control)?;
                        }
                        Ok(names)
                    },
                )?;
                tracing::info!(
                    target: "neoethos_data::prepare_multitimeframe_features",
                    higher_tf = %task.timeframe,
                    output_columns = contract.as_ref().map_or(0, |block| block.projected_names.len()),
                    elapsed_ms = task_started.elapsed().as_millis(),
                    "higher-timeframe feature task completed"
                );
                Ok(CompletedHigherTimeframeFeatureBlock {
                    contract,
                    sink: task_sink,
                    normalization_fits: task_normalization_fits,
                    dropped: task_dropped,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for prepared in prepared_wave {
            dropped.extend(prepared.dropped);
            normalization_fits.extend(prepared.normalization_fits);
            if let Some(contract) = prepared.contract {
                retained_columns = retained_columns.saturating_add(contract.projected_names.len());
                sink.append(prepared.sink)?;
                contract_blocks.push(contract);
            }
        }
        tracing::info!(
            target: "neoethos_data::prepare_multitimeframe_features",
            wave = wave_index + 1,
            elapsed_ms = higher_started.elapsed().as_millis(),
            "higher-timeframe parallel wave completed and released to the selected sink"
        );
        wave_index += 1;
    }
    anyhow::ensure!(
        pending_tasks.is_empty(),
        "higher-timeframe wave planner left {} tasks unexecuted",
        pending_tasks.len()
    );

    if let Some(training_rows) = &normalization_training_rows {
        anyhow::ensure!(
            !contract_blocks.is_empty(),
            "normalization training range {:?} supports no feature columns",
            training_rows
        );
        if !dropped.is_empty() {
            tracing::info!(
                target: "neoethos_data::prepare_multitimeframe_features",
                dropped_columns = dropped.len(),
                retained_columns,
                training_start = training_rows.start,
                training_end = training_rows.end,
                "projected columns without normalization support in the exact training range"
            );
        }
    }
    control.report("feature_contract", "sealing complete timeframe set", 0, 1)?;
    let (plan, provenance, source_leases) = build_multitimeframe_feature_contract(
        &contract_blocks,
        feature_math_authority,
        normalize.then_some(normalization_fits.as_slice()),
    )?;
    let normalization_fitted_state = normalize
        .then(|| {
            SearchNormalizationFittedStateV1::new(plan.final_outputs().to_vec(), normalization_fits)
        })
        .transpose()?;
    if let Some(expected) = fitted_normalization {
        anyhow::ensure!(
            normalization_fitted_state.as_ref() == Some(expected),
            "current feature producers cannot reproduce the complete persisted normalization schema/order; no fit was recomputed"
        );
    }
    control.checkpoint()?;
    let frame = match sink {
        MultiTimeframeFeatureSink::InMemory { columns } => {
            FeatureFrame::from_canonical_columns(base_ns, columns, plan, provenance, source_leases)
        }
        MultiTimeframeFeatureSink::Vortex { shards, .. } => {
            let stores = std::sync::Arc::new(
                crate::core::vortex_feature_store::VortexFeatureStoreSet::new(shards)?,
            );
            FeatureFrame::from_canonical_vortex_set(
                base_ns,
                stores,
                plan,
                provenance,
                source_leases,
            )
        }
    }?;
    let mut recorded_options = opts.clone();
    recorded_options.classic_ta_working_set = classic_run_plan.recorded_working_set();
    let frame = frame.with_feature_build_options(recorded_options);
    match normalization_fitted_state {
        Some(state) => frame.with_normalization_fitted_state(state),
        None => Ok(frame),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_feature_test_report() -> Result<crate::core::hpc_ta::ClassicTaExecutionReport> {
        let plan = crate::core::hpc_ta::prepare_classic_ta_run_plan(
            1_000_000,
            crate::core::hpc_ta::IndicatorComputePolicy::CpuOnly,
        )?;
        let mut report = plan.admission_report();
        report.admitted_base_columns = 3;
        report.historical_sweep_reserved_columns = 5;
        report.extended_planned_columns = 7;
        // Deliberately unrelated: admission must use planned output, not a
        // completed execution count or the vocabulary's nominal width ceiling.
        report.produced_columns = 999;
        report.max_columns = 1;
        Ok(report)
    }

    #[test]
    fn source_feature_output_admission_counts_all_sweeps_at_actual_tf_rows() -> Result<()> {
        let report = source_feature_test_report()?;
        let before = report.clone();
        assert_eq!(std::mem::size_of::<FeatureCellValidity>(), 1);
        // 15 columns * 100 actual source rows * (8-byte f64 + 1 validity).
        assert_eq!(
            require_source_feature_output_headroom(100, &report, 13_500)?,
            13_500
        );
        assert!(require_source_feature_output_headroom(100, &report, 13_499).is_err());
        assert_eq!(
            require_source_feature_output_headroom(20, &report, 13_500)?,
            2_700
        );
        assert_eq!(
            report, before,
            "admission must never narrow the frozen plan"
        );
        Ok(())
    }

    #[test]
    fn source_feature_output_admission_rejects_zero_and_unaffordable_floor() -> Result<()> {
        let mut report = source_feature_test_report()?;
        assert!(require_source_feature_output_headroom(100, &report, 0).is_err());
        report.admitted_base_columns = 0;
        report.historical_sweep_reserved_columns = crate::core::feature_budget::MIN_COLUMNS;
        report.extended_planned_columns = 0;
        // Planning still preserves the historical 66-column vocabulary. Its
        // output alone cannot fit into this reading, before any scratch exists.
        let error = require_source_feature_output_headroom(1_000_000, &report, 100_000_000)
            .expect_err("the planning floor cannot overrule live headroom");
        assert!(error.to_string().contains("594000000 bytes"));
        assert!(error.to_string().contains("before computation"));
        Ok(())
    }

    #[test]
    fn source_feature_output_admission_rejects_overflow_and_outgrown_plan() -> Result<()> {
        let mut report = source_feature_test_report()?;
        assert!(require_source_feature_output_headroom(1_000_001, &report, u64::MAX).is_err());
        report.admitted_base_columns = usize::MAX;
        assert!(require_source_feature_output_headroom(1, &report, u64::MAX).is_err());
        report.admitted_base_columns = usize::MAX;
        report.historical_sweep_reserved_columns = 0;
        report.extended_planned_columns = 0;
        report.budget_rows = usize::MAX;
        assert!(require_source_feature_output_headroom(usize::MAX, &report, u64::MAX).is_err());
        Ok(())
    }

    #[test]
    fn normalization_support_projection_never_looks_into_the_holdout_suffix() -> Result<()> {
        use crate::core::features::{FeatureCellValidity, FeatureColumnF64};

        let late_only = FeatureColumnF64::new(
            "late_only",
            vec![f64::NAN, f64::NAN, 7.0, 8.0],
            vec![
                FeatureCellValidity::Warmup,
                FeatureCellValidity::Warmup,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
            ],
        )?;
        let in_sample = FeatureColumnF64::new(
            "in_sample",
            vec![f64::NAN, 2.0, 3.0, 4.0],
            vec![
                FeatureCellValidity::Warmup,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
            ],
        )?;

        let mut columns = vec![late_only, in_sample];
        let dropped = retain_columns_with_normalization_training_support(&mut columns, 0..2)?;

        assert_eq!(dropped, vec!["late_only"]);
        assert_eq!(
            columns
                .iter()
                .map(|column| column.name.as_str())
                .collect::<Vec<_>>(),
            vec!["in_sample"]
        );
        Ok(())
    }

    #[test]
    fn higher_timeframe_wave_plan_is_memory_bounded_ordered_and_complete() -> Result<()> {
        // Dimensions observed in the actual M5 run, not allocated test data.
        let rows = [264_543, 132_282, 66_394, 16_647, 5_570, 2_800, 554, 127];
        let base_rows = 791_263;
        let columns = 1_000;
        let budget = 12_725_157_136;
        let greedy =
            plan_higher_timeframe_task_memory(rows[0], base_rows, columns, true, budget, true, 10)?;
        assert_eq!(greedy.alignment_batch_columns, 478);
        assert_eq!(greedy.normalization_workers, 10);
        assert_eq!(greedy.estimated_peak_bytes, budget);
        // Equal division cannot even admit the larger complete source. The
        // joint planner must preserve its unequal minimum, not fake a peak.
        assert!(
            plan_higher_timeframe_task_memory(
                rows[0],
                base_rows,
                columns,
                true,
                budget / 2,
                true,
                1,
            )
            .is_err()
        );
        let waves =
            plan_higher_timeframe_build_waves(&rows, base_rows, columns, true, budget, true, 10)?;
        assert_eq!(waves[0].task_range, 0..2);
        for wave in &waves {
            assert_eq!(wave.task_memory.len(), wave.task_range.len());
            assert_eq!(
                wave.estimated_peak_bytes,
                wave.task_memory
                    .iter()
                    .map(|memory| memory.estimated_peak_bytes)
                    .sum::<u64>()
            );
            assert!(wave.estimated_peak_bytes <= budget);
            assert!(
                wave.task_memory
                    .iter()
                    .map(|memory| memory.normalization_workers)
                    .sum::<usize>()
                    <= 10
            );
            for (index, memory) in wave.task_range.clone().zip(&wave.task_memory) {
                if wave.task_range.len() > 1 {
                    assert!(memory.alignment_batch_columns >= columns.div_ceil(10));
                    assert!(columns.div_ceil(memory.alignment_batch_columns) <= 10);
                }
                assert_eq!(
                    *memory,
                    plan_higher_timeframe_task_memory(
                        rows[index],
                        base_rows,
                        columns,
                        true,
                        memory.estimated_peak_bytes,
                        true,
                        memory.normalization_workers,
                    )?,
                    "actual-width recheck must retain the admitted CPU and RAM bounds"
                );
                let covered = (0..columns).collect::<Vec<_>>();
                assert_eq!(
                    covered
                        .chunks(memory.alignment_batch_columns)
                        .flatten()
                        .copied()
                        .collect::<Vec<_>>(),
                    covered
                );
            }
        }
        let visited = waves
            .iter()
            .flat_map(|wave| wave.task_range.clone())
            .collect::<Vec<_>>();
        assert_eq!(visited, (0..rows.len()).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn higher_timeframe_wave_plan_refuses_oversized_and_zero_headroom_tasks() -> Result<()> {
        let rows = [264_543, 132_282, 66_394];
        let base = 791_263;
        let columns = 1_000;
        let per_column = base as u64 * HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES;
        let first = higher_timeframe_fixed_work_bytes(rows[0], base, columns, true) + per_column;
        let second = higher_timeframe_fixed_work_bytes(rows[1], base, columns, true) + per_column;
        assert_eq!(first, 6_458_354_176);
        let waves = plan_higher_timeframe_build_waves(
            &rows,
            base,
            columns,
            true,
            first + second - 1,
            true,
            10,
        )?;
        assert_eq!(
            waves[0].task_range,
            0..1,
            "a second source minimum does not fit"
        );
        let narrow =
            plan_higher_timeframe_build_waves(&rows, base, columns, true, first, true, 10)?;
        assert_eq!(
            narrow[0].task_memory[0].alignment_batch_columns, 1,
            "granularity is a preference, not a new low-memory refusal"
        );
        let serial =
            plan_higher_timeframe_build_waves(&rows, base, columns, true, 12_725_157_136, true, 1)?;
        assert!(serial.iter().all(|wave| wave.task_range.len() == 1));
        let error =
            plan_higher_timeframe_build_waves(&rows, base, columns, true, first - 1, true, 10)
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("oversized task must be streamed or refused")
        );
        assert!(
            plan_higher_timeframe_build_waves(&rows, base, columns, true, 0, true, 10).is_err()
        );
        assert!(
            plan_higher_timeframe_build_waves(&rows, base, columns, true, u64::MAX, true, 0)
                .is_err()
        );
        assert!(plan_higher_timeframe_build_waves(&[], 0, 0, true, 0, true, 0)?.is_empty());
        // The RAM sink must reserve every aligned column, not silently use the
        // Vortex one-column minimum while retaining the complete output.
        assert!(
            plan_higher_timeframe_build_waves(
                &rows,
                base,
                columns,
                true,
                12_725_157_136,
                false,
                10,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn higher_timeframe_wave_plan_does_not_saturate_an_overflowing_sum() -> Result<()> {
        assert!(
            plan_higher_timeframe_build_waves(
                &[usize::MAX],
                usize::MAX,
                usize::MAX,
                true,
                u64::MAX,
                true,
                2,
            )
            .is_err()
        );
        for (rows, base, columns) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert!(
                plan_higher_timeframe_build_waves(&[rows], base, columns, true, u64::MAX, true, 2,)
                    .is_err()
            );
        }
        #[cfg(target_pointer_width = "64")]
        {
            // Both individual fixed estimates fit, but their sum overflows.
            let rows = usize::MAX / 80;
            let waves =
                plan_higher_timeframe_build_waves(&[rows, rows], 1, 1, false, u64::MAX, true, 2)?;
            assert_eq!(waves.len(), 2);
            assert_eq!(waves[0].task_range, 0..1);
            assert_eq!(waves[1].task_range, 1..2);
        }
        Ok(())
    }

    #[test]
    fn higher_timeframe_alignment_batches_keep_every_column_within_the_budget() -> Result<()> {
        for normalize in [false, true] {
            for (source_rows, base_rows) in [(10, 100), (100, 10_000), (10_000, 100)] {
                let width = 11;
                let fixed =
                    higher_timeframe_fixed_work_bytes(source_rows, base_rows, width, normalize);
                let per_column = base_rows as u64 * HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES;
                let budget = fixed + 3 * per_column;
                let memory = plan_higher_timeframe_task_memory(
                    source_rows,
                    base_rows,
                    width,
                    normalize,
                    budget,
                    true,
                    1,
                )?;
                assert_eq!(memory.alignment_batch_columns, 3);
                assert_eq!(memory.estimated_peak_bytes, budget);
                let ids = (0..width).collect::<Vec<_>>();
                let batches = ids
                    .chunks(memory.alignment_batch_columns)
                    .collect::<Vec<_>>();
                assert_eq!(batches.concat(), ids);
                assert!(
                    batches
                        .iter()
                        .all(|batch| fixed + batch.len() as u64 * per_column <= budget)
                );
                assert!(
                    plan_higher_timeframe_task_memory(
                        source_rows,
                        base_rows,
                        width,
                        normalize,
                        budget,
                        false,
                        1,
                    )
                    .is_err()
                );
                assert!(
                    plan_higher_timeframe_task_memory(
                        source_rows,
                        base_rows,
                        width,
                        normalize,
                        fixed + per_column - 1,
                        true,
                        1,
                    )
                    .is_err()
                );
                let full = estimate_higher_timeframe_task_peak_bytes(
                    source_rows,
                    base_rows,
                    width,
                    normalize,
                    1,
                );
                assert_eq!(
                    plan_higher_timeframe_task_memory(
                        source_rows,
                        base_rows,
                        width,
                        normalize,
                        full,
                        false,
                        1,
                    )?
                    .alignment_batch_columns,
                    width
                );
            }
        }
        assert!(
            plan_higher_timeframe_task_memory(
                usize::MAX,
                usize::MAX,
                usize::MAX,
                true,
                u64::MAX,
                true,
                1,
            )
            .is_err()
        );
        assert!(plan_higher_timeframe_task_memory(1, 1, 1, false, 0, true, 1).is_err());
        Ok(())
    }

    #[test]
    fn higher_timeframe_column_batches_preserve_normalization_fits_and_projection() -> Result<()> {
        use FeatureCellValidity::{Degenerate, Valid, Warmup};
        let columns = vec![
            FeatureColumnF64::new("varying", vec![1.0, 3.0, 5.0, 7.0, 100.0], vec![Valid; 5])?,
            FeatureColumnF64::new(
                "late",
                vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 9.0],
                vec![Warmup, Warmup, Warmup, Warmup, Valid],
            )?,
            FeatureColumnF64::new("constant", vec![2.0; 5], vec![Valid; 5])?,
        ];
        let mut whole = columns.clone();
        let (whole_fits, whole_dropped) = prepare_multitimeframe_feature_columns(
            &mut whole,
            true,
            Some(0..4),
            true,
            2,
            &FeatureBuildControl::default(),
            None,
        )?;
        let mut batched = Vec::new();
        let mut fits = Vec::new();
        let mut dropped = Vec::new();
        for column in columns {
            let mut batch = vec![column];
            let (fit, removed) = prepare_multitimeframe_feature_columns(
                &mut batch,
                true,
                Some(0..4),
                true,
                1,
                &FeatureBuildControl::default(),
                None,
            )?;
            fits.extend(fit);
            dropped.extend(removed);
            batched.extend(batch);
        }
        assert_eq!(fits, whole_fits);
        assert_eq!(dropped, whole_dropped);
        assert_eq!(dropped, ["late"]);
        assert_eq!(fits[0].median, 4.0);
        assert_eq!(fits[0].scale, 2.0 * 1.4826);
        assert_eq!(
            batched[0].values[0].to_bits(),
            (-3.0_f64 / (2.0 * 1.4826)).to_bits()
        );
        assert_eq!(batched[0].values[4], 10.0);
        assert!(batched[1].validity.iter().all(|state| *state == Degenerate));
        for (actual, expected) in batched.iter().zip(&whole) {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.validity, expected.validity);
            assert_eq!(
                actual
                    .values
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[test]
    fn normalization_parallel_workers_are_charged_before_alignment() -> Result<()> {
        let rows = 100;
        let width = 12;
        let fixed = higher_timeframe_fixed_work_bytes(20, rows, width, true);
        let aligned = rows as u64 * HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES;
        let scratch = normalization_scratch_bytes(rows);
        for workers in 1..=4 {
            // One live aligned column per simultaneous normalizer, plus the
            // one normalizer already included in the fixed minimum.
            let budget = fixed + (workers - 1) * scratch + workers * aligned;
            let plan = plan_higher_timeframe_task_memory(20, rows, width, true, budget, true, 4)?;
            assert_eq!(plan.normalization_workers, workers as usize);
            assert!(plan.alignment_batch_columns >= workers as usize);
            assert!(plan.estimated_peak_bytes <= budget);
            assert_eq!(
                plan.estimated_peak_bytes,
                fixed + (workers - 1) * scratch + plan.alignment_batch_columns as u64 * aligned
            );
        }
        let full_with_one = fixed + width as u64 * aligned;
        let ram =
            plan_higher_timeframe_task_memory(20, rows, width, true, full_with_one, false, 4)?;
        assert_eq!(ram.alignment_batch_columns, width);
        assert_eq!(ram.normalization_workers, 1);
        assert!(
            plan_higher_timeframe_task_memory(20, rows, width, true, full_with_one, false, 0)
                .is_err()
        );
        assert_eq!(
            admit_normalization_workers(rows, width, 3 * scratch, 10)?,
            3
        );
        assert_eq!(admit_normalization_workers(rows, 2, 3 * scratch, 10)?, 2);
        assert_eq!(admit_normalization_workers(rows, width, 3 * scratch, 1)?, 1);
        assert!(admit_normalization_workers(rows, width, scratch - 1, 10).is_err());
        assert!(admit_normalization_workers(0, width, scratch, 10).is_err());
        assert_eq!(admit_normalization_workers(rows, 0, 0, 0)?, 0);
        Ok(())
    }

    #[test]
    fn parallel_search_normalization_preserves_canonical_smc_states() -> Result<()> {
        let mut columns = vec![
            FeatureColumnF64::new(
                "smc_ob",
                vec![1.0, 1.0, 1.0, -1.0],
                vec![FeatureCellValidity::Valid; 4],
            )?,
            FeatureColumnF64::new(
                "H1_smc_eqh",
                vec![0.0, 0.0, 0.0, 1.0],
                vec![FeatureCellValidity::Valid; 4],
            )?,
            FeatureColumnF64::new(
                "smc_trend_bias",
                vec![2.0, 3.0, 4.0, -0.1],
                vec![FeatureCellValidity::Valid; 4],
            )?,
        ];
        let expected = columns.clone();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build()?;
        let (fits, dropped) = pool.install(|| {
            prepare_multitimeframe_feature_columns(
                &mut columns,
                true,
                Some(0..3),
                false,
                3,
                &FeatureBuildControl::default(),
                None,
            )
        })?;
        assert!(dropped.is_empty());
        assert_eq!(fits.len(), expected.len());
        for ((actual, expected), fit) in columns.iter().zip(expected).zip(fits) {
            assert_eq!(actual.values, expected.values);
            assert_eq!(actual.validity, expected.validity);
            assert_eq!(fit.median, 0.0);
            assert_eq!(fit.scale, 1.0);
            assert!(!fit.degenerate);
        }
        Ok(())
    }

    #[test]
    fn parallel_normalization_preserves_scalar_bits_fit_order_and_progress() -> Result<()> {
        use FeatureCellValidity::{Gap, Valid};
        use std::sync::{Arc, Mutex};
        let rows = 8192;
        let training = 17..6144;
        let mut columns = (0..32)
            .map(|index| {
                let values = (0..rows)
                    .map(|row| match index % 4 {
                        0 => 2.0, // degenerate fit
                        1 => {
                            if row % 10 == 0 {
                                1.0
                            } else {
                                0.0
                            }
                        } // zero-MAD fallback
                        _ => (row as f64 * 0.31 + index as f64).sin() + row as f64 * 0.001,
                    })
                    .collect();
                let validity = (0..rows)
                    .map(|row| if row % 13 == 0 { Gap } else { Valid })
                    .collect();
                FeatureColumnF64::new(format!("column_{index}"), values, validity)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut expected = columns.clone();
        let fits = expected
            .iter_mut()
            .map(|column| {
                crate::core::normalization::normalize_feature_column_f64(column, training.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let control = FeatureBuildControl::default().with_observer(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event, rayon::current_thread_index()));
        });
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        let (actual_fits, dropped) = pool.install(|| {
            prepare_multitimeframe_feature_columns(
                &mut columns,
                true,
                Some(training),
                false,
                4,
                &control,
                None,
            )
        })?;
        assert!(dropped.is_empty());
        assert_eq!(actual_fits, fits);
        let names = columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            normalization_fit_hash(&names, &actual_fits)?,
            normalization_fit_hash(&names, &fits)?
        );
        for (actual, expected) in columns.iter().zip(expected) {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.validity, expected.validity);
            assert_eq!(
                actual
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .map(|(event, _)| event.completed)
                .collect::<Vec<_>>(),
            (0..=32).collect::<Vec<_>>()
        );
        assert!(events.iter().all(|(event, _)| event.total == 32));
        let worker_ids = events
            .iter()
            .skip(1)
            .filter_map(|(_, worker)| *worker)
            .collect::<std::collections::HashSet<_>>();
        assert!(
            worker_ids.len() > 1,
            "normalization did not execute on multiple installed-pool workers: {worker_ids:?}"
        );
        Ok(())
    }

    #[test]
    fn parallel_normalization_cancellation_joins_the_batch_and_stops_new_scratch() -> Result<()> {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let mut columns = (0..4)
            .map(|index| {
                FeatureColumnF64::new(
                    format!("column_{index}"),
                    vec![1.0, 3.0, 7.0],
                    vec![FeatureCellValidity::Valid; 3],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let flag = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&flag);
        let control = FeatureBuildControl::new(flag).with_observer(move |event| {
            if event.completed == 1 {
                stop.store(true, Ordering::Release);
            }
        });
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
        let error = pool
            .install(|| {
                prepare_multitimeframe_feature_columns(
                    &mut columns,
                    true,
                    Some(0..3),
                    false,
                    2,
                    &control,
                    None,
                )
            })
            .unwrap_err();
        assert!(FeatureBuildCancelled::matches(&error));
        // Only the admitted first batch may have been touched. Returning the
        // error joins its Rayon tasks before caller-owned leases can be freed.
        assert_eq!(columns[2].values, [1.0, 3.0, 7.0]);
        assert_eq!(columns[3].values, [1.0, 3.0, 7.0]);
        Ok(())
    }

    #[test]
    fn higher_timeframe_parallel_budget_reserves_the_final_ram_cube() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            higher_timeframe_parallel_budget_bytes(true, 20 * gib, 10 * gib, 1 * gib),
            9 * gib
        );
        assert_eq!(
            higher_timeframe_parallel_budget_bytes(false, 20 * gib, 10 * gib, 1 * gib),
            12 * gib
        );
    }

    #[test]
    fn normalize_ohlcv_is_validation_only_and_rejects_noncanonical_order() -> Result<()> {
        let base_ms = 1_700_000_000_000_i64;
        let invalid = Ohlcv {
            timestamp: Some(vec![base_ms + 60_000, base_ms, base_ms]),
            open: vec![1.2, 1.1, 1.1],
            high: vec![1.3, 1.2, 1.2],
            low: vec![1.1, 1.0, 1.0],
            close: vec![1.25, 1.15, 1.15],
            volume: Some(vec![2.0, 1.0, 1.0]),
        };
        let err = normalize_ohlcv(&invalid)
            .expect_err("canonical writer must not sort or deduplicate input rows");
        let err_chain = format!("{err:#}");
        assert!(
            err_chain.contains("strictly increasing"),
            "unexpected error: {err_chain}"
        );

        let canonical = Ohlcv {
            timestamp: Some(vec![base_ms, base_ms + 60_000, base_ms + 120_000]),
            open: vec![1.1, 1.2, 1.3],
            high: vec![1.2, 1.3, 1.4],
            low: vec![1.0, 1.1, 1.2],
            close: vec![1.15, 1.25, 1.35],
            volume: Some(vec![0.0, 1.5, 2.0]),
        };
        let normalized = normalize_ohlcv(&canonical)?;
        assert_eq!(normalized.timestamp, canonical.timestamp);
        assert_eq!(normalized.open, canonical.open);
        assert_eq!(normalized.high, canonical.high);
        assert_eq!(normalized.low, canonical.low);
        assert_eq!(normalized.close, canonical.close);
        assert_eq!(normalized.volume, canonical.volume);
        Ok(())
    }
}

#[cfg(test)]
mod cube_assembly_tests {
    use super::*;

    /// Serializes the test-only process-global cube override. The production
    /// path has no override, but the parity test needs to exercise both sinks.
    /// Without this guard, Rust's parallel test runner can expose a temporary
    /// force-RAM value to an unrelated budget test.
    static TEST_CUBE_MODE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_test_cube_mode<T>(mode: u8, normalization_mode: u8, f: impl FnOnce() -> T) -> T {
        let _lock = TEST_CUBE_MODE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        super::TEST_CUBE_MODE.store(mode, std::sync::atomic::Ordering::SeqCst);
        super::TEST_NORMALIZATION_MODE
            .store(normalization_mode, std::sync::atomic::Ordering::SeqCst);

        struct ResetCubeMode;
        impl Drop for ResetCubeMode {
            fn drop(&mut self) {
                super::TEST_CUBE_MODE.store(0, std::sync::atomic::Ordering::SeqCst);
                super::TEST_NORMALIZATION_MODE.store(0, std::sync::atomic::Ordering::SeqCst);
                super::TEST_HIGHER_TIMEFRAME_AVAILABLE_MEMORY
                    .store(u64::MAX, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let _reset = ResetCubeMode;
        f()
    }

    /// Build a small multi-timeframe dataset where M1 and M5 are independent
    /// direct-source fixtures. Production must never manufacture M5 from M1.
    fn tiny_dataset(n: usize) -> (tempfile::TempDir, SymbolDataset) {
        let base_ms = 1_700_000_100_000_i64;
        let mut timestamp = Vec::with_capacity(n);
        let (mut open, mut high, mut low, mut close, mut volume) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            // Deterministic, non-degenerate series (constant columns would
            // normalize to zero everywhere and weaken the comparison).
            let t = i as f64;
            let px = 1.10 + (t * 0.7).sin() * 0.01 + t * 1e-5;
            timestamp.push(base_ms + (i as i64) * 60_000);
            open.push(px);
            high.push(px + 0.0008);
            low.push(px - 0.0008);
            close.push(px + (t * 0.3).cos() * 0.0004);
            volume.push(100.0 + (i % 17) as f64);
        }
        let m1 = Ohlcv {
            timestamp: Some(timestamp),
            open,
            high,
            low,
            close,
            volume: Some(volume),
        };
        let m5_len = n / 5;
        let mut m5_timestamp = Vec::with_capacity(m5_len);
        let (mut m5_open, mut m5_high, mut m5_low, mut m5_close, mut m5_volume) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..m5_len {
            let source_index = i * 5;
            let t = source_index as f64;
            let px = 1.10 + (t * 0.7).sin() * 0.01 + t * 1e-5;
            m5_timestamp.push(base_ms + (i as i64) * 300_000);
            m5_open.push(px);
            m5_high.push(px + 0.0009);
            m5_low.push(px - 0.0009);
            m5_close.push(px + (t * 0.3).cos() * 0.0003);
            m5_volume.push(500.0 + (i % 19) as f64);
        }
        let m5 = Ohlcv {
            timestamp: Some(m5_timestamp),
            open: m5_open,
            high: m5_high,
            low: m5_low,
            close: m5_close,
            volume: Some(m5_volume),
        };
        let root = tempfile::tempdir().expect("canonical multi-timeframe root");
        let mut frames = std::collections::HashMap::new();
        let mut source_artifacts = std::collections::HashMap::new();
        for (timeframe, canonical, frame) in [
            ("M1", CanonicalTimeframe::M1, m1),
            ("M5", CanonicalTimeframe::M5, m5),
        ] {
            let identity = CanonicalDatasetIdentity::external(
                "multi-timeframe-parity-test",
                "TESTFX",
                canonical,
                BarTimestampConvention::BarOpen,
            )
            .expect("test identity");
            let timestamps = frame.timestamp.as_ref().expect("test timestamps");
            let producer = crate::core::dataset_manifest::ProducerProvenanceEnvelopeV1::new(
                "neoethos.multi-timeframe-parity-test.v1",
                format!("deterministic-{timeframe}").into_bytes(),
            )
            .expect("test producer provenance");
            crate::core::dataset_manifest::publish_vortex_generation(
                crate::core::dataset_manifest::PublishRequest {
                    configured_root: root.path(),
                    identity: &identity,
                    expected_generation: None,
                    timestamp_range: crate::core::dataset_manifest::DatasetTimestampRange::new(
                        timestamps[0],
                        timestamps[timestamps.len() - 1],
                    )
                    .expect("test timestamp range"),
                    provenance: &producer,
                    chunks: ohlcv_to_vortex_chunks(&frame, 512).expect("test Vortex chunks"),
                },
            )
            .expect("publish test generation");
            let loaded = load_canonical_timeframe(root.path(), &identity)
                .expect("load pinned test generation");
            frames.insert(timeframe.to_owned(), loaded.ohlcv().clone());
            source_artifacts.insert(timeframe.to_owned(), loaded.artifact().clone());
        }
        (
            root,
            SymbolDataset {
                symbol: "TESTFX".to_string(),
                frames,
                source_artifacts,
            },
        )
    }

    #[test]
    fn stop_inside_base_production_never_starts_a_higher_timeframe_or_returns_a_cube() -> Result<()>
    {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };
        let (_root, dataset) = tiny_dataset(100);
        let options = FeatureBuildOptions {
            higher_tfs: vec!["M5".into()],
            ..Default::default()
        };
        let flag = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&flag);
        let outputs = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&outputs);
        let control = FeatureBuildControl::new(flag).with_observer(move |event| {
            assert_ne!(
                event.timeframe, "M5",
                "higher TF started after base cancellation"
            );
            if event.stage == "indicator" && event.completed == 1 {
                counted.fetch_add(1, Ordering::Relaxed);
                stop.store(true, Ordering::Release);
            }
        });
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
        let error = pool
            .install(|| {
                prepare_multitimeframe_features_with_control(&dataset, "M1", &options, &control)
            })
            .expect_err("a stopped producer must not return a partial feature cube");
        assert!(FeatureBuildCancelled::matches(&error), "{error:#}");
        assert!(outputs.load(Ordering::Relaxed) > 0);
        Ok(())
    }

    /// Exercise the storage boundary directly with multiple shards and every
    /// important validity shape. The full production graph has its own explicit
    /// acceptance test below; recomputing thousands of indicator/period points
    /// twice is not necessary to prove that this sink preserves already-built
    /// f64 columns.
    #[test]
    fn ram_and_vortex_sinks_preserve_identical_values_and_validity() {
        use crate::core::features::FeatureCellValidity::{Gap, Valid, Warmup};

        let timestamps = (0..8)
            .map(|row| 1_700_000_100_000_i64 + row * 60_000)
            .collect::<Vec<_>>();
        let base_columns = vec![
            FeatureColumnF64::new(
                "base_price",
                vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                vec![Valid; 8],
            )
            .expect("base price column"),
            FeatureColumnF64::new(
                "base_warmup",
                vec![0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                vec![Warmup, Warmup, Valid, Valid, Valid, Valid, Valid, Valid],
            )
            .expect("base warmup column"),
        ];
        let higher_columns = vec![
            FeatureColumnF64::new(
                "M5_signal",
                vec![0.0, 0.0, 10.0, 10.0, 0.0, 11.0, 11.0, 11.0],
                vec![Gap, Gap, Valid, Valid, Gap, Valid, Valid, Valid],
            )
            .expect("aligned higher-timeframe column"),
        ];

        let mut ram =
            MultiTimeframeFeatureSink::new(true, "TESTFX", "M1", 2).expect("in-memory sink");
        ram.push(
            "M1",
            &timestamps,
            base_columns.clone(),
            &FeatureBuildControl::default(),
        )
        .expect("append RAM base shard");
        ram.push(
            "M5",
            &timestamps,
            higher_columns.clone(),
            &FeatureBuildControl::default(),
        )
        .expect("append RAM higher shard");
        let ram_columns = match ram {
            MultiTimeframeFeatureSink::InMemory { columns } => columns,
            MultiTimeframeFeatureSink::Vortex { .. } => panic!("RAM sink changed backing"),
        };

        let mut vortex =
            MultiTimeframeFeatureSink::new(false, "TESTFX", "M1", 2).expect("Vortex sink");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("two writer workers");
        let completed = pool
            .install(|| {
                [("M1", base_columns), ("M5", higher_columns)]
                    .into_par_iter()
                    .map(|(tf, columns)| {
                        let mut peer = vortex.empty_peer();
                        peer.push(tf, &timestamps, columns, &FeatureBuildControl::default())?;
                        Ok(peer)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .expect("independent parallel Vortex writes");
        for peer in completed {
            vortex.append(peer).expect("ordered task-output merge");
        }
        let shards = match vortex {
            MultiTimeframeFeatureSink::Vortex { shards, .. } => shards,
            MultiTimeframeFeatureSink::InMemory { .. } => panic!("Vortex sink changed backing"),
        };
        assert_eq!(
            shards.len(),
            2,
            "one Vortex shard is required per timeframe"
        );
        let stores = crate::core::vortex_feature_store::VortexFeatureStoreSet::new(shards)
            .expect("Vortex shard set");
        let column_indices = (0..ram_columns.len()).collect::<Vec<_>>();
        let persisted = stores
            .project(&column_indices, 0..timestamps.len())
            .expect("read persisted feature columns");

        assert_eq!(persisted.timestamps, timestamps);
        assert_eq!(persisted.columns.len(), ram_columns.len());
        for (ram_column, vortex_column) in ram_columns.iter().zip(&persisted.columns) {
            assert_eq!(ram_column.name, vortex_column.name);
            assert_eq!(ram_column.validity, vortex_column.validity);
            for (row, (ram_value, vortex_value)) in ram_column
                .values
                .iter()
                .zip(&vortex_column.values)
                .enumerate()
            {
                assert!(
                    (ram_value.is_nan() && vortex_value.is_nan())
                        || ram_value.to_bits() == vortex_value.to_bits(),
                    "value mismatch at row {row} in {}: RAM={ram_value:?}, Vortex={vortex_value:?}",
                    ram_column.name
                );
            }
        }

        // The same sink can release an already-retained RAM cube when a later
        // live probe no longer leaves enough producer scratch beside it.
        let mut spill = MultiTimeframeFeatureSink::InMemory {
            columns: ram_columns,
        };
        assert!(spill.resident_column_bytes() > 0);
        spill
            .spill_to_vortex(
                "TESTFX",
                "M1",
                &timestamps,
                4 * 1024 * 1024 * 1024,
                &FeatureBuildControl::default(),
            )
            .expect("spill existing RAM columns");
        assert_eq!(spill.resident_column_bytes(), 0);
        let MultiTimeframeFeatureSink::Vortex { shards, .. } = spill else {
            panic!("spill retained RAM backing")
        };
        let spill_stores = crate::core::vortex_feature_store::VortexFeatureStoreSet::new(shards)
            .expect("spilled shard set");
        let spilled = spill_stores
            .project(&column_indices, 0..timestamps.len())
            .expect("read spilled columns");
        assert_eq!(spilled.timestamps, persisted.timestamps);
        for (actual, expected) in spilled.columns.iter().zip(&persisted.columns) {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.validity, expected.validity);
            assert_eq!(
                actual
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn budgeted_vortex_alignment_keeps_the_full_canonical_cube() -> Result<()> {
        let (_root, ds) = tiny_dataset(100);
        let opts = FeatureBuildOptions {
            prefix_base_features: true,
            higher_tfs: vec!["M5".to_owned()],
            ..Default::default()
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
        let ram = with_test_cube_mode(1, 2, || {
            pool.install(|| prepare_multitimeframe_features_with_options(&ds, "M1", &opts))
        })?;
        let base_width = ram
            .names
            .iter()
            .filter(|name| name.starts_with("M1_"))
            .count();
        assert!(base_width > 5);
        let batch_columns = (base_width / 5).max(1);
        let fixed = higher_timeframe_fixed_work_bytes(20, 100, base_width, false);
        let budget = fixed + 100 * HIGHER_TIMEFRAME_ESTIMATED_CELL_BYTES * batch_columns as u64;
        let disk = with_test_cube_mode(2, 2, || {
            TEST_HIGHER_TIMEFRAME_AVAILABLE_MEMORY.store(
                HIGHER_TIMEFRAME_SYSTEM_RESERVE_BYTES + budget * 3 / 2,
                std::sync::atomic::Ordering::SeqCst,
            );
            pool.install(|| prepare_multitimeframe_features_with_options(&ds, "M1", &opts))
        })?;
        let FeatureData::VortexSet(stores) = &disk.data else {
            anyhow::bail!("bounded feature build must use Vortex shards");
        };
        assert!(
            stores.shard_count() > 2,
            "higher-timeframe output must be partitioned, not one oversized block"
        );
        assert_eq!(ram.names, disk.names);
        assert_eq!(ram.timestamps, disk.timestamps);
        assert_eq!(ram.plan_identity(), disk.plan_identity());
        assert_eq!(ram.provenance_identity(), disk.provenance_identity());
        let FeatureData::InMemory(expected) = &ram.data else {
            anyhow::bail!("reference must retain the complete RAM cube");
        };
        let indices = (0..expected.len()).collect::<Vec<_>>();
        let actual = stores.project(&indices, 0..ram.n_samples())?;
        assert_eq!(actual.columns.len(), expected.len());
        for (actual, expected) in actual.columns.iter().zip(expected) {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.validity, expected.validity);
            assert_eq!(
                actual
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        eprintln!(
            "bounded_alignment rows={} base_columns={base_width} requested_batch_columns={batch_columns} final_columns={} shards={} task_budget_bytes={budget}",
            ram.n_samples(),
            ram.names.len(),
            stores.shard_count()
        );
        Ok(())
    }

    #[test]
    fn persisted_normalization_replays_the_production_cube_and_recipe_without_refitting()
    -> Result<()> {
        let (_root, ds) = tiny_dataset(100);
        let opts = FeatureBuildOptions {
            prefix_base_features: true,
            higher_tfs: vec!["M5".into()],
            normalization_training_rows: Some(0..80),
            drop_columns_without_normalization_training_support: true,
            ..Default::default()
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
        let batch = std::sync::Arc::new(crate::core::hpc_ta::search_working_set_batch_seeded(
            17, 8, true, 7,
        ));
        let producer_control = FeatureBuildControl::default()
            .with_indicator_compute_policy(crate::core::hpc_ta::IndicatorComputePolicy::CpuOnly);
        let training = with_test_cube_mode(1, 1, || {
            pool.install(|| {
                prepare_multitimeframe_features_batch_with_options_and_control(
                    &ds,
                    "M1",
                    &opts,
                    Some(std::sync::Arc::clone(&batch)),
                    &producer_control,
                )
            })
        })?;
        let recorded = training
            .feature_build_options()
            .expect("exact producer recipe");
        assert_eq!(
            recorded.classic_ta_working_set.as_ref(),
            Some(batch.as_ref())
        );
        let mut expected_options = opts;
        expected_options.classic_ta_working_set = Some((*batch).clone());
        assert_eq!(recorded, &expected_options);
        let opts: FeatureBuildOptions = serde_json::from_slice(&serde_json::to_vec(recorded)?)?;
        let state = training
            .normalization_fitted_state()
            .expect("producer retains fitted payload")
            .clone();
        state.validate_plan(training.plan())?;
        assert_eq!(training.feature_build_options(), Some(&opts));
        let encoded = serde_json::to_vec(&state)?;
        eprintln!(
            "production_normalization_payload rows={} columns={} plan_canonical_bytes={} plan_json_bytes={} fit_json_bytes={} recipe_json_bytes={}",
            training.n_samples(),
            training.n_features(),
            training.plan().canonical_bytes().len(),
            serde_json::to_vec(&training.plan().canonical_bytes())?.len(),
            encoded.len(),
            serde_json::to_vec(&opts)?.len(),
        );
        let restored: SearchNormalizationFittedStateV1 = serde_json::from_slice(&encoded)?;
        let cancelled_events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed_cancelled = std::sync::Arc::clone(&cancelled_events);
        let cancelled = FeatureBuildControl::new(std::sync::Arc::new(
            std::sync::atomic::AtomicBool::new(true),
        ))
        .with_observer(move |event| observed_cancelled.lock().unwrap().push(event));
        for error in [
            prepare_multitimeframe_features_batch_with_options_and_control(
                &ds,
                "M1",
                &opts,
                Some(std::sync::Arc::clone(&batch)),
                &cancelled,
            )
            .unwrap_err(),
            prepare_multitimeframe_features_raw_with_options_and_control(
                &ds, "M1", &opts, &cancelled,
            )
            .unwrap_err(),
            prepare_multitimeframe_features_with_fitted_normalization_and_control(
                &ds, "M1", &opts, &restored, &cancelled,
            )
            .unwrap_err(),
        ] {
            assert!(FeatureBuildCancelled::matches(&error));
        }
        let conflicts = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        assert!(
            cancelled_events.lock().unwrap().is_empty(),
            "Stop must refuse before any producer event"
        );
        let observed_conflicts = std::sync::Arc::clone(&conflicts);
        let conflict_control = producer_control
            .clone()
            .with_observer(move |event| observed_conflicts.lock().unwrap().push(event));
        let other = crate::core::hpc_ta::search_working_set_batch_seeded(17, 8, true, 19);
        assert_ne!(*batch, other);
        let error = prepare_multitimeframe_features_batch_with_options_and_control(
            &ds,
            "M1",
            &opts,
            Some(std::sync::Arc::new(other)),
            &conflict_control,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("conflicts with the recorded exact selection")
        );
        assert!(
            conflicts.lock().unwrap().is_empty(),
            "conflict must refuse before producer events"
        );
        let gpu_events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed_gpu = std::sync::Arc::clone(&gpu_events);
        let gpu_control = FeatureBuildControl::default()
            .with_indicator_compute_policy(crate::core::hpc_ta::IndicatorComputePolicy::GpuOnly)
            .with_observer(move |event| observed_gpu.lock().unwrap().push(event));
        for error in [
            prepare_multitimeframe_features_raw_with_options_and_control(
                &ds,
                "M1",
                &opts,
                &gpu_control,
            )
            .unwrap_err(),
            prepare_multitimeframe_features_with_fitted_normalization_and_control(
                &ds,
                "M1",
                &opts,
                &restored,
                &gpu_control,
            )
            .unwrap_err(),
        ] {
            assert!(format!("{error:#}").contains("GpuOnly"), "{error:#}");
        }
        assert!(
            gpu_events.lock().unwrap().is_empty(),
            "no producer may start"
        );
        let source = ds.canonical_frame("M1")?;
        let fallback_error = compute_production_feature_columns(
            MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy,
            ProductionFeatureProducerId::ClassicVectorTa,
            &source,
            source.len(),
            None,
            &gpu_control,
        )
        .unwrap_err();
        assert!(format!("{fallback_error:#}").contains("GpuOnly"));
        assert!(
            gpu_events
                .lock()
                .unwrap()
                .iter()
                .all(|event| { event.stage == "feature_family" && event.completed == 0 }),
            "fallback admission must not execute a CPU indicator"
        );
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = std::sync::Arc::clone(&events);
        let control = FeatureBuildControl::default()
            .with_indicator_compute_policy(crate::core::hpc_ta::IndicatorComputePolicy::CpuOnly)
            .with_observer(move |event| {
                observed.lock().unwrap().push(event);
            });
        // Ambient mode is deliberately OFF; fitted replay is artifact-owned.
        // Exercise the disk sink too, so the fit survives both physical paths.
        let replay = with_test_cube_mode(2, 2, || {
            pool.install(|| {
                prepare_multitimeframe_features_with_fitted_normalization_and_control(
                    &ds, "M1", &opts, &restored, &control,
                )
            })
        })?;
        assert!(
            !events.lock().unwrap().is_empty(),
            "frozen replay must forward producer progress"
        );
        assert_eq!(training.names, replay.names);
        assert_eq!(training.plan_identity(), replay.plan_identity());
        assert_eq!(training.provenance_identity(), replay.provenance_identity());
        assert_eq!(
            training.feature_build_options(),
            replay.feature_build_options()
        );
        assert_eq!(
            training.normalization_fitted_state(),
            replay.normalization_fitted_state()
        );
        let indices = (0..training.n_features()).collect::<Vec<_>>();
        let expected = training.project_columns(&indices, 0..training.n_samples())?;
        let actual = replay.project_columns(&indices, 0..replay.n_samples())?;
        for (actual, expected) in actual.columns.iter().zip(&expected.columns) {
            assert_eq!(actual.validity, expected.validity);
            assert_eq!(
                actual
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        // Raw replay must also remain raw when ambient training is enabled,
        // even if its recorded options carried inert historical fit flags.
        let raw = with_test_cube_mode(1, 1, || {
            pool.install(|| {
                prepare_multitimeframe_features_raw_with_options_and_control(
                    &ds, "M1", &opts, &control,
                )
            })
        })?;
        assert!(raw.normalization_fitted_state().is_none());
        assert_eq!(raw.feature_build_options(), Some(&opts));
        assert!(!raw.plan().nodes().iter().any(|node| node.operation()
            == neoethos_feature_contracts::FeatureOperationTagV1::Normalization));
        let indices = training
            .names
            .iter()
            .map(|name| {
                raw.names
                    .iter()
                    .position(|candidate| candidate == name)
                    .expect("raw producer schema includes fitted outputs")
            })
            .collect::<Vec<_>>();
        let batch = raw.project_columns(&indices, 0..raw.n_samples())?;
        let mut columns = batch.columns.clone();
        restored.apply_columns(&mut columns)?;
        for (actual, expected) in columns.iter().zip(&expected.columns) {
            assert_eq!(actual.validity, expected.validity);
            assert_eq!(
                actual
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        // The model/fold path reuses this actual producer's raw cube. Its
        // lazy fit must be bit-identical to the established eager CPU math,
        // with no extra indicator build or normalized full-matrix copy.
        let raw = std::sync::Arc::new(raw);
        let model_fit =
            pool.install(|| raw.fit_normalization(0..80, true, &FeatureBuildControl::default()))?;
        assert_eq!(model_fit, restored);
        let model_frame = raw.with_fitted_normalization(&model_fit)?;
        assert_eq!(model_frame.names, training.names);
        let model_batch = model_frame
            .project_columns(&(0..model_frame.n_features()).collect::<Vec<_>>(), 0..100)?;
        for (actual, expected) in model_batch.columns.iter().zip(&expected.columns) {
            assert_eq!(actual.validity, expected.validity);
            assert!(
                actual
                    .values
                    .iter()
                    .zip(&expected.values)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
            );
        }
        Ok(())
    }

    /// Full end-to-end acceptance proof: the in-RAM assembly (allocate once,
    /// fill per timeframe) must produce a cube byte-identical to the Vortex
    /// scratch path. This deliberately computes the complete production feature
    /// graph twice and therefore runs only when explicitly requested.
    #[test]
    #[ignore = "full production RAM/Vortex parity is an explicit long-running acceptance test"]
    fn full_production_ram_and_vortex_cubes_are_identical() {
        let (_root, ds) = tiny_dataset(512);
        let opts = FeatureBuildOptions {
            higher_tfs: vec!["M5".to_string()],
            normalization_training_rows: Some(0..400),
            drop_columns_without_normalization_training_support: true,
            ..Default::default()
        };

        // 2026-08-10: was `NEOETHOS_FEATURE_CUBE_MODE`, now the `#[cfg(test)]`
        // seam — the forcing this test needs, without a lever production can
        // be handed by a shell.
        let ram = with_test_cube_mode(1, 1, || {
            prepare_multitimeframe_features_with_options(&ds, "M1", &opts).expect("in-RAM cube")
        });
        let disk = with_test_cube_mode(2, 1, || {
            prepare_multitimeframe_features_with_options(&ds, "M1", &opts).expect("disk cube")
        });

        assert!(
            matches!(ram.data, crate::core::features::FeatureData::InMemory(_)),
            "ram mode must not touch disk"
        );
        assert!(
            matches!(
                &disk.data,
                crate::core::features::FeatureData::VortexSet(stores)
                    if stores.shard_count() == 2
            ),
            "disk mode must persist one independently releasable Vortex shard per timeframe"
        );
        assert_eq!(ram.names, disk.names, "column ORDER + names must match");
        assert_eq!(ram.timestamps, disk.timestamps);
        assert_eq!(ram.n_samples(), disk.n_samples());
        assert_eq!(ram.names.len(), disk.names.len());
        assert_eq!(ram.plan_identity(), disk.plan_identity());
        assert_eq!(ram.provenance_identity(), disk.provenance_identity());
        assert!(ram.n_samples() > 0 && !ram.names.is_empty());

        for r in 0..ram.n_samples() {
            for c in 0..ram.names.len() {
                let a = ram.cell(r, c).expect("RAM feature cell");
                let b = disk.cell(r, c).expect("Vortex feature cell");
                assert_eq!(
                    a.validity, b.validity,
                    "validity mismatch at row {r} col {c} ({})",
                    ram.names[c]
                );
                match (a.value.is_nan(), b.value.is_nan()) {
                    (true, true) => {}
                    _ => assert_eq!(
                        a.value.to_bits(),
                        b.value.to_bits(),
                        "cube mismatch at row {r} col {c} ({})",
                        ram.names[c]
                    ),
                }
            }
        }
    }

    /// 2026-08-10: renamed from `..._and_honours_the_override`. There is no
    /// override to honour any more — `NEOETHOS_FEATURE_CUBE_MODE=ram` used to
    /// return BEFORE the free-RAM check below, so the one input that could
    /// cause an OOM was the one input that skipped the OOM guard. The
    /// decision is now derived, full stop, and what this test pins is that the
    /// derivation is real rather than a constant.
    #[test]
    fn in_ram_budget_tracks_available_memory() {
        with_test_cube_mode(0, 0, || {
            // The decision must scale with the cube AND the machine — not a fixed
            // fraction. A byte-sized cube always fits; an absurd one never does.
            assert!(should_build_cube_in_ram(1));
            assert!(!should_build_cube_in_ram(u64::MAX / 4));

            // The 1.5x + 2 GB rule, pinned against a FIXED reading.
            //
            // 2026-08-10: this used to read `available_memory_bytes()` to pick a
            // cube size that should just fail, and then call
            // `should_build_cube_in_ram`, which read the probe a SECOND time. Free
            // RAM moves between two calls on a machine that is doing anything at
            // all — it moved during the build that caught this — so the "must not
            // fit" cube fit, and a correct implementation failed its own test.
            // The rule is now checked where it is deterministic.
            for available in [8_000_000_000u64, 32_000_000_000, 128_000_000_000] {
                let just_fits = (((available as f64) - 2.0e9) / 1.5) as u64;
                assert!(
                    cube_fits_in(just_fits.saturating_sub(1_000_000), available),
                    "a cube just under the budget must fit at {available} bytes available"
                );
                assert!(
                    !cube_fits_in(just_fits + 1_000_000_000, available),
                    "a cube 1 GB over the budget must NOT fit at {available} bytes available"
                );
            }

            // A failed probe is the safe answer, never 'plenty of room'.
            assert!(
                !cube_fits_in(1, 0),
                "a 0-byte probe must take the disk path"
            );

            // An absurd cube can no longer be forced into RAM by any ambient
            // value — which is the never-OOM invariant, stated as a test.
            unsafe { std::env::set_var("NEOETHOS_FEATURE_CUBE_MODE", "ram") };
            assert!(
                !should_build_cube_in_ram(u64::MAX / 4),
                "NEOETHOS_FEATURE_CUBE_MODE is retired; it must not be able to skip the \
                 free-RAM check"
            );
            unsafe { std::env::remove_var("NEOETHOS_FEATURE_CUBE_MODE") };
        });
    }

    /// The config successor must default to the derived answer, so a process
    /// that never installs one behaves exactly as it always has.
    ///
    /// Deliberately does NOT install a policy: the slot is a process-wide
    /// `OnceLock`, and a test that filled it would decide the answer for every
    /// other test in the binary depending on which ran first. The `disk` arm is
    /// pinned by inspection of the branch above, and `ram` is pinned by the
    /// type itself — see `FeatureCubeMode`, which has no such variant.
    #[test]
    fn feature_cube_policy_defaults_to_derive() {
        assert_eq!(
            super::current_feature_cube_policy(),
            neoethos_core::config::FeatureCubeMode::Auto
        );
    }
}

#[cfg(test)]
mod strict_price_tests {
    use super::*;

    /// Both read and write paths call this same validator. There is no cleanup
    /// fallback that can silently change row count or backtest history.
    #[test]
    fn the_row_validator_now_rejects_a_zero_price() {
        let row = OhlcvRow {
            timestamp: 1_700_000_000_000,
            open: 0.0,
            high: 0.83762,
            low: 0.0,
            close: 0.83417,
            volume: Some(1.0),
        };
        let err = validate_ohlcv_row(&row).expect_err("a zero price must not validate");
        assert!(
            err.to_string().contains("non-positive"),
            "unexpected message: {err}"
        );
    }
}
