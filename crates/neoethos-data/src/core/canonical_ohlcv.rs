//! Verified canonical OHLCV input bound to one immutable Vortex generation.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use neoethos_dataset_contracts::{CanonicalDatasetIdentity, CanonicalTimeframe};
use neoethos_feature_contracts::{SourceArtifactBindingV1, SourceSegmentV1};

use crate::Ohlcv;
use crate::core::dataset_generation_lease::DatasetGenerationLease;
use crate::core::dataset_manifest::{
    DatasetManifestV1, SelectedDatasetGenerationV1, open_current_dataset_generation,
    open_exact_dataset_generation,
};

/// Concrete immutable source artifact used by a production feature plan.
///
/// The reader lease is intentionally owned by this value so garbage
/// collection cannot remove the generation while a derived feature frame is
/// still being built or lazily consumed.
#[derive(Clone, Debug)]
pub struct CanonicalDatasetArtifactV1 {
    identity: CanonicalDatasetIdentity,
    manifest_schema_id: String,
    manifest_hash: [u8; 32],
    generation_id: String,
    vortex_hash: [u8; 32],
    source_row_count: u64,
    source_timestamp_start_ms: i64,
    source_timestamp_end_ms: i64,
    lease: Arc<DatasetGenerationLease>,
}

impl CanonicalDatasetArtifactV1 {
    pub(crate) fn from_manifest(
        manifest: &DatasetManifestV1,
        lease: Arc<DatasetGenerationLease>,
    ) -> Result<Self> {
        ensure!(
            lease.path() == manifest.generation_path(),
            "dataset lease path does not match the atomically resolved manifest generation"
        );
        let timestamp_range = manifest.timestamp_range();
        Ok(Self {
            identity: manifest.identity().clone(),
            manifest_schema_id: manifest.schema_id().to_owned(),
            manifest_hash: parse_sha256(
                "dataset manifest binding",
                manifest.manifest_binding_sha256(),
            )?,
            generation_id: manifest.generation_id().to_owned(),
            vortex_hash: parse_sha256("dataset Vortex generation", manifest.vortex_sha256())?,
            source_row_count: manifest.row_count(),
            source_timestamp_start_ms: timestamp_range.start_ms(),
            source_timestamp_end_ms: timestamp_range.end_ms(),
            lease,
        })
    }

    pub const fn identity(&self) -> &CanonicalDatasetIdentity {
        &self.identity
    }

    pub fn generation_id(&self) -> &str {
        &self.generation_id
    }

    pub const fn row_count(&self) -> u64 {
        self.source_row_count
    }

    pub const fn timestamp_start_ms(&self) -> i64 {
        self.source_timestamp_start_ms
    }

    pub const fn timestamp_end_ms(&self) -> i64 {
        self.source_timestamp_end_ms
    }

    pub const fn frame_timeframe(&self) -> CanonicalTimeframe {
        self.identity.timeframe()
    }

    pub fn lease(&self) -> &Arc<DatasetGenerationLease> {
        &self.lease
    }

    pub fn source_binding(
        &self,
        source_node_id: impl Into<String>,
    ) -> Result<SourceArtifactBindingV1> {
        self.source_binding_for_segments(
            source_node_id,
            vec![SourceSegmentV1::new(
                0,
                self.source_row_count,
                self.source_timestamp_start_ms,
                self.source_timestamp_end_ms,
            )?],
        )
    }

    fn source_binding_for_segments(
        &self,
        source_node_id: impl Into<String>,
        segments: Vec<SourceSegmentV1>,
    ) -> Result<SourceArtifactBindingV1> {
        Ok(SourceArtifactBindingV1::new(
            source_node_id,
            self.identity.clone(),
            self.manifest_schema_id.clone(),
            self.manifest_hash,
            self.generation_id.clone(),
            self.vortex_hash,
            self.identity.bar_timestamp_convention(),
            segments,
        )?)
    }

    fn verify_materialized_rows(&self, ohlcv: &Ohlcv) -> Result<()> {
        let rows = u64::try_from(ohlcv.len()).context("OHLCV row count does not fit u64")?;
        ensure!(
            rows == self.source_row_count,
            "materialized OHLCV has {rows} rows but manifest generation {} declares {}",
            self.generation_id,
            self.source_row_count
        );
        let timestamps = ohlcv
            .timestamp
            .as_deref()
            .context("canonical OHLCV is missing timestamp_ms")?;
        ensure!(
            !timestamps.is_empty(),
            "canonical OHLCV generation is empty"
        );
        ensure!(
            timestamps.first().copied() == Some(self.source_timestamp_start_ms)
                && timestamps.last().copied() == Some(self.source_timestamp_end_ms),
            "materialized OHLCV timestamp range does not match manifest generation {}",
            self.generation_id
        );
        Ok(())
    }
}

/// Fully materialized OHLCV values plus the exact pinned generation from which
/// they were decoded. Bare `Ohlcv` cannot enter production feature computation.
#[derive(Clone, Debug)]
pub struct CanonicalOhlcvFrame {
    ohlcv: Ohlcv,
    artifact: CanonicalDatasetArtifactV1,
    source_row_range: Range<u64>,
    source_segment: SourceSegmentV1,
}

impl CanonicalOhlcvFrame {
    pub(crate) fn from_parts(ohlcv: Ohlcv, artifact: CanonicalDatasetArtifactV1) -> Result<Self> {
        artifact.verify_materialized_rows(&ohlcv)?;
        let source_row_range = 0..artifact.row_count();
        let source_segment = SourceSegmentV1::new(
            source_row_range.start,
            source_row_range.end,
            artifact.timestamp_start_ms(),
            artifact.timestamp_end_ms(),
        )?;
        Ok(Self {
            ohlcv,
            artifact,
            source_row_range,
            source_segment,
        })
    }

    pub fn ohlcv(&self) -> &Ohlcv {
        &self.ohlcv
    }

    pub const fn artifact(&self) -> &CanonicalDatasetArtifactV1 {
        &self.artifact
    }

    /// Bind a feature source node to the immutable full generation plus only
    /// the exact original rows this frame consumes.
    pub fn source_binding(
        &self,
        source_node_id: impl Into<String>,
    ) -> Result<SourceArtifactBindingV1> {
        self.artifact
            .source_binding_for_segments(source_node_id, vec![self.source_segment.clone()])
    }

    /// Materialize a checked half-open row window while retaining the same
    /// immutable generation lease and recording absolute offsets into it.
    pub fn row_window(&self, start: usize, end: usize) -> Result<Self> {
        ensure!(
            start < end,
            "canonical OHLCV row window must be non-empty: {start}..{end}"
        );
        ensure!(
            end <= self.len(),
            "canonical OHLCV row window {start}..{end} is outside 0..{}",
            self.len()
        );
        let absolute_start = self
            .source_row_range
            .start
            .checked_add(u64::try_from(start).context("row-window start does not fit u64")?)
            .context("canonical OHLCV absolute row-window start overflow")?;
        let absolute_end = self
            .source_row_range
            .start
            .checked_add(u64::try_from(end).context("row-window end does not fit u64")?)
            .context("canonical OHLCV absolute row-window end overflow")?;
        ensure!(
            absolute_end <= self.source_row_range.end && absolute_end <= self.artifact.row_count(),
            "canonical OHLCV absolute row window {absolute_start}..{absolute_end} is outside the pinned generation 0..{}",
            self.artifact.row_count()
        );

        let ohlcv = crate::slice_ohlcv(&self.ohlcv, start, end, None);
        let timestamps = ohlcv
            .timestamp
            .as_deref()
            .context("canonical OHLCV row window lost timestamp_ms")?;
        let timestamp_start_ms = *timestamps
            .first()
            .context("canonical OHLCV row window is empty")?;
        let timestamp_end_ms = *timestamps
            .last()
            .context("canonical OHLCV row window is empty")?;
        let source_row_range = absolute_start..absolute_end;
        let source_segment = SourceSegmentV1::new(
            source_row_range.start,
            source_row_range.end,
            timestamp_start_ms,
            timestamp_end_ms,
        )?;
        Ok(Self {
            ohlcv,
            artifact: self.artifact.clone(),
            source_row_range,
            source_segment,
        })
    }

    /// Retain every direct row whose canonical timestamp is strictly before
    /// `end_exclusive_ms`. Each timeframe applies this same cutoff to its own
    /// independently downloaded rows; no timeframe is synthesized or sampled
    /// from another one.
    pub fn prefix_before_timestamp_ms(&self, end_exclusive_ms: i64) -> Result<Self> {
        let timestamps = self
            .ohlcv
            .timestamp
            .as_deref()
            .context("canonical OHLCV is missing timestamp_ms")?;
        let end = timestamps.partition_point(|timestamp| *timestamp < end_exclusive_ms);
        ensure!(
            end > 0,
            "canonical OHLCV half-open prefix before {end_exclusive_ms} ms is empty"
        );
        self.row_window(0, end)
    }

    pub fn len(&self) -> usize {
        self.ohlcv.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ohlcv.is_empty()
    }
}

pub fn load_canonical_timeframe(
    configured_root: impl AsRef<Path>,
    identity: &CanonicalDatasetIdentity,
) -> Result<CanonicalOhlcvFrame> {
    let (manifest, lease) = open_current_dataset_generation(configured_root, identity)?;
    let lease = Arc::new(lease);
    let ohlcv = crate::load_vortex(lease.path()).with_context(|| {
        format!(
            "decode verified canonical Vortex generation {}",
            lease.path().display()
        )
    })?;
    let artifact = CanonicalDatasetArtifactV1::from_manifest(&manifest, lease)?;
    CanonicalOhlcvFrame::from_parts(ohlcv, artifact)
}

/// Load only the exact generation+manifest receipt selected by the caller.
/// This never substitutes a newer current generation and never derives one
/// timeframe from another.
pub fn load_exact_canonical_timeframe(
    configured_root: impl AsRef<Path>,
    selected: &SelectedDatasetGenerationV1,
) -> Result<CanonicalOhlcvFrame> {
    let (manifest, lease) = open_exact_dataset_generation(configured_root, selected)?;
    let lease = Arc::new(lease);
    let ohlcv = crate::load_vortex(lease.path()).with_context(|| {
        format!(
            "decode exact verified canonical Vortex generation {}",
            lease.path().display()
        )
    })?;
    let artifact = CanonicalDatasetArtifactV1::from_manifest(&manifest, lease)?;
    CanonicalOhlcvFrame::from_parts(ohlcv, artifact)
}

pub(crate) fn materialize_pinned_canonical_timeframe_v1(
    manifest: DatasetManifestV1,
    lease: Arc<DatasetGenerationLease>,
) -> Result<CanonicalOhlcvFrame> {
    let array = lease.reopen_verified().with_context(|| {
        format!(
            "reopen and verify pinned canonical Vortex generation {}",
            lease.path().display()
        )
    })?;
    let ohlcv = crate::vortex_array_to_ohlcv(array).with_context(|| {
        format!(
            "decode verified pinned canonical Vortex generation {}",
            lease.path().display()
        )
    })?;
    let artifact = CanonicalDatasetArtifactV1::from_manifest(&manifest, lease)?;
    CanonicalOhlcvFrame::from_parts(ohlcv, artifact)
}

fn parse_sha256(label: &str, value: &str) -> Result<[u8; 32]> {
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label} is not canonical SHA-256 hex"
    );
    let mut decoded = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = decode_hex_nibble(pair[0]).context("invalid SHA-256 high nibble")?;
        let low = decode_hex_nibble(pair[1]).context("invalid SHA-256 low nibble")?;
        decoded[index] = (high << 4) | low;
    }
    Ok(decoded)
}

fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
