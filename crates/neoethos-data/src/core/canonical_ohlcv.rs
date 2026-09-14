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

    /// Move a complete generation into a dataset without duplicating its price
    /// buffers or releasing its reader lease. A window must remain a frame:
    /// detaching it would lose its exact source segment and imply full history.
    pub(crate) fn into_full_parts(self) -> Result<(Ohlcv, CanonicalDatasetArtifactV1)> {
        ensure!(
            self.source_row_range == (0..self.artifact.row_count()),
            "cannot detach a partial canonical OHLCV window as a complete generation"
        );
        Ok((self.ohlcv, self.artifact))
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
        Self::copy_row_window(
            &self.ohlcv,
            &self.artifact,
            &self.source_row_range,
            start,
            end,
        )
    }

    /// Copy only the requested prefix from an already-owned full dataset.
    /// Building a full cloned frame first would temporarily allocate both the
    /// entire generation and this window in addition to the caller's input.
    pub(crate) fn copy_full_prefix_before_timestamp_ms(
        ohlcv: &Ohlcv,
        artifact: &CanonicalDatasetArtifactV1,
        end_exclusive_ms: i64,
    ) -> Result<Self> {
        artifact.verify_materialized_rows(ohlcv)?;
        let end = Self::prefix_row_end(ohlcv, end_exclusive_ms)?;
        Self::copy_row_window(ohlcv, artifact, &(0..artifact.row_count()), 0, end)
    }

    fn copy_row_window(
        source: &Ohlcv,
        artifact: &CanonicalDatasetArtifactV1,
        source_row_range: &Range<u64>,
        start: usize,
        end: usize,
    ) -> Result<Self> {
        ensure!(
            start < end,
            "canonical OHLCV row window must be non-empty: {start}..{end}"
        );
        ensure!(
            end <= source.len(),
            "canonical OHLCV row window {start}..{end} is outside 0..{}",
            source.len()
        );
        let absolute_start = source_row_range
            .start
            .checked_add(u64::try_from(start).context("row-window start does not fit u64")?)
            .context("canonical OHLCV absolute row-window start overflow")?;
        let absolute_end = source_row_range
            .start
            .checked_add(u64::try_from(end).context("row-window end does not fit u64")?)
            .context("canonical OHLCV absolute row-window end overflow")?;
        ensure!(
            absolute_end <= source_row_range.end && absolute_end <= artifact.row_count(),
            "canonical OHLCV absolute row window {absolute_start}..{absolute_end} is outside the pinned generation 0..{}",
            artifact.row_count()
        );

        let ohlcv = crate::slice_ohlcv(source, start, end, None);
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
            artifact: artifact.clone(),
            source_row_range,
            source_segment,
        })
    }

    /// Retain every direct row whose canonical timestamp is strictly before
    /// `end_exclusive_ms`. Each timeframe applies this same cutoff to its own
    /// independently downloaded rows; no timeframe is synthesized or sampled
    /// from another one.
    pub fn prefix_before_timestamp_ms(&self, end_exclusive_ms: i64) -> Result<Self> {
        let end = Self::prefix_row_end(&self.ohlcv, end_exclusive_ms)?;
        self.row_window(0, end)
    }

    fn prefix_row_end(ohlcv: &Ohlcv, end_exclusive_ms: i64) -> Result<usize> {
        let timestamps = ohlcv
            .timestamp
            .as_deref()
            .context("canonical OHLCV is missing timestamp_ms")?;
        let end = timestamps.partition_point(|timestamp| *timestamp < end_exclusive_ms);
        ensure!(
            end > 0,
            "canonical OHLCV half-open prefix before {end_exclusive_ms} ms is empty"
        );
        Ok(end)
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

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use crate::core::dataset_manifest::{
        DatasetTimestampRange, ProducerProvenanceEnvelopeV1, PublishRequest,
        publish_vortex_generation,
    };

    fn fixture(with_volume: bool) -> (tempfile::TempDir, Ohlcv, CanonicalOhlcvFrame) {
        let root = tempfile::tempdir().expect("canonical ownership root");
        let identity = CanonicalDatasetIdentity::external(
            "ownership-test",
            "EURUSD",
            CanonicalTimeframe::M1,
            neoethos_dataset_contracts::BarTimestampConvention::BarOpen,
        )
        .expect("canonical identity");
        let data = Ohlcv {
            timestamp: Some(vec![
                1_700_000_040_000,
                1_700_000_100_000,
                1_700_000_160_000,
            ]),
            open: vec![1.1234567890123, 1.2, 1.3],
            high: vec![1.2, 1.3, 1.4],
            low: vec![1.1, 1.1, 1.2],
            close: vec![1.1734567890123, 1.25, 1.35],
            volume: with_volume.then(|| vec![0.0, 16_777_217.0, 1.25]),
        };
        let provenance = ProducerProvenanceEnvelopeV1::new(
            "neoethos.ownership-test.v1",
            b"complete-generation-move".to_vec(),
        )
        .expect("fixture provenance");
        publish_vortex_generation(PublishRequest {
            configured_root: root.path(),
            identity: &identity,
            expected_generation: None,
            timestamp_range: DatasetTimestampRange::new(1_700_000_040_000, 1_700_000_160_000)
                .expect("timestamp range"),
            provenance: &provenance,
            chunks: crate::ohlcv_to_vortex_chunks(&data, 2).expect("Vortex chunks"),
        })
        .expect("publish complete generation");
        let frame = load_canonical_timeframe(root.path(), &identity).expect("verified frame");
        (root, data, frame)
    }

    fn buffer_addresses(data: &Ohlcv) -> [usize; 6] {
        [
            data.timestamp
                .as_ref()
                .map_or(0, |values| values.as_ptr() as usize),
            data.open.as_ptr() as usize,
            data.high.as_ptr() as usize,
            data.low.as_ptr() as usize,
            data.close.as_ptr() as usize,
            data.volume
                .as_ref()
                .map_or(0, |values| values.as_ptr() as usize),
        ]
    }

    fn dataset_fixture(with_volume: bool) -> (tempfile::TempDir, Ohlcv, crate::SymbolDataset) {
        let (root, expected, frame) = fixture(with_volume);
        let (ohlcv, artifact) = frame.into_full_parts().expect("complete fixture");
        let dataset = crate::SymbolDataset {
            symbol: "EURUSD".to_owned(),
            frames: std::collections::HashMap::from([("M1".to_owned(), ohlcv)]),
            source_artifacts: std::collections::HashMap::from([("M1".to_owned(), artifact)]),
        };
        (root, expected, dataset)
    }

    #[test]
    fn final_dataset_handoff_moves_original_buffers_and_the_exact_artifact() {
        for with_volume in [false, true] {
            let (_root, expected, dataset) = dataset_fixture(with_volume);
            let addresses = buffer_addresses(&dataset.frames["M1"]);
            let artifact = &dataset.source_artifacts["M1"];
            let binding = artifact
                .source_binding("source:test")
                .expect("full binding");
            let lease = Arc::downgrade(artifact.lease());

            let frame = dataset
                .into_canonical_frame("M1")
                .expect("move final base frame");
            assert_eq!(buffer_addresses(frame.ohlcv()), addresses);
            assert_eq!(frame.ohlcv().timestamp, expected.timestamp);
            assert_eq!(frame.ohlcv().volume, expected.volume);
            assert_eq!(
                frame.source_binding("source:test").expect("moved binding"),
                binding
            );
            assert!(Arc::ptr_eq(
                &lease.upgrade().expect("live lease"),
                frame.artifact().lease()
            ));
            drop(frame);
            assert!(lease.upgrade().is_none());
        }
    }

    #[test]
    fn borrowed_dataset_prefix_copies_only_requested_rows_and_keeps_exact_source_offsets() {
        for with_volume in [false, true] {
            let (_root, expected, dataset) = dataset_fixture(with_volume);
            let original_addresses = buffer_addresses(&dataset.frames["M1"]);
            let start_ms = expected.timestamp.as_ref().unwrap()[0];
            for (cutoff, rows) in [
                (start_ms + 1, 1),
                (start_ms + 60_000, 1),
                (start_ms + 60_001, 2),
                (i64::MAX, 3),
            ] {
                let window = dataset
                    .canonical_frame_before_timestamp_ms("M1", cutoff)
                    .expect("copy only the direct half-open prefix");
                assert_eq!(window.source_row_range, 0..rows as u64);
                assert_eq!(
                    window.source_segment,
                    SourceSegmentV1::new(
                        0,
                        rows as u64,
                        start_ms,
                        start_ms + (rows as i64 - 1) * 60_000,
                    )
                    .expect("independent expected segment")
                );
                assert_eq!(
                    window.ohlcv().timestamp.as_deref(),
                    Some(&expected.timestamp.as_ref().unwrap()[..rows])
                );
                assert_eq!(window.ohlcv().timestamp.as_ref().unwrap().capacity(), rows);
                for (actual, full) in [
                    (&window.ohlcv().open, &expected.open),
                    (&window.ohlcv().high, &expected.high),
                    (&window.ohlcv().low, &expected.low),
                    (&window.ohlcv().close, &expected.close),
                ] {
                    assert_eq!(
                        actual.capacity(),
                        rows,
                        "do not retain an allocation for full history"
                    );
                    assert!(
                        actual
                            .iter()
                            .map(|v| v.to_bits())
                            .eq(full[..rows].iter().map(|v| v.to_bits()))
                    );
                }
                assert_eq!(
                    window.ohlcv().volume.as_deref(),
                    expected.volume.as_ref().map(|values| &values[..rows])
                );
                if let Some(volume) = &window.ohlcv().volume {
                    assert_eq!(volume.capacity(), rows);
                }
                assert!(Arc::ptr_eq(
                    window.artifact().lease(),
                    dataset.source_artifacts["M1"].lease()
                ));
                assert_eq!(buffer_addresses(&dataset.frames["M1"]), original_addresses);
            }
        }
    }

    #[test]
    fn borrowed_prefix_and_owned_handoff_refuse_missing_or_mismatched_full_sources() {
        let (_root, _expected, mut dataset) = dataset_fixture(true);
        assert!(
            dataset
                .canonical_frame_before_timestamp_ms("M5", i64::MAX)
                .is_err()
        );
        assert!(
            dataset
                .canonical_frame_before_timestamp_ms("M1", 1_700_000_040_000)
                .is_err()
        );
        dataset.frames.get_mut("M1").unwrap().close.pop();
        let prefix_error = dataset
            .canonical_frame_before_timestamp_ms("M1", 1_700_000_100_000)
            .expect_err("a matching prefix cannot hide a damaged full generation");
        assert!(prefix_error.to_string().contains("manifest generation"));
        assert!(dataset.into_canonical_frame("M1").is_err());

        let (_root, _expected, mut dataset) = dataset_fixture(false);
        dataset.source_artifacts.clear();
        assert!(
            dataset
                .canonical_frame_before_timestamp_ms("M1", i64::MAX)
                .is_err()
        );
        assert!(dataset.into_canonical_frame("M1").is_err());
    }

    #[test]
    fn full_generation_moves_every_buffer_and_the_same_reader_lease() {
        for with_volume in [false, true] {
            let (_root, expected, frame) = fixture(with_volume);
            let addresses = buffer_addresses(frame.ohlcv());
            let binding = frame
                .source_binding("source:test")
                .expect("original binding");
            let lease = Arc::downgrade(frame.artifact().lease());
            assert_eq!(lease.strong_count(), 1);

            let (ohlcv, artifact) = frame.into_full_parts().expect("move complete generation");
            assert_eq!(
                buffer_addresses(&ohlcv),
                addresses,
                "no column may be copied"
            );
            assert_eq!(ohlcv.timestamp, expected.timestamp);
            for (actual, expected) in [
                (&ohlcv.open, &expected.open),
                (&ohlcv.high, &expected.high),
                (&ohlcv.low, &expected.low),
                (&ohlcv.close, &expected.close),
            ] {
                assert!(
                    actual
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(expected.iter().map(|v| v.to_bits()))
                );
            }
            assert_eq!(ohlcv.volume, expected.volume);
            assert_eq!(
                lease.strong_count(),
                1,
                "the moved artifact must retain its lease"
            );
            assert!(Arc::ptr_eq(
                &lease.upgrade().expect("same live lease"),
                artifact.lease()
            ));
            artifact
                .lease()
                .reopen_verified()
                .expect("still pinned and verified");

            let restored = CanonicalOhlcvFrame::from_parts(ohlcv, artifact).expect("dataset frame");
            assert_eq!(
                restored
                    .source_binding("source:test")
                    .expect("restored binding"),
                binding
            );
            drop(restored);
            assert!(
                lease.upgrade().is_none(),
                "dropping the owner releases the lease"
            );
        }
    }

    #[test]
    fn a_partial_window_cannot_lose_its_exact_source_segment_on_detach() {
        let (_root, _expected, frame) = fixture(true);
        for (start, end) in [(0, 2), (1, 3), (1, 2)] {
            let window = frame.row_window(start, end).expect("valid partial window");
            let error = window
                .into_full_parts()
                .expect_err("partial history must stay typed");
            assert!(error.to_string().contains("partial canonical OHLCV window"));
        }
        frame
            .row_window(0, 3)
            .expect("full window")
            .into_full_parts()
            .expect("a complete window preserves full-generation provenance");
    }
}
