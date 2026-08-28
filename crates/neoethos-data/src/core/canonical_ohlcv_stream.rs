use crate::core::dataset_manifest::{
    CandidateWriteOutcome, DatasetTimestampRange, ProducerProvenanceEnvelopeV1,
    PublishMetadataRequest, PublishResult, SelectedDatasetGenerationV1,
    publish_vortex_generation_streaming, publish_vortex_generation_streaming_exact,
};
use crate::{CanonicalDatasetIdentity, CanonicalVolumeRef, Ohlcv};
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use vortex_array::ToCanonical;
use vortex_array::dtype::{DType, PType};

static SPOOL_NONCE: AtomicU64 = AtomicU64::new(1);

/// One owned, bounded canonical OHLCV input chunk.
///
/// Ownership prevents a producer from mutating a page while Vortex encodes it.
/// The volume variant preserves the source's physical numeric type; broker tick
/// counts therefore never make an intermediate round-trip through `f64`.
#[derive(Debug)]
pub struct CanonicalOhlcvChunk {
    pub timestamp_ms: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: CanonicalVolumeChunk,
}

#[derive(Debug)]
pub enum CanonicalVolumeChunk {
    Absent,
    Float64(Vec<f64>),
    UInt64(Vec<u64>),
    Int64(Vec<i64>),
}

impl CanonicalVolumeChunk {
    fn kind(&self) -> CanonicalVolumeKind {
        match self {
            Self::Absent => CanonicalVolumeKind::Absent,
            Self::Float64(_) => CanonicalVolumeKind::Float64,
            Self::UInt64(_) => CanonicalVolumeKind::UInt64,
            Self::Int64(_) => CanonicalVolumeKind::Int64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CanonicalVolumeKind {
    Absent,
    Float64,
    UInt64,
    Int64,
}

pub struct CanonicalOhlcvStreamPublishRequest<'a, I> {
    pub configured_root: &'a Path,
    pub identity: &'a CanonicalDatasetIdentity,
    pub expected_generation: Option<&'a str>,
    pub provenance: &'a ProducerProvenanceEnvelopeV1,
    /// Inclusive lower bound accepted from the direct source request.
    pub requested_from_ms: i64,
    /// Exclusive upper bound accepted from the direct source request.
    pub requested_to_ms: i64,
    pub expected_first_timestamp_ms: i64,
    pub expected_last_timestamp_ms: i64,
    pub expected_row_count: u64,
    /// Hard resident-input ceiling enforced before a chunk is encoded.
    pub max_chunk_rows: usize,
    pub chunks: I,
}

/// Publish a fallible iterator of owned OHLCV chunks as one immutable Vortex
/// generation.
///
/// Validation spans chunk boundaries: timestamps must be strictly increasing,
/// every row must remain inside the source's half-open request, volume dtype is
/// stable, and the streamed first/last/count must equal the producer's sealed
/// summary. The manifest publisher removes staged candidates on every error and
/// changes the current pointer only after the completed generation reopens.
pub fn publish_canonical_ohlcv_stream<I>(
    request: CanonicalOhlcvStreamPublishRequest<'_, I>,
) -> Result<PublishResult>
where
    I: IntoIterator<Item = Result<CanonicalOhlcvChunk>>,
{
    publish_canonical_ohlcv_stream_inner(request, None)
}

/// Publish a canonical stream only while one exact selected manifest remains
/// current at the publication linearization point.
pub fn publish_canonical_ohlcv_stream_exact<I>(
    request: CanonicalOhlcvStreamPublishRequest<'_, I>,
    selected: &SelectedDatasetGenerationV1,
) -> Result<PublishResult>
where
    I: IntoIterator<Item = Result<CanonicalOhlcvChunk>>,
{
    publish_canonical_ohlcv_stream_inner(request, Some(selected))
}

fn publish_canonical_ohlcv_stream_inner<I>(
    request: CanonicalOhlcvStreamPublishRequest<'_, I>,
    exact_selection: Option<&SelectedDatasetGenerationV1>,
) -> Result<PublishResult>
where
    I: IntoIterator<Item = Result<CanonicalOhlcvChunk>>,
{
    validate_expected_stream_summary(&request)?;
    let requested_from_ms = request.requested_from_ms;
    let requested_to_ms = request.requested_to_ms;
    let expected_first_timestamp_ms = request.expected_first_timestamp_ms;
    let expected_last_timestamp_ms = request.expected_last_timestamp_ms;
    let expected_row_count = request.expected_row_count;
    let max_chunk_rows = request.max_chunk_rows;
    let chunks = request.chunks;

    let metadata = PublishMetadataRequest {
        configured_root: request.configured_root,
        identity: request.identity,
        expected_generation: request.expected_generation,
        provenance: request.provenance,
    };
    let write_candidate = move |candidate_path: &Path| {
        let mut state = StreamValidationState::default();
        let write_stats = {
            let arrays = chunks.into_iter().enumerate().map(|(chunk_index, chunk)| {
                let chunk = chunk.with_context(|| {
                    format!("canonical OHLCV source failed before chunk {chunk_index}")
                })?;
                state.validate_chunk(
                    &chunk,
                    chunk_index,
                    requested_from_ms,
                    requested_to_ms,
                    max_chunk_rows,
                )?;
                chunk.into_vortex_array()
            });
            crate::core::vortex_io::write_vortex_chunks_fallible(candidate_path, arrays)?
        };
        let timestamp_range = state.finish(
            expected_first_timestamp_ms,
            expected_last_timestamp_ms,
            expected_row_count,
        )?;
        if write_stats.row_count != expected_row_count {
            bail!(
                "Vortex writer row count {} disagrees with sealed stream row count {expected_row_count}",
                write_stats.row_count
            );
        }
        Ok(CandidateWriteOutcome {
            write_stats,
            timestamp_range,
        })
    };
    match exact_selection {
        Some(selected) => {
            publish_vortex_generation_streaming_exact(metadata, selected, write_candidate)
        }
        None => publish_vortex_generation_streaming(metadata, write_candidate),
    }
}

fn validate_expected_stream_summary<I>(
    request: &CanonicalOhlcvStreamPublishRequest<'_, I>,
) -> Result<()> {
    if request.requested_from_ms >= request.requested_to_ms {
        bail!("canonical OHLCV source request is empty or descending");
    }
    if request.expected_row_count == 0 {
        bail!("canonical OHLCV stream cannot declare zero rows");
    }
    if request.max_chunk_rows == 0 {
        bail!("canonical OHLCV stream max_chunk_rows must be greater than zero");
    }
    if request.expected_first_timestamp_ms > request.expected_last_timestamp_ms
        || request.expected_first_timestamp_ms < request.requested_from_ms
        || request.expected_last_timestamp_ms >= request.requested_to_ms
    {
        bail!("canonical OHLCV sealed timestamp summary is outside the half-open source request");
    }
    Ok(())
}

#[derive(Default)]
struct StreamValidationState {
    first_timestamp_ms: Option<i64>,
    last_timestamp_ms: Option<i64>,
    row_count: u64,
    volume_kind: Option<CanonicalVolumeKind>,
}

impl StreamValidationState {
    fn validate_chunk(
        &mut self,
        chunk: &CanonicalOhlcvChunk,
        chunk_index: usize,
        requested_from_ms: i64,
        requested_to_ms: i64,
        max_chunk_rows: usize,
    ) -> Result<()> {
        if chunk.timestamp_ms.is_empty() {
            bail!("canonical OHLCV chunk {chunk_index} is empty");
        }
        if chunk.timestamp_ms.len() > max_chunk_rows {
            bail!(
                "canonical OHLCV chunk {chunk_index} has {} rows, above the hard limit of {max_chunk_rows}",
                chunk.timestamp_ms.len()
            );
        }
        let chunk_first = chunk.timestamp_ms[0];
        let chunk_last = chunk.timestamp_ms[chunk.timestamp_ms.len() - 1];
        if chunk_first < requested_from_ms || chunk_last >= requested_to_ms {
            bail!(
                "canonical OHLCV chunk {chunk_index} timestamp range {chunk_first}..={chunk_last} is outside the half-open source request [{requested_from_ms}, {requested_to_ms})"
            );
        }
        if let Some(previous) = self.last_timestamp_ms
            && chunk_first <= previous
        {
            bail!(
                "canonical OHLCV chunks overlap or descend at {previous} -> {chunk_first}; refusing sort/dedup repair"
            );
        }
        let volume_kind = chunk.volume.kind();
        if let Some(expected) = self.volume_kind {
            if expected != volume_kind {
                bail!(
                    "canonical OHLCV volume dtype changed between chunks: {expected:?} -> {volume_kind:?}"
                );
            }
        } else {
            self.volume_kind = Some(volume_kind);
        }
        let chunk_rows = u64::try_from(chunk.timestamp_ms.len())
            .context("canonical OHLCV chunk row count exceeds u64")?;
        self.row_count = self
            .row_count
            .checked_add(chunk_rows)
            .context("canonical OHLCV stream row count overflows u64")?;
        self.first_timestamp_ms.get_or_insert(chunk_first);
        self.last_timestamp_ms = Some(chunk_last);
        Ok(())
    }

    fn finish(
        self,
        expected_first_timestamp_ms: i64,
        expected_last_timestamp_ms: i64,
        expected_row_count: u64,
    ) -> Result<DatasetTimestampRange> {
        let first = self
            .first_timestamp_ms
            .context("canonical OHLCV stream produced no chunks")?;
        let last = self
            .last_timestamp_ms
            .context("canonical OHLCV stream produced no chunks")?;
        if first != expected_first_timestamp_ms
            || last != expected_last_timestamp_ms
            || self.row_count != expected_row_count
        {
            bail!(
                "canonical OHLCV stream disagrees with sealed summary: expected first={expected_first_timestamp_ms} last={expected_last_timestamp_ms} rows={expected_row_count}, got first={first} last={last} rows={}",
                self.row_count
            );
        }
        DatasetTimestampRange::new(first, last)
    }
}

impl CanonicalOhlcvChunk {
    fn into_vortex_array(self) -> Result<vortex_array::ArrayRef> {
        let Self {
            timestamp_ms,
            open,
            high,
            low,
            close,
            volume,
        } = self;
        let mut ohlcv = Ohlcv {
            timestamp: Some(timestamp_ms),
            open,
            high,
            low,
            close,
            volume: None,
        };
        match volume {
            CanonicalVolumeChunk::Absent => crate::ohlcv_to_vortex_array_with_canonical_volume(
                &ohlcv,
                CanonicalVolumeRef::Absent,
            ),
            CanonicalVolumeChunk::Float64(values) => {
                ohlcv.volume = Some(values);
                let values = ohlcv
                    .volume
                    .as_deref()
                    .expect("Float64 volume was installed in the owned OHLCV chunk");
                crate::ohlcv_to_vortex_array_with_canonical_volume(
                    &ohlcv,
                    CanonicalVolumeRef::Float64(values),
                )
            }
            CanonicalVolumeChunk::UInt64(values) => {
                crate::ohlcv_to_vortex_array_with_canonical_volume(
                    &ohlcv,
                    CanonicalVolumeRef::UInt64(&values),
                )
            }
            CanonicalVolumeChunk::Int64(values) => {
                crate::ohlcv_to_vortex_array_with_canonical_volume(
                    &ohlcv,
                    CanonicalVolumeRef::Int64(&values),
                )
            }
        }
    }
}

/// A bounded reverse-order spool for sources, such as cTrader history, that
/// deliver the newest page first while canonical Vortex generations require
/// oldest-to-newest chunks.
///
/// Every temporary page is itself Vortex. No CSV/JSON/private binary runtime
/// format is introduced, and only one bounded page is materialized while the
/// final generation is encoded.
pub struct CanonicalOhlcvReverseSpool {
    state: Option<ReverseSpoolState>,
}

struct ReverseSpoolState {
    path: PathBuf,
    max_chunk_rows: usize,
    max_chunks: usize,
    page_count: usize,
    earliest_pushed_timestamp_ms: Option<i64>,
    volume_kind: Option<CanonicalVolumeKind>,
}

impl CanonicalOhlcvReverseSpool {
    pub fn create(
        configured_root: impl AsRef<Path>,
        max_chunk_rows: usize,
        max_chunks: usize,
    ) -> Result<Self> {
        if max_chunk_rows == 0 || max_chunks == 0 {
            bail!("canonical OHLCV spool limits must be greater than zero");
        }
        let configured_root = configured_root.as_ref();
        fs::create_dir_all(configured_root).with_context(|| {
            format!(
                "failed to create canonical OHLCV spool root {}",
                configured_root.display()
            )
        })?;
        let path = create_unique_spool_directory(configured_root)?;
        Ok(Self {
            state: Some(ReverseSpoolState {
                path,
                max_chunk_rows,
                max_chunks,
                page_count: 0,
                earliest_pushed_timestamp_ms: None,
                volume_kind: None,
            }),
        })
    }

    pub fn path(&self) -> &Path {
        &self.state.as_ref().expect("live spool has state").path
    }

    /// Append the next page returned by a newest-to-oldest source.
    pub fn push_latest(&mut self, chunk: CanonicalOhlcvChunk) -> Result<()> {
        let state = self
            .state
            .as_mut()
            .context("canonical OHLCV spool was consumed")?;
        if state.page_count >= state.max_chunks {
            bail!(
                "canonical OHLCV spool exceeded its hard page limit of {}",
                state.max_chunks
            );
        }
        let rows = chunk.timestamp_ms.len();
        if rows == 0 || rows > state.max_chunk_rows {
            bail!(
                "canonical OHLCV spool page has {rows} rows; required range is 1..={} rows",
                state.max_chunk_rows
            );
        }
        let first = chunk.timestamp_ms[0];
        let last = chunk.timestamp_ms[rows - 1];
        if let Some(newer_first) = state.earliest_pushed_timestamp_ms
            && last >= newer_first
        {
            bail!(
                "newest-first OHLCV spool pages overlap or descend at {last} -> {newer_first}; refusing sort/dedup repair"
            );
        }
        let volume_kind = chunk.volume.kind();
        if let Some(expected) = state.volume_kind {
            if expected != volume_kind {
                bail!(
                    "canonical OHLCV spool volume dtype changed between pages: {expected:?} -> {volume_kind:?}"
                );
            }
        }
        let array = chunk.into_vortex_array()?;
        let page_path = spool_page_path(&state.path, state.page_count);
        crate::core::vortex_io::write_vortex_array(&page_path, array).with_context(|| {
            format!(
                "failed to write bounded Vortex spool page {}",
                page_path.display()
            )
        })?;
        state.page_count += 1;
        state.earliest_pushed_timestamp_ms = Some(first);
        state.volume_kind.get_or_insert(volume_kind);
        Ok(())
    }

    pub fn into_oldest_first(mut self) -> CanonicalOhlcvReverseSpoolIter {
        let state = self.state.take().expect("live spool has state");
        let next_page = state.page_count;
        CanonicalOhlcvReverseSpoolIter {
            state: Some(state),
            next_page,
        }
    }
}

impl Drop for CanonicalOhlcvReverseSpool {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            cleanup_spool(state);
        }
    }
}

pub struct CanonicalOhlcvReverseSpoolIter {
    state: Option<ReverseSpoolState>,
    next_page: usize,
}

impl Iterator for CanonicalOhlcvReverseSpoolIter {
    type Item = Result<CanonicalOhlcvChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_page == 0 {
            if let Some(state) = self.state.take() {
                cleanup_spool(state);
            }
            return None;
        }
        self.next_page -= 1;
        let state = self
            .state
            .as_ref()
            .expect("active spool iterator has state");
        let page_path = spool_page_path(&state.path, self.next_page);
        let chunk = read_spool_chunk(&page_path).with_context(|| {
            format!(
                "failed to read bounded Vortex spool page {}",
                page_path.display()
            )
        });
        let removal = fs::remove_file(&page_path).with_context(|| {
            format!(
                "failed to remove consumed Vortex spool page {}",
                page_path.display()
            )
        });
        Some(match (chunk, removal) {
            (Ok(chunk), Ok(())) => Ok(chunk),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        })
    }
}

impl Drop for CanonicalOhlcvReverseSpoolIter {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            cleanup_spool(state);
        }
    }
}

fn create_unique_spool_directory(root: &Path) -> Result<PathBuf> {
    for _ in 0..128 {
        let nonce = SPOOL_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            ".canonical-ohlcv-spool-{}-{nonce}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create canonical OHLCV spool {}", path.display())
                });
            }
        }
    }
    bail!("failed to allocate a unique canonical OHLCV spool directory")
}

fn spool_page_path(spool_path: &Path, page: usize) -> PathBuf {
    spool_path.join(format!("page-{page:05}.vortex"))
}

fn cleanup_spool(state: ReverseSpoolState) {
    for page in 0..state.page_count {
        let path = spool_page_path(&state.path, page);
        if let Err(error) = fs::remove_file(&path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "neoethos_data::canonical_ohlcv_stream",
                path = %path.display(),
                error = %error,
                "failed to remove canonical OHLCV spool page"
            );
        }
    }
    if let Err(error) = fs::remove_dir(&state.path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            target: "neoethos_data::canonical_ohlcv_stream",
            path = %state.path.display(),
            error = %error,
            "failed to remove canonical OHLCV spool directory"
        );
    }
}

fn read_spool_chunk(path: &Path) -> Result<CanonicalOhlcvChunk> {
    let array = crate::core::vortex_io::read_vortex_array(path)?;
    let struct_array = array.to_struct();
    let field = |name: &str| {
        struct_array
            .unmasked_field_by_name(name)
            .with_context(|| format!("Vortex spool field {name} is missing"))
    };
    let timestamp_ms =
        crate::extract_non_null_primitive_vec::<i64>(field("timestamp")?, "timestamp")?;
    let open = crate::extract_non_null_primitive_vec::<f64>(field("open")?, "open")?;
    let high = crate::extract_non_null_primitive_vec::<f64>(field("high")?, "high")?;
    let low = crate::extract_non_null_primitive_vec::<f64>(field("low")?, "low")?;
    let close = crate::extract_non_null_primitive_vec::<f64>(field("close")?, "close")?;
    let volume = match struct_array.unmasked_field_by_name_opt("volume") {
        None => CanonicalVolumeChunk::Absent,
        Some(values) => match values.dtype() {
            DType::Primitive(PType::F64, _) => CanonicalVolumeChunk::Float64(
                crate::extract_non_null_primitive_vec::<f64>(values, "volume")?,
            ),
            DType::Primitive(PType::U64, _) => CanonicalVolumeChunk::UInt64(
                crate::extract_non_null_primitive_vec::<u64>(values, "volume")?,
            ),
            DType::Primitive(PType::I64, _) => CanonicalVolumeChunk::Int64(
                crate::extract_non_null_primitive_vec::<i64>(values, "volume")?,
            ),
            other => bail!("Vortex spool volume must be f64/u64/i64, got {other}"),
        },
    };
    Ok(CanonicalOhlcvChunk {
        timestamp_ms,
        open,
        high,
        low,
        close,
        volume,
    })
}
