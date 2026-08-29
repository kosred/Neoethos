use anyhow::{Context, Result, bail};
use neoethos_data::core::canonical_ohlcv_stream::publish_canonical_ohlcv_stream_exact;
use neoethos_data::core::dataset_manifest::{ProducerProvenanceEnvelopeV1, PublishResult};
use neoethos_data::{
    CanonicalDatasetIdentity, CanonicalOhlcvChunk, CanonicalOhlcvStreamPublishRequest,
    CanonicalVolumeChunk, SelectedDatasetGenerationV1, publish_canonical_ohlcv_stream,
};
use std::path::Path;

const PROVENANCE_DOMAIN: &[u8] = b"neoethos.ctrader-trendbar-provenance.v1\0";
const PROVENANCE_VERSION: u16 = 1;
const TIMESTAMP_ENCODING_MINUTES_TO_MILLIS: u8 = 1;
const PRICE_ENCODING_RELATIVE_1E5_ROUNDED_TO_DIGITS: u8 = 1;
const VOLUME_ENCODING_TICK_COUNT_INT64: u8 = 1;
pub const BROKER_TRENDBARS_PER_CHUNK_LIMIT: usize = 5_000;
pub struct BrokerTrendbarStreamRequest<'a, I> {
    pub configured_root: &'a Path,
    pub identity: &'a CanonicalDatasetIdentity,
    pub expected_generation: Option<&'a str>,
    /// Inclusive lower bound of the logical broker request.
    pub requested_from_ms: i64,
    /// Exclusive upper bound of the logical broker request.
    pub requested_to_ms: i64,
    pub retrieved_unix_ms: u64,
    pub returned_from_ms: i64,
    pub returned_to_ms: i64,
    pub row_count: u64,
    pub chunks: I,
}

/// Convert NeoEthos' logical exclusive upper bound to cTrader's inclusive
/// `toTimestamp` wire field. This conversion belongs only at the protocol
/// boundary; manifests and every internal range remain half-open.
pub fn ctrader_inclusive_wire_to_ms(logical_exclusive_to_ms: i64) -> Result<i64> {
    logical_exclusive_to_ms
        .checked_sub(1)
        .context("cTrader half-open upper bound cannot be translated to an inclusive wire value")
}

/// Exact cTrader trendbar source contract carried inside the generic dataset
/// manifest. The schema version fixes the official timestamp, price and tick
/// volume wire mappings used before these canonical values are published.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CTraderTrendbarProvenanceV1 {
    dataset_identity: CanonicalDatasetIdentity,
    requested_from_ms: i64,
    requested_to_ms: i64,
    returned_from_ms: i64,
    returned_to_ms: i64,
    row_count: u64,
    retrieved_unix_ms: u64,
}

impl CTraderTrendbarProvenanceV1 {
    pub const SCHEMA_ID: &'static str = "neoethos.ctrader-trendbar-provenance.v1";

    fn new(
        dataset_identity: CanonicalDatasetIdentity,
        requested_from_ms: i64,
        requested_to_ms: i64,
        returned_from_ms: i64,
        returned_to_ms: i64,
        row_count: u64,
        retrieved_unix_ms: u64,
    ) -> Result<Self> {
        let value = Self {
            dataset_identity,
            requested_from_ms,
            requested_to_ms,
            returned_from_ms,
            returned_to_ms,
            row_count,
            retrieved_unix_ms,
        };
        value.validate()?;
        Ok(value)
    }

    pub const fn dataset_identity(&self) -> &CanonicalDatasetIdentity {
        &self.dataset_identity
    }

    pub const fn requested_range_ms(&self) -> (i64, i64) {
        (self.requested_from_ms, self.requested_to_ms)
    }

    pub const fn returned_range_ms(&self) -> (i64, i64) {
        (self.returned_from_ms, self.returned_to_ms)
    }

    pub const fn row_count(&self) -> u64 {
        self.row_count
    }

    pub const fn retrieved_unix_ms(&self) -> u64 {
        self.retrieved_unix_ms
    }

    fn validate(&self) -> Result<()> {
        if !self.dataset_identity.is_broker_real() {
            bail!("cTrader trendbar provenance requires a broker-bound dataset identity");
        }
        if self.requested_from_ms >= self.requested_to_ms {
            bail!("cTrader requested timestamp range is empty or descending");
        }
        if self.returned_from_ms > self.returned_to_ms
            || self.returned_from_ms < self.requested_from_ms
            || self.returned_to_ms >= self.requested_to_ms
        {
            bail!("cTrader returned timestamp range is outside the half-open request");
        }
        if self.row_count == 0 {
            bail!("cTrader trendbar provenance cannot describe zero rows");
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(PROVENANCE_DOMAIN);
        bytes.extend_from_slice(&PROVENANCE_VERSION.to_be_bytes());
        push_bytes(&mut bytes, &self.dataset_identity.canonical_bytes());
        bytes.extend_from_slice(&self.requested_from_ms.to_be_bytes());
        bytes.extend_from_slice(&self.requested_to_ms.to_be_bytes());
        bytes.extend_from_slice(&self.returned_from_ms.to_be_bytes());
        bytes.extend_from_slice(&self.returned_to_ms.to_be_bytes());
        bytes.extend_from_slice(&self.row_count.to_be_bytes());
        bytes.extend_from_slice(&self.retrieved_unix_ms.to_be_bytes());
        bytes.push(TIMESTAMP_ENCODING_MINUTES_TO_MILLIS);
        bytes.push(PRICE_ENCODING_RELATIVE_1E5_ROUNDED_TO_DIGITS);
        bytes.push(VOLUME_ENCODING_TICK_COUNT_INT64);
        bytes
    }

    fn to_envelope(&self) -> Result<ProducerProvenanceEnvelopeV1> {
        self.validate()?;
        ProducerProvenanceEnvelopeV1::new(Self::SCHEMA_ID, self.canonical_bytes())
    }

    pub fn from_envelope(envelope: &ProducerProvenanceEnvelopeV1) -> Result<Self> {
        envelope.validate()?;
        if envelope.schema_id() != Self::SCHEMA_ID {
            bail!(
                "cTrader trendbar provenance schema mismatch: expected {}, got {}",
                Self::SCHEMA_ID,
                envelope.schema_id()
            );
        }
        let mut cursor = Cursor::new(envelope.canonical_payload());
        cursor.require_exact(PROVENANCE_DOMAIN, "domain")?;
        if cursor.read_u16("version")? != PROVENANCE_VERSION {
            bail!("unsupported cTrader trendbar provenance version");
        }
        let dataset_identity =
            CanonicalDatasetIdentity::from_canonical_bytes(cursor.read_bytes("dataset identity")?)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let value = Self {
            dataset_identity,
            requested_from_ms: cursor.read_i64("requested from")?,
            requested_to_ms: cursor.read_i64("requested to")?,
            returned_from_ms: cursor.read_i64("returned from")?,
            returned_to_ms: cursor.read_i64("returned to")?,
            row_count: cursor.read_u64("row count")?,
            retrieved_unix_ms: cursor.read_u64("retrieved timestamp")?,
        };
        cursor.require_tag(TIMESTAMP_ENCODING_MINUTES_TO_MILLIS, "timestamp encoding")?;
        cursor.require_tag(
            PRICE_ENCODING_RELATIVE_1E5_ROUNDED_TO_DIGITS,
            "price encoding",
        )?;
        cursor.require_tag(VOLUME_ENCODING_TICK_COUNT_INT64, "volume encoding")?;
        if !cursor.is_empty() {
            bail!("cTrader trendbar provenance has trailing bytes");
        }
        value.validate()?;
        if value.canonical_bytes() != envelope.canonical_payload() {
            bail!("cTrader trendbar provenance is not canonically encoded");
        }
        Ok(value)
    }
}

pub fn publish_broker_trendbar_chunks<I>(
    request: BrokerTrendbarStreamRequest<'_, I>,
) -> Result<PublishResult>
where
    I: IntoIterator<Item = Result<CanonicalOhlcvChunk>>,
{
    publish_broker_trendbar_chunks_inner(request, None)
}

/// Publish broker trendbars only while the exact selected manifest remains
/// current at the dataset publication linearization point.
pub fn publish_broker_trendbar_chunks_exact<I>(
    request: BrokerTrendbarStreamRequest<'_, I>,
    selected: &SelectedDatasetGenerationV1,
) -> Result<PublishResult>
where
    I: IntoIterator<Item = Result<CanonicalOhlcvChunk>>,
{
    selected.validate()?;
    if request.identity != selected.identity() {
        bail!(
            "broker exact-publication identity {} does not match selected identity {}",
            request.identity.to_path_component(),
            selected.identity().to_path_component()
        );
    }
    if request.expected_generation != Some(selected.generation_id()) {
        bail!(
            "broker exact-publication generation {:?} does not match selected generation {}",
            request.expected_generation,
            selected.generation_id()
        );
    }
    publish_broker_trendbar_chunks_inner(request, Some(selected))
}

fn publish_broker_trendbar_chunks_inner<I>(
    request: BrokerTrendbarStreamRequest<'_, I>,
    exact_selection: Option<&SelectedDatasetGenerationV1>,
) -> Result<PublishResult>
where
    I: IntoIterator<Item = Result<CanonicalOhlcvChunk>>,
{
    if !request.identity.is_broker_real() {
        bail!("broker trendbar publication requires a broker-bound dataset identity");
    }
    let timeframe = request.identity.timeframe();
    // Native H4/H12 bars can cross broker weekend and DST session boundaries
    // with a non-multiple gap (for example 51 hours). Their exact bar-open
    // timestamps remain authoritative; strict ordering is still enforced.
    let fixed_period_ms = match timeframe {
        neoethos_data::CanonicalTimeframe::H4 | neoethos_data::CanonicalTimeframe::H12 => None,
        _ => timeframe.fixed_duration_ms(),
    };
    if request.row_count == 0 {
        bail!("cannot publish an empty cTrader trendbar response");
    }
    if request.returned_from_ms > request.returned_to_ms {
        bail!(
            "cTrader trendbar stream is duplicate or descending for {}: {} -> {}",
            request.identity.timeframe(),
            request.returned_from_ms,
            request.returned_to_ms
        );
    }
    let provenance = CTraderTrendbarProvenanceV1::new(
        request.identity.clone(),
        request.requested_from_ms,
        request.requested_to_ms,
        request.returned_from_ms,
        request.returned_to_ms,
        request.row_count,
        request.retrieved_unix_ms,
    )?;
    let envelope = provenance.to_envelope()?;
    let mut previous_timestamp_ms = None;
    let mut volume_kind = None;
    let chunks = request
        .chunks
        .into_iter()
        .enumerate()
        .map(move |(chunk_index, chunk)| {
            let chunk = chunk.with_context(|| {
                format!("cTrader trendbar spool failed before chunk {chunk_index}")
            })?;
            validate_direct_broker_chunk(
                &chunk,
                chunk_index,
                timeframe,
                fixed_period_ms,
                &mut previous_timestamp_ms,
                &mut volume_kind,
            )?;
            Ok(chunk)
        });
    let request = CanonicalOhlcvStreamPublishRequest {
        configured_root: request.configured_root,
        identity: request.identity,
        expected_generation: request.expected_generation,
        provenance: &envelope,
        requested_from_ms: request.requested_from_ms,
        requested_to_ms: request.requested_to_ms,
        expected_first_timestamp_ms: request.returned_from_ms,
        expected_last_timestamp_ms: request.returned_to_ms,
        expected_row_count: request.row_count,
        max_chunk_rows: BROKER_TRENDBARS_PER_CHUNK_LIMIT,
        chunks,
    };
    match exact_selection {
        Some(selected) => publish_canonical_ohlcv_stream_exact(request, selected),
        None => publish_canonical_ohlcv_stream(request),
    }
}

fn validate_direct_broker_chunk(
    chunk: &CanonicalOhlcvChunk,
    chunk_index: usize,
    timeframe: neoethos_data::CanonicalTimeframe,
    fixed_period_ms: Option<i64>,
    previous_timestamp_ms: &mut Option<i64>,
    volume_kind: &mut Option<bool>,
) -> Result<()> {
    let carries_volume = match &chunk.volume {
        CanonicalVolumeChunk::Absent => false,
        CanonicalVolumeChunk::Int64(_) => true,
        CanonicalVolumeChunk::Float64(_) | CanonicalVolumeChunk::UInt64(_) => {
            bail!("cTrader tick volume must retain its physical Int64 type")
        }
    };
    if let Some(expected) = *volume_kind {
        if expected != carries_volume {
            bail!("cTrader tick volume presence changed between chunks")
        }
    } else {
        *volume_kind = Some(carries_volume);
    }
    for (row, &timestamp_ms) in chunk.timestamp_ms.iter().enumerate() {
        if let Some(previous) = *previous_timestamp_ms {
            let delta = timestamp_ms - previous;
            if delta <= 0 {
                bail!(
                    "cTrader trendbar stream is duplicate or descending for {timeframe} at chunk {chunk_index} row {row}: {previous} -> {timestamp_ms}"
                );
            }
            if let Some(period_ms) = fixed_period_ms
                && delta % period_ms != 0
            {
                bail!(
                    "cTrader trendbar stream has an invalid {timeframe} fixed-period gap of {delta} ms at chunk {chunk_index} row {row}"
                );
            }
        }
        *previous_timestamp_ms = Some(timestamp_ms);
    }
    Ok(())
}

fn push_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(
        &u32::try_from(value.len())
            .expect("validated provenance field length fits u32")
            .to_be_bytes(),
    );
    target.extend_from_slice(value);
}

struct Cursor<'a> {
    remaining: &'a [u8],
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    const fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }

    fn take(&mut self, length: usize, field: &str) -> Result<&'a [u8]> {
        if self.remaining.len() < length {
            bail!("cTrader trendbar provenance is truncated at {field}");
        }
        let (value, rest) = self.remaining.split_at(length);
        self.remaining = rest;
        Ok(value)
    }

    fn require_exact(&mut self, expected: &[u8], field: &str) -> Result<()> {
        if self.take(expected.len(), field)? != expected {
            bail!("invalid cTrader trendbar provenance {field}");
        }
        Ok(())
    }

    fn read_u16(&mut self, field: &str) -> Result<u16> {
        Ok(u16::from_be_bytes(self.read_array(field)?))
    }

    fn read_u64(&mut self, field: &str) -> Result<u64> {
        Ok(u64::from_be_bytes(self.read_array(field)?))
    }

    fn read_i64(&mut self, field: &str) -> Result<i64> {
        Ok(i64::from_be_bytes(self.read_array(field)?))
    }

    fn read_bytes(&mut self, field: &str) -> Result<&'a [u8]> {
        let length = u32::from_be_bytes(self.read_array(field)?) as usize;
        self.take(length, field)
    }

    fn require_tag(&mut self, expected: u8, field: &str) -> Result<()> {
        let actual = self.take(1, field)?[0];
        if actual != expected {
            bail!("unsupported cTrader trendbar provenance {field} {actual}");
        }
        Ok(())
    }

    fn read_array<const N: usize>(&mut self, field: &str) -> Result<[u8; N]> {
        self.take(N, field)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid cTrader trendbar provenance {field}"))
    }
}
