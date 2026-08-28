//! One shared Polars-free source-to-Vortex importer.

use crate::Ohlcv;
use crate::core::dataset_manifest::{
    CandidateWriteOutcome, DatasetManifestV1, DatasetTimestampRange, PublishMetadataRequest,
    publish_vortex_generation_streaming,
};
use crate::core::import_limits::ImportLimits;
use crate::core::import_provenance::{ImportProvenanceV1, ImportSourceFormat, VolumeMappingV1};
use crate::core::source_snapshot::SourceSnapshot;
use crate::core::timestamps::{
    MAX_CANONICAL_MARKET_TIMESTAMP_MS, MIN_CANONICAL_MARKET_TIMESTAMP_MS,
};
use crate::core::vortex_io::{
    VortexWriteStats, read_vortex_file_metadata, read_vortex_ohlcv_projection_range,
    read_vortex_projection_range, write_vortex_chunks_fallible_limited,
};
use anyhow::{Context, Result, bail};
use arrow_array::{
    Array, Float64Array, Int64Array, RecordBatch, TimestampMillisecondArray, UInt64Array,
};
use arrow_ipc::reader::{
    FileReaderBuilder as IpcFileReaderBuilder, StreamReader as IpcStreamReader, read_footer_length,
};
use arrow_schema::{DataType, SchemaRef, TimeUnit};
use csv::{ByteRecord, Reader};
use neoethos_core::execution_budget::AuxiliarySlotLease;
use neoethos_dataset_contracts::CanonicalDatasetIdentity;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
// Arrow-rs 58.3 exposes no maintained public page-header preflight API. Its
// own official `parquet-layout` binary uses this public compatibility type.
// Keep the deprecation allowance narrow: an upgrade that removes it must be
// held until an equally bounded maintained API or a reviewed replacement is
// available.
#[allow(deprecated)]
use parquet::format::PageHeader;
use parquet::thrift::TSerializable;
use serde::Deserialize;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, Visitor};
use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use thrift::protocol::TCompactInputProtocol;
use vortex_array::dtype::{DType, PType};
use vortex_array::{ArrayRef, ToCanonical};

pub struct ImportRequest<'a> {
    pub source_path: &'a Path,
    pub configured_root: &'a Path,
    pub identity: &'a CanonicalDatasetIdentity,
    pub declared_format: ImportSourceFormat,
    pub expected_generation: Option<&'a str>,
    pub limits: &'a ImportLimits,
    pub auxiliary_slot: &'a AuxiliarySlotLease,
}

#[derive(Clone, Debug)]
pub struct ImportResult {
    manifest: DatasetManifestV1,
    provenance: ImportProvenanceV1,
    durable_commit_id: String,
}

impl ImportResult {
    pub const fn row_count(&self) -> u64 {
        self.manifest.row_count()
    }

    pub fn generation(&self) -> &str {
        self.manifest.generation_id()
    }

    pub fn durable_commit_id(&self) -> &str {
        &self.durable_commit_id
    }

    pub const fn manifest(&self) -> &DatasetManifestV1 {
        &self.manifest
    }

    pub const fn provenance(&self) -> &ImportProvenanceV1 {
        &self.provenance
    }
}

pub fn import_path_to_vortex(request: ImportRequest<'_>) -> Result<ImportResult> {
    if !request
        .identity
        .bar_timestamp_convention()
        .is_canonical_bar_open()
    {
        bail!("canonical import requires explicitly evidenced bar_open timestamps");
    }
    let fixed_duration_ms = request.identity.timeframe().fixed_duration_ms().with_context(|| {
        format!(
            "{} import is unsupported until its exact cTrader calendar/session grid is evidenced",
            request.identity.timeframe()
        )
    })?;
    let staging_parent = request.configured_root.join(".import-staging");
    let snapshot = SourceSnapshot::capture_path(
        request.source_path,
        &staging_parent,
        request.limits,
        request.auxiliary_slot,
    )?;

    let format = request.declared_format;
    let text_detection = match format {
        ImportSourceFormat::Csv
        | ImportSourceFormat::Tsv
        | ImportSourceFormat::JsonArray
        | ImportSourceFormat::JsonLines => {
            let detected = detect_text_source_format(snapshot.path(), request.limits)?;
            if detected.format != format {
                bail!(
                    "declared {} source format does not match sealed bytes; detected {}",
                    format,
                    detected.format
                );
            }
            Some(detected)
        }
        _ => None,
    };
    let (volume_mapping, writer): (VolumeMappingV1, Box<CandidateWriter<'_>>) = match format {
        ImportSourceFormat::Csv | ImportSourceFormat::Tsv => {
            let delimiter = text_detection
                .as_ref()
                .and_then(|detection| detection.delimiter)
                .context("delimited text detection produced no delimiter")?;
            validate_delimited_structure(snapshot.path(), delimiter, request.limits)?;
            let volume_mapping = detect_volume_mapping(snapshot.path(), delimiter, request.limits)?;
            let path = snapshot.path();
            let limits = request.limits;
            let writer_volume_mapping = volume_mapping.clone();
            (
                volume_mapping,
                Box::new(move |candidate_path| {
                    stream_delimited_to_candidate(
                        path,
                        delimiter,
                        fixed_duration_ms,
                        writer_volume_mapping,
                        limits,
                        candidate_path,
                    )
                }),
            )
        }
        ImportSourceFormat::Parquet
        | ImportSourceFormat::ArrowIpcFile
        | ImportSourceFormat::ArrowIpcStream => {
            let prepared = prepare_binary_source(snapshot.path(), format, request.limits)?;
            let volume_mapping = prepared.volume_mapping.clone();
            let limits = request.limits;
            (
                volume_mapping,
                Box::new(move |candidate_path| {
                    stream_binary_to_candidate(prepared, fixed_duration_ms, limits, candidate_path)
                }),
            )
        }
        ImportSourceFormat::JsonArray | ImportSourceFormat::JsonLines => {
            let volume_mapping =
                inspect_json_volume_mapping(snapshot.path(), format, request.limits)?;
            let path = snapshot.path();
            let limits = request.limits;
            let writer_volume_mapping = volume_mapping.clone();
            (
                volume_mapping,
                Box::new(move |candidate_path| {
                    stream_json_to_candidate(
                        path,
                        format,
                        fixed_duration_ms,
                        writer_volume_mapping,
                        limits,
                        candidate_path,
                    )
                }),
            )
        }
        ImportSourceFormat::Vortex => {
            let metadata =
                validate_vortex_source(snapshot.path(), fixed_duration_ms, request.limits)?;
            let volume_mapping = match metadata.volume_type {
                None => VolumeMappingV1::Absent,
                Some(VolumePhysicalType::Float64) => VolumeMappingV1::SourceFloat64,
                Some(VolumePhysicalType::UInt64) => VolumeMappingV1::ExactUnsignedInteger {
                    bit_width: 64,
                    unit: "source-units".to_owned(),
                },
                Some(VolumePhysicalType::Int64) => VolumeMappingV1::ExactSignedInteger {
                    bit_width: 64,
                    unit: "source-units".to_owned(),
                },
            };
            let source_path = snapshot.path();
            let limits = request.limits;
            (
                volume_mapping,
                Box::new(move |candidate_path| {
                    copy_verified_vortex_candidate(source_path, metadata, limits, candidate_path)
                }),
            )
        }
    };
    let provenance = ImportProvenanceV1::new(
        format,
        format,
        *snapshot.source_sha256(),
        snapshot.source_size(),
        snapshot.stable_source_identity(),
        request.identity.clone(),
        unix_time_ms()?,
        volume_mapping,
    )?;
    let envelope = provenance.to_envelope()?;
    let publication = publish_vortex_generation_streaming(
        PublishMetadataRequest {
            configured_root: request.configured_root,
            identity: request.identity,
            expected_generation: request.expected_generation,
            provenance: &envelope,
        },
        writer,
    )?;

    Ok(ImportResult {
        manifest: publication.manifest().clone(),
        provenance,
        durable_commit_id: publication.durable_commit_id().to_owned(),
    })
}

type CandidateWriter<'a> = dyn FnOnce(&Path) -> Result<CandidateWriteOutcome> + 'a;

type RecordBatchIterator = Box<dyn Iterator<Item = Result<RecordBatch>>>;

struct PreparedBinarySource {
    batches: RecordBatchIterator,
    columns: BinaryColumnMap,
    volume_mapping: VolumeMappingV1,
}

fn prepare_binary_source(
    path: &Path,
    format: ImportSourceFormat,
    limits: &ImportLimits,
) -> Result<PreparedBinarySource> {
    match format {
        ImportSourceFormat::Parquet => prepare_parquet(path, limits),
        ImportSourceFormat::ArrowIpcFile => prepare_ipc_file(path, limits),
        ImportSourceFormat::ArrowIpcStream => prepare_ipc_stream(path, limits),
        _ => bail!("{} is not a binary batch format", format.as_str()),
    }
}

fn prepare_parquet(path: &Path, limits: &ImportLimits) -> Result<PreparedBinarySource> {
    let footer_offset = preflight_parquet_footer(path, limits)?;
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open bounded Parquet reader {}", path.display()))?;
    validate_parquet_pages(path, builder.metadata(), footer_offset, limits)?;
    validate_parquet_metadata(builder.metadata(), limits)?;
    let schema = builder.schema().clone();
    let (columns, volume_mapping) = BinaryColumnMap::from_schema(&schema, limits)?;
    let reader = builder
        .with_batch_size(limits.max_arrow_batch_rows())
        .build()
        .context("build bounded Parquet batch reader")?;
    Ok(PreparedBinarySource {
        batches: Box::new(reader.map(|batch| batch.map_err(anyhow::Error::new))),
        columns,
        volume_mapping,
    })
}

fn preflight_parquet_footer(path: &Path, limits: &ImportLimits) -> Result<u64> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    if length < 8 {
        bail!("Parquet source is truncated before its footer");
    }
    let mut leading_magic = [0_u8; 4];
    file.read_exact(&mut leading_magic)?;
    if &leading_magic != b"PAR1" {
        bail!("Parquet source has no leading PAR1 magic");
    }
    file.seek(SeekFrom::End(-8))?;
    let mut trailer = [0_u8; 8];
    file.read_exact(&mut trailer)?;
    if &trailer[4..] != b"PAR1" {
        bail!("Parquet source has no PAR1 footer magic");
    }
    let footer_bytes = usize::try_from(u32::from_le_bytes(trailer[..4].try_into()?))
        .context("Parquet footer length does not fit usize")?;
    limits.check_parquet_footer_bytes(footer_bytes)?;
    let footer_with_trailer = u64::try_from(footer_bytes)
        .context("Parquet footer length does not fit u64")?
        .checked_add(8)
        .context("Parquet footer arithmetic overflow")?;
    if footer_with_trailer > length {
        bail!("Parquet footer length exceeds staged source length");
    }
    Ok(length - footer_with_trailer)
}

#[allow(deprecated)]
fn validate_parquet_pages(
    path: &Path,
    metadata: &parquet::file::metadata::ParquetMetaData,
    footer_offset: u64,
    limits: &ImportLimits,
) -> Result<()> {
    let mut file = File::open(path)?;
    let mut cumulative_uncompressed = 0_u64;
    for (row_group_index, row_group) in metadata.row_groups().iter().enumerate() {
        for (column_index, column) in row_group.columns().iter().enumerate() {
            let data_offset =
                nonnegative_i64(column.data_page_offset(), "Parquet data-page offset")?;
            let start = match column.dictionary_page_offset() {
                Some(offset) => {
                    data_offset.min(nonnegative_i64(offset, "Parquet dictionary-page offset")?)
                }
                None => data_offset,
            };
            let compressed_chunk =
                nonnegative_i64(column.compressed_size(), "Parquet column compressed bytes")?;
            let end = start
                .checked_add(compressed_chunk)
                .context("Parquet column byte-range overflow")?;
            if start < 4 || end > footer_offset {
                bail!(
                    "Parquet row group {row_group_index} column {column_index} escapes the bounded data region"
                );
            }
            let mut cursor = start;
            while cursor < end {
                let (header_bytes, header) = read_bounded_parquet_page_header(
                    &mut file, cursor, end, limits,
                )
                .with_context(|| {
                    format!(
                        "read Parquet row group {row_group_index} column {column_index} page header"
                    )
                })?;
                let compressed =
                    nonnegative_i32(header.compressed_page_size, "Parquet page compressed bytes")?;
                let uncompressed = nonnegative_i32(
                    header.uncompressed_page_size,
                    "Parquet page uncompressed bytes",
                )?;
                limits.check_parquet_page_bytes(compressed)?;
                if header.dictionary_page_header.is_some() {
                    limits.check_parquet_dictionary_bytes(uncompressed)?;
                } else {
                    limits.check_parquet_page_bytes(uncompressed)?;
                }
                limits.check_parquet_compression_ratio(compressed, uncompressed)?;
                cumulative_uncompressed = cumulative_uncompressed
                    .checked_add(uncompressed)
                    .context("Parquet cumulative page output overflow")?;
                limits.check_parquet_decompressed_bytes(cumulative_uncompressed)?;
                let next = cursor
                    .checked_add(u64::try_from(header_bytes)?)
                    .and_then(|value| value.checked_add(compressed))
                    .context("Parquet page range overflow")?;
                if next <= cursor || next > end {
                    bail!(
                        "Parquet row group {row_group_index} column {column_index} page extends beyond its declared chunk"
                    );
                }
                cursor = next;
            }
            if cursor != end {
                bail!(
                    "Parquet row group {row_group_index} column {column_index} page boundaries do not match its declared chunk"
                );
            }
        }
    }
    Ok(())
}

#[allow(deprecated)]
fn read_bounded_parquet_page_header(
    file: &mut File,
    offset: u64,
    column_end: u64,
    limits: &ImportLimits,
) -> Result<(usize, PageHeader)> {
    let remaining = column_end
        .checked_sub(offset)
        .context("Parquet page header starts after its column chunk")?;
    let maximum = limits.max_parquet_page_header_bytes();
    let read_bytes = usize::try_from(
        remaining.min(
            u64::try_from(maximum)?
                .checked_add(1)
                .context("Parquet page-header read limit overflow")?,
        ),
    )?;
    if read_bytes == 0 {
        bail!("Parquet column has an empty trailing page header");
    }
    let mut bytes = vec![0_u8; read_bytes];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut bytes)?;
    let mut cursor = std::io::Cursor::new(&bytes);
    let decoded = {
        let mut protocol = TCompactInputProtocol::new(&mut cursor);
        PageHeader::read_from_in_protocol(&mut protocol)
    };
    let header = match decoded {
        Ok(header) => header,
        Err(error) if read_bytes > maximum => {
            limits.check_parquet_page_header_bytes(read_bytes)?;
            return Err(error).context("decode bounded Parquet page header");
        }
        Err(error) => return Err(error).context("decode bounded Parquet page header"),
    };
    let consumed = usize::try_from(cursor.position())?;
    limits.check_parquet_page_header_bytes(consumed)?;
    Ok((consumed, header))
}

fn validate_parquet_metadata(
    metadata: &parquet::file::metadata::ParquetMetaData,
    limits: &ImportLimits,
) -> Result<()> {
    let mut total_rows = 0_u64;
    let mut total_uncompressed = 0_u64;
    for (row_group_index, row_group) in metadata.row_groups().iter().enumerate() {
        let rows = nonnegative_i64(row_group.num_rows(), "Parquet row-group rows")?;
        total_rows = total_rows
            .checked_add(rows)
            .context("Parquet total row overflow")?;
        limits.check_total_rows(total_rows)?;
        let compressed = nonnegative_i64(
            row_group.compressed_size(),
            "Parquet row-group compressed bytes",
        )?;
        let uncompressed = nonnegative_i64(
            row_group.total_byte_size(),
            "Parquet row-group uncompressed bytes",
        )?;
        limits.check_parquet_row_group_bytes(uncompressed)?;
        limits.check_parquet_compression_ratio(compressed, uncompressed)?;
        total_uncompressed = total_uncompressed
            .checked_add(uncompressed)
            .context("Parquet cumulative decompressed byte overflow")?;
        limits.check_parquet_decompressed_bytes(total_uncompressed)?;
        for (column_index, column) in row_group.columns().iter().enumerate() {
            let column_uncompressed = nonnegative_i64(
                column.uncompressed_size(),
                "Parquet column uncompressed bytes",
            )?;
            limits
                .check_parquet_page_bytes(column_uncompressed)
                .with_context(|| {
                    format!(
                        "Parquet row group {row_group_index} column {column_index} is too large to prove a bounded page/dictionary allocation"
                    )
                })?;
        }
    }
    Ok(())
}

fn prepare_ipc_file(path: &Path, limits: &ImportLimits) -> Result<PreparedBinarySource> {
    preflight_ipc_file(path, limits)?;
    let file = File::open(path)?;
    let reader = IpcFileReaderBuilder::new()
        .with_max_footer_fb_tables(limits.max_arrow_columns().saturating_mul(16))
        .with_max_footer_fb_depth(limits.max_json_nesting())
        .build(file)
        .with_context(|| format!("open bounded Arrow IPC file {}", path.display()))?;
    let schema = reader.schema();
    let (columns, volume_mapping) = BinaryColumnMap::from_schema(&schema, limits)?;
    Ok(PreparedBinarySource {
        batches: Box::new(reader.map(|batch| batch.map_err(anyhow::Error::new))),
        columns,
        volume_mapping,
    })
}

fn preflight_ipc_file(path: &Path, limits: &ImportLimits) -> Result<()> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    if length < 10 {
        bail!("Arrow IPC file is truncated before its trailer");
    }
    file.seek(SeekFrom::End(-10))?;
    let mut trailer = [0_u8; 10];
    file.read_exact(&mut trailer)?;
    let footer_bytes = read_footer_length(trailer).context("read Arrow IPC footer length")?;
    limits.check_arrow_metadata_bytes(footer_bytes)?;
    let footer_bytes_u64 =
        u64::try_from(footer_bytes).context("IPC footer length does not fit u64")?;
    if footer_bytes_u64
        .checked_add(10)
        .context("IPC footer arithmetic overflow")?
        > length
    {
        bail!("Arrow IPC footer exceeds staged source length");
    }
    file.seek(SeekFrom::End(-10 - i64::try_from(footer_bytes)?))?;
    let mut footer_data = vec![0_u8; footer_bytes];
    file.read_exact(&mut footer_data)?;
    let footer =
        arrow_ipc::root_as_footer(&footer_data).context("decode bounded Arrow IPC footer")?;
    for blocks in [footer.dictionaries(), footer.recordBatches()] {
        for block in blocks.iter().flatten() {
            let offset = nonnegative_i64(block.offset(), "IPC block offset")?;
            let metadata = nonnegative_i32(block.metaDataLength(), "IPC block metadata")?;
            let body = nonnegative_i64(block.bodyLength(), "IPC block body")?;
            limits.check_arrow_message_bytes(usize::try_from(metadata)?)?;
            limits.check_arrow_body_bytes(usize::try_from(body)?)?;
            let end = offset
                .checked_add(metadata)
                .and_then(|value| value.checked_add(body))
                .context("IPC block range overflow")?;
            if end > length {
                bail!("Arrow IPC block extends beyond staged source");
            }
            file.seek(SeekFrom::Start(offset))?;
            let mut message_block = vec![0_u8; usize::try_from(metadata)?];
            file.read_exact(&mut message_block)
                .context("truncated Arrow IPC file message metadata")?;
            let message = parse_encapsulated_ipc_message(&message_block)?;
            validate_ipc_message(&message, usize::try_from(body)?, limits)?;
        }
    }
    Ok(())
}

fn prepare_ipc_stream(path: &Path, limits: &ImportLimits) -> Result<PreparedBinarySource> {
    preflight_ipc_stream(path, limits)?;
    let file = File::open(path)?;
    let reader = IpcStreamReader::try_new(file, None)
        .with_context(|| format!("open bounded Arrow IPC stream {}", path.display()))?;
    let schema = reader.schema();
    let (columns, volume_mapping) = BinaryColumnMap::from_schema(&schema, limits)?;
    Ok(PreparedBinarySource {
        batches: Box::new(reader.map(|batch| batch.map_err(anyhow::Error::new))),
        columns,
        volume_mapping,
    })
}

fn preflight_ipc_stream(path: &Path, limits: &ImportLimits) -> Result<()> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    loop {
        let mut prefix = [0_u8; 4];
        let first = file.read(&mut prefix)?;
        if first == 0 {
            break;
        }
        if first != prefix.len() {
            file.read_exact(&mut prefix[first..])
                .context("truncated Arrow IPC stream metadata prefix")?;
        }
        if prefix == [0xff; 4] {
            file.read_exact(&mut prefix)
                .context("truncated Arrow IPC continuation prefix")?;
        }
        let signed_metadata = i32::from_le_bytes(prefix);
        if signed_metadata < 0 {
            bail!("Arrow IPC stream has a negative metadata length");
        }
        let metadata_bytes = usize::try_from(signed_metadata)?;
        if metadata_bytes == 0 {
            if file.stream_position()? != length {
                bail!("Arrow IPC stream has trailing bytes after its end marker");
            }
            break;
        }
        limits.check_arrow_metadata_bytes(metadata_bytes)?;
        limits.check_arrow_message_bytes(metadata_bytes)?;
        let mut metadata = vec![0_u8; metadata_bytes];
        file.read_exact(&mut metadata)
            .context("truncated Arrow IPC stream metadata")?;
        let message = arrow_ipc::root_as_message(&metadata)
            .context("decode bounded Arrow IPC stream message")?;
        let body_bytes = nonnegative_i64(message.bodyLength(), "IPC stream body")?;
        limits.check_arrow_body_bytes(usize::try_from(body_bytes)?)?;
        validate_ipc_message(&message, usize::try_from(body_bytes)?, limits)?;
        let body_end = file
            .stream_position()?
            .checked_add(body_bytes)
            .context("IPC stream body range overflow")?;
        if body_end > length {
            bail!("Arrow IPC stream body extends beyond staged source");
        }
        file.seek(SeekFrom::Start(body_end))?;
    }
    Ok(())
}

fn parse_encapsulated_ipc_message(bytes: &[u8]) -> Result<arrow_ipc::Message<'_>> {
    if bytes.len() < 4 {
        bail!("Arrow IPC encapsulated message is shorter than its length prefix");
    }
    let (length_offset, message_offset): (usize, usize) = if bytes[..4] == [0xff; 4] {
        if bytes.len() < 8 {
            bail!("Arrow IPC continuation marker has no metadata length");
        }
        (4, 8)
    } else {
        (0, 4)
    };
    let declared = i32::from_le_bytes(bytes[length_offset..length_offset + 4].try_into()?);
    if declared < 0 {
        bail!("Arrow IPC encapsulated message has a negative metadata length");
    }
    let declared = usize::try_from(declared)?;
    let end = message_offset
        .checked_add(declared)
        .context("Arrow IPC encapsulated message range overflow")?;
    if end > bytes.len() {
        bail!("Arrow IPC encapsulated message metadata is truncated");
    }
    arrow_ipc::root_as_message(&bytes[message_offset..end])
        .context("decode bounded Arrow IPC file message")
}

fn validate_ipc_message(
    message: &arrow_ipc::Message<'_>,
    body_bytes: usize,
    limits: &ImportLimits,
) -> Result<()> {
    use arrow_ipc::MessageHeader;

    let batch = match message.header_type() {
        MessageHeader::RecordBatch => message
            .header_as_record_batch()
            .context("Arrow IPC message declares RecordBatch without a batch header")?,
        MessageHeader::DictionaryBatch => message
            .header_as_dictionary_batch()
            .and_then(|dictionary| dictionary.data())
            .context("Arrow IPC dictionary message has no record-batch payload")?,
        _ => return Ok(()),
    };
    let rows = nonnegative_i64(batch.length(), "Arrow IPC record-batch rows")?;
    limits.check_arrow_batch_rows(usize::try_from(rows)?)?;
    limits.check_arrow_batch_bytes(limits.checked_layout_bytes(
        usize::try_from(rows)?,
        6,
        std::mem::size_of::<f64>(),
    )?)?;
    if let Some(nodes) = batch.nodes() {
        limits.check_arrow_columns(nodes.len())?;
    }
    if let Some(buffers) = batch.buffers() {
        let maximum_buffers = limits
            .max_arrow_columns()
            .checked_mul(3)
            .context("Arrow IPC buffer-count limit overflow")?;
        limits.check_arrow_columns(buffers.len().div_ceil(3))?;
        if buffers.len() > maximum_buffers {
            bail!(
                "Arrow IPC batch declares {} buffers above the canonical limit of {maximum_buffers}",
                buffers.len()
            );
        }
        for (index, buffer) in buffers.iter().enumerate() {
            let offset = nonnegative_i64(buffer.offset(), "Arrow IPC buffer offset")?;
            let length = nonnegative_i64(buffer.length(), "Arrow IPC buffer length")?;
            let end = offset
                .checked_add(length)
                .context("Arrow IPC buffer range overflow")?;
            if end > u64::try_from(body_bytes)? {
                bail!("Arrow IPC buffer {index} extends beyond its bounded message body");
            }
        }
    }
    Ok(())
}

fn nonnegative_i64(value: i64, field: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("{field} is negative"))
}

fn nonnegative_i32(value: i32, field: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("{field} is negative"))
}

fn stream_binary_to_candidate(
    prepared: PreparedBinarySource,
    fixed_duration_ms: i64,
    limits: &ImportLimits,
    candidate_path: &Path,
) -> Result<CandidateWriteOutcome> {
    let mut chunks = BinaryChunkReader::new(prepared, fixed_duration_ms, limits);
    let write_stats = write_vortex_chunks_fallible_limited(
        candidate_path,
        &mut chunks,
        limits.max_candidate_bytes(),
    )?;
    let timestamp_range = chunks.timestamp_range()?;
    if write_stats.row_count != chunks.row_count {
        bail!(
            "binary reader row-count mismatch: parser {}, Vortex writer {}",
            chunks.row_count,
            write_stats.row_count
        );
    }
    Ok(CandidateWriteOutcome {
        write_stats,
        timestamp_range,
    })
}

#[derive(Clone, Copy, Debug)]
enum TimestampPhysicalType {
    Int64,
    MillisecondTimestamp,
}

#[derive(Clone, Copy, Debug)]
enum VolumePhysicalType {
    Float64,
    UInt64,
    Int64,
}

#[derive(Debug)]
enum CanonicalVolumeChunk {
    Absent,
    Float64(Vec<f64>),
    UInt64(Vec<u64>),
    Int64(Vec<i64>),
}

#[derive(Clone, Copy, Debug)]
enum TextVolumeValue {
    UInt64(u64),
    Float64(f64),
}

#[derive(Default)]
struct TextVolumeClassifier {
    saw_integer: bool,
    saw_float: bool,
    saw_inexact_integer: bool,
}

impl TextVolumeClassifier {
    fn observe(&mut self, value: TextVolumeValue) {
        match value {
            TextVolumeValue::UInt64(value) => {
                self.saw_integer = true;
                self.saw_inexact_integer |= !crate::u64_has_exact_f64_mapping(value);
            }
            TextVolumeValue::Float64(_) => self.saw_float = true,
        }
    }

    fn finish(self, has_volume: bool, source_kind: &str) -> Result<VolumeMappingV1> {
        if !has_volume {
            return Ok(VolumeMappingV1::Absent);
        }
        if self.saw_float {
            if self.saw_inexact_integer {
                bail!(
                    "{source_kind} mixes decimal volume with an integer that has no exact f64 mapping; one canonical physical volume type cannot preserve both"
                );
            }
            return Ok(VolumeMappingV1::SourceFloat64);
        }
        if self.saw_integer {
            return Ok(VolumeMappingV1::ExactUnsignedInteger {
                bit_width: 64,
                unit: "source-units".to_owned(),
            });
        }
        bail!("{source_kind} declared volume but produced no typed volume values")
    }
}

#[derive(Clone, Copy, Debug)]
struct BinaryColumnMap {
    timestamp: usize,
    timestamp_type: TimestampPhysicalType,
    open: usize,
    high: usize,
    low: usize,
    close: usize,
    volume: Option<(usize, VolumePhysicalType)>,
}

impl BinaryColumnMap {
    fn from_schema(schema: &SchemaRef, limits: &ImportLimits) -> Result<(Self, VolumeMappingV1)> {
        limits.check_arrow_columns(schema.fields().len())?;
        let mut timestamp = None;
        let mut open = None;
        let mut high = None;
        let mut low = None;
        let mut close = None;
        let mut volume = None;
        for (index, field) in schema.fields().iter().enumerate() {
            if matches!(field.data_type(), DataType::Float32) {
                bail!(
                    "binary field {} is Float32 and precision-unrecoverable; implicit widening is forbidden",
                    field.name()
                );
            }
            let slot = match field.name().trim().to_ascii_lowercase().as_str() {
                "timestamp" | "time" => Some(("timestamp", &mut timestamp)),
                "open" | "o" => Some(("open", &mut open)),
                "high" | "h" => Some(("high", &mut high)),
                "low" | "l" => Some(("low", &mut low)),
                "close" | "c" => Some(("close", &mut close)),
                "volume" | "vol" | "v" => Some(("volume", &mut volume)),
                other => bail!("binary schema has unexpected noncanonical field {other}"),
            };
            if let Some((name, slot)) = slot {
                if slot.replace(index).is_some() {
                    bail!("ambiguous duplicate {name} alias in binary schema");
                }
            }
        }
        let timestamp = timestamp.context("binary schema has no timestamp/time field")?;
        let timestamp_type = match schema.field(timestamp).data_type() {
            DataType::Int64 => TimestampPhysicalType::Int64,
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                TimestampPhysicalType::MillisecondTimestamp
            }
            other => bail!(
                "binary timestamp must be physical Int64 or Timestamp(Millisecond), got {other}"
            ),
        };
        let open = require_float64(schema, open, "open")?;
        let high = require_float64(schema, high, "high")?;
        let low = require_float64(schema, low, "low")?;
        let close = require_float64(schema, close, "close")?;
        let (volume, volume_mapping) = match volume {
            None => (None, VolumeMappingV1::Absent),
            Some(index) => match schema.field(index).data_type() {
                DataType::Float64 => (
                    Some((index, VolumePhysicalType::Float64)),
                    VolumeMappingV1::SourceFloat64,
                ),
                DataType::UInt64 => (
                    Some((index, VolumePhysicalType::UInt64)),
                    VolumeMappingV1::ExactUnsignedInteger {
                        bit_width: 64,
                        unit: "source-units".to_owned(),
                    },
                ),
                DataType::Int64 => (
                    Some((index, VolumePhysicalType::Int64)),
                    VolumeMappingV1::ExactSignedInteger {
                        bit_width: 64,
                        unit: "source-units".to_owned(),
                    },
                ),
                other => bail!(
                    "binary volume must be Float64 or an exact integer physical type, got {other}"
                ),
            },
        };
        Ok((
            Self {
                timestamp,
                timestamp_type,
                open,
                high,
                low,
                close,
                volume,
            },
            volume_mapping,
        ))
    }
}

fn require_float64(schema: &SchemaRef, index: Option<usize>, name: &str) -> Result<usize> {
    let index = index.with_context(|| format!("binary schema has no {name} field"))?;
    match schema.field(index).data_type() {
        DataType::Float64 => Ok(index),
        DataType::Float32 => bail!(
            "binary field {name} is Float32 and precision-unrecoverable; implicit widening is forbidden"
        ),
        other => bail!("binary field {name} must be Float64, got {other}"),
    }
}

struct BinaryChunkReader<'a> {
    batches: RecordBatchIterator,
    columns: BinaryColumnMap,
    fixed_duration_ms: i64,
    limits: &'a ImportLimits,
    row_count: u64,
    previous_timestamp: Option<i64>,
    first_timestamp: Option<i64>,
    last_timestamp: Option<i64>,
}

impl<'a> BinaryChunkReader<'a> {
    fn new(
        prepared: PreparedBinarySource,
        fixed_duration_ms: i64,
        limits: &'a ImportLimits,
    ) -> Self {
        Self {
            batches: prepared.batches,
            columns: prepared.columns,
            fixed_duration_ms,
            limits,
            row_count: 0,
            previous_timestamp: None,
            first_timestamp: None,
            last_timestamp: None,
        }
    }

    fn timestamp_range(&self) -> Result<DatasetTimestampRange> {
        DatasetTimestampRange::new(
            self.first_timestamp.context("binary source has no rows")?,
            self.last_timestamp.context("binary source has no rows")?,
        )
    }

    fn read_chunk(&mut self) -> Result<Option<ArrayRef>> {
        let Some(batch) = self.batches.next().transpose()? else {
            return Ok(None);
        };
        self.limits.check_arrow_batch_rows(batch.num_rows())?;
        self.limits.check_arrow_columns(batch.num_columns())?;
        self.limits
            .check_arrow_batch_bytes(batch.get_array_memory_size())?;
        if batch.num_rows() == 0 {
            return self.read_chunk();
        }

        let timestamps = extract_timestamps(&batch, self.columns)?;
        let open = extract_float64(&batch, self.columns.open, "open")?;
        let high = extract_float64(&batch, self.columns.high, "high")?;
        let low = extract_float64(&batch, self.columns.low, "low")?;
        let close = extract_float64(&batch, self.columns.close, "close")?;
        let volume = match self.columns.volume {
            Some((index, physical)) => extract_volume(&batch, index, physical)?,
            None => CanonicalVolumeChunk::Absent,
        };
        for (offset, &timestamp) in timestamps.iter().enumerate() {
            let source_row = self
                .row_count
                .checked_add(u64::try_from(offset)?)
                .and_then(|value| value.checked_add(1))
                .context("binary source row-number overflow")?;
            validate_grid_timestamp(
                timestamp,
                self.previous_timestamp,
                self.fixed_duration_ms,
                source_row,
                "binary",
            )?;
            self.previous_timestamp = Some(timestamp);
            self.first_timestamp.get_or_insert(timestamp);
            self.last_timestamp = Some(timestamp);
        }
        self.row_count = self
            .row_count
            .checked_add(u64::try_from(batch.num_rows())?)
            .context("binary total row-count overflow")?;
        self.limits.check_total_rows(self.row_count)?;
        canonical_market_chunk_to_vortex_array(timestamps, open, high, low, close, volume).map(Some)
    }
}

impl Iterator for BinaryChunkReader<'_> {
    type Item = Result<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_chunk().transpose()
    }
}

fn extract_timestamps(batch: &RecordBatch, columns: BinaryColumnMap) -> Result<Vec<i64>> {
    let array = batch.column(columns.timestamp);
    reject_nulls(array.as_ref(), "timestamp")?;
    match columns.timestamp_type {
        TimestampPhysicalType::Int64 => Ok(array
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("binary timestamp array disagrees with Int64 schema")?
            .values()
            .to_vec()),
        TimestampPhysicalType::MillisecondTimestamp => Ok(array
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .context("binary timestamp array disagrees with millisecond schema")?
            .values()
            .to_vec()),
    }
}

fn extract_float64(batch: &RecordBatch, index: usize, label: &str) -> Result<Vec<f64>> {
    let array = batch.column(index);
    reject_nulls(array.as_ref(), label)?;
    Ok(array
        .as_any()
        .downcast_ref::<Float64Array>()
        .with_context(|| format!("binary {label} array disagrees with Float64 schema"))?
        .values()
        .to_vec())
}

fn extract_volume(
    batch: &RecordBatch,
    index: usize,
    physical: VolumePhysicalType,
) -> Result<CanonicalVolumeChunk> {
    let array = batch.column(index);
    reject_nulls(array.as_ref(), "volume")?;
    match physical {
        VolumePhysicalType::Float64 => Ok(CanonicalVolumeChunk::Float64(
            array
                .as_any()
                .downcast_ref::<Float64Array>()
                .context("binary volume disagrees with Float64 schema")?
                .values()
                .to_vec(),
        )),
        VolumePhysicalType::UInt64 => Ok(CanonicalVolumeChunk::UInt64(
            array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .context("binary volume disagrees with UInt64 schema")?
                .values()
                .to_vec(),
        )),
        VolumePhysicalType::Int64 => {
            let values = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("binary volume disagrees with Int64 schema")?
                .values()
                .to_vec();
            if let Some(value) = values.iter().find(|value| **value < 0) {
                bail!("raw Int64 volume {value} is negative");
            }
            Ok(CanonicalVolumeChunk::Int64(values))
        }
    }
}

fn canonical_market_chunk_to_vortex_array(
    timestamps: Vec<i64>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: CanonicalVolumeChunk,
) -> Result<ArrayRef> {
    let mut ohlcv = Ohlcv {
        timestamp: Some(timestamps),
        open,
        high,
        low,
        close,
        volume: None,
    };
    match volume {
        CanonicalVolumeChunk::Absent => crate::ohlcv_to_vortex_array_with_canonical_volume(
            &ohlcv,
            crate::CanonicalVolumeRef::Absent,
        ),
        CanonicalVolumeChunk::Float64(values) => {
            ohlcv.volume = Some(values);
            let values = ohlcv
                .volume
                .as_deref()
                .context("canonical Float64 volume disappeared")?;
            crate::ohlcv_to_vortex_array_with_canonical_volume(
                &ohlcv,
                crate::CanonicalVolumeRef::Float64(values),
            )
        }
        CanonicalVolumeChunk::UInt64(values) => crate::ohlcv_to_vortex_array_with_canonical_volume(
            &ohlcv,
            crate::CanonicalVolumeRef::UInt64(&values),
        ),
        CanonicalVolumeChunk::Int64(values) => crate::ohlcv_to_vortex_array_with_canonical_volume(
            &ohlcv,
            crate::CanonicalVolumeRef::Int64(&values),
        ),
    }
}

fn append_text_volume(
    destination: &mut CanonicalVolumeChunk,
    value: Option<TextVolumeValue>,
    source_row: u64,
    source_kind: &str,
) -> Result<()> {
    match (destination, value) {
        (CanonicalVolumeChunk::Absent, None) => Ok(()),
        (CanonicalVolumeChunk::UInt64(values), Some(TextVolumeValue::UInt64(value))) => {
            values.push(value);
            Ok(())
        }
        (CanonicalVolumeChunk::Float64(values), Some(TextVolumeValue::Float64(value))) => {
            values.push(value);
            Ok(())
        }
        (CanonicalVolumeChunk::Float64(values), Some(TextVolumeValue::UInt64(value))) => {
            if !crate::u64_has_exact_f64_mapping(value) {
                bail!(
                    "{source_kind} row {source_row} raw integer volume {value} cannot enter a Float64 canonical column exactly"
                );
            }
            values.push(value as f64);
            Ok(())
        }
        (CanonicalVolumeChunk::Int64(_), _) => {
            bail!("{source_kind} text input cannot produce a signed raw volume column")
        }
        (CanonicalVolumeChunk::Absent, Some(_)) => {
            bail!("{source_kind} row {source_row} unexpectedly contains volume")
        }
        (_, None) => bail!("{source_kind} row {source_row} is missing declared volume"),
        (CanonicalVolumeChunk::UInt64(_), Some(TextVolumeValue::Float64(_))) => bail!(
            "{source_kind} row {source_row} decimal volume disagrees with the raw UInt64 preflight"
        ),
    }
}

fn reject_nulls(array: &dyn Array, label: &str) -> Result<()> {
    if array.null_count() != 0 {
        bail!("binary field {label} contains null values");
    }
    Ok(())
}

fn validate_grid_timestamp(
    timestamp: i64,
    previous: Option<i64>,
    fixed_duration_ms: i64,
    source_row: u64,
    source_kind: &str,
) -> Result<()> {
    if !(MIN_CANONICAL_MARKET_TIMESTAMP_MS..=MAX_CANONICAL_MARKET_TIMESTAMP_MS).contains(&timestamp)
    {
        bail!(
            "{source_kind} row {source_row} timestamp is not an in-range i64 Unix millisecond value: {timestamp}"
        );
    }
    if timestamp.rem_euclid(fixed_duration_ms) != 0 {
        bail!(
            "{source_kind} row {source_row} timestamp {timestamp} is off the declared {fixed_duration_ms} ms bar-open grid"
        );
    }
    if let Some(previous) = previous {
        if timestamp <= previous {
            bail!(
                "{source_kind} row {source_row} timestamp {timestamp} is not strictly after {previous}"
            );
        }
        let gap = timestamp - previous;
        if gap.rem_euclid(fixed_duration_ms) != 0 {
            bail!(
                "{source_kind} row {source_row} timestamp gap {gap} is not a legal multiple of {fixed_duration_ms} ms"
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct VerifiedVortexSource {
    row_count: u64,
    file_size: u64,
    volume_type: Option<VolumePhysicalType>,
    timestamp_range: DatasetTimestampRange,
}

fn validate_vortex_source(
    path: &Path,
    fixed_duration_ms: i64,
    limits: &ImportLimits,
) -> Result<VerifiedVortexSource> {
    preflight_vortex_footer(path, limits)?;
    let metadata = read_vortex_file_metadata(path)?;
    limits.check_vortex_metadata_bytes(metadata.footer_bytes())?;
    limits.check_vortex_chunk_bytes(metadata.max_segment_bytes())?;
    limits.check_total_rows(metadata.row_count())?;
    if metadata.row_count() == 0 {
        bail!("Vortex source contains no market rows");
    }
    let fields = match metadata.dtype() {
        DType::Struct(fields, nullability) if !nullability.is_nullable() => fields,
        DType::Struct(_, _) => bail!("Vortex root struct must be non-nullable"),
        other => bail!("Vortex source root must be a canonical OHLCV struct, got {other}"),
    };
    let allowed = ["timestamp", "open", "high", "low", "close", "volume"];
    let mut seen = HashSet::with_capacity(fields.nfields());
    for name in fields.names().iter() {
        let name = name.as_ref();
        if !allowed.contains(&name) {
            bail!("Vortex source has unexpected canonical field {name}");
        }
        if !seen.insert(name) {
            bail!("Vortex source has duplicate canonical field {name}");
        }
    }
    for required in ["timestamp", "open", "high", "low", "close"] {
        if !seen.contains(required) {
            bail!("Vortex source is missing canonical field {required}");
        }
    }
    require_vortex_primitive(fields, "timestamp", PType::I64)?;
    for price in ["open", "high", "low", "close"] {
        require_vortex_primitive(fields, price, PType::F64)?;
    }
    let volume_type = if seen.contains("volume") {
        Some(vortex_volume_physical_type(fields)?)
    } else {
        None
    };
    if fields.nfields() != if volume_type.is_some() { 6 } else { 5 } {
        bail!("Vortex canonical schema contains an ambiguous field set");
    }

    let rows_per_batch = u64::try_from(limits.max_arrow_batch_rows())
        .context("Vortex batch-row limit does not fit u64")?;
    if rows_per_batch == 0 {
        bail!("Vortex validation batch-row limit must be positive");
    }
    let mut start = 0_u64;
    let mut previous_timestamp = None;
    let mut first_timestamp = None;
    let mut last_timestamp = None;
    while start < metadata.row_count() {
        let end = start
            .checked_add(rows_per_batch)
            .context("Vortex validation row-range overflow")?
            .min(metadata.row_count());
        let rows = usize::try_from(end - start)?;
        let columns = if volume_type.is_some() { 6 } else { 5 };
        let projected_bytes = limits.checked_layout_bytes(rows, columns, 8)?;
        limits.check_arrow_batch_bytes(projected_bytes)?;
        let ohlcv = read_vortex_ohlcv_projection_range(
            path,
            matches!(volume_type, Some(VolumePhysicalType::Float64)),
            start..end,
        )?;
        if let Some(raw_type @ (VolumePhysicalType::UInt64 | VolumePhysicalType::Int64)) =
            volume_type
        {
            validate_vortex_raw_volume(path, raw_type, start..end, rows)?;
        }
        let timestamps = ohlcv
            .timestamp
            .as_ref()
            .context("projected Vortex batch has no timestamp field")?;
        if timestamps.len() != rows {
            bail!(
                "Vortex projection row-count mismatch: requested {rows}, got {}",
                timestamps.len()
            );
        }
        for (offset, &timestamp) in timestamps.iter().enumerate() {
            let source_row = start
                .checked_add(u64::try_from(offset)?)
                .and_then(|value| value.checked_add(1))
                .context("Vortex source row-number overflow")?;
            validate_grid_timestamp(
                timestamp,
                previous_timestamp,
                fixed_duration_ms,
                source_row,
                "vortex",
            )?;
            previous_timestamp = Some(timestamp);
            first_timestamp.get_or_insert(timestamp);
            last_timestamp = Some(timestamp);
        }
        start = end;
    }
    let timestamp_range = DatasetTimestampRange::new(
        first_timestamp.context("Vortex source has no first timestamp")?,
        last_timestamp.context("Vortex source has no last timestamp")?,
    )?;
    Ok(VerifiedVortexSource {
        row_count: metadata.row_count(),
        file_size: fs::metadata(path)?.len(),
        volume_type,
        timestamp_range,
    })
}

fn vortex_volume_physical_type(
    fields: &vortex_array::dtype::StructFields,
) -> Result<VolumePhysicalType> {
    let dtype = fields
        .field("volume")
        .context("Vortex source is missing canonical field volume")?;
    match dtype {
        DType::Primitive(PType::F64, nullability) if !nullability.is_nullable() => {
            Ok(VolumePhysicalType::Float64)
        }
        DType::Primitive(PType::U64, nullability) if !nullability.is_nullable() => {
            Ok(VolumePhysicalType::UInt64)
        }
        DType::Primitive(PType::I64, nullability) if !nullability.is_nullable() => {
            Ok(VolumePhysicalType::Int64)
        }
        DType::Primitive(PType::F32, _) => bail!(
            "Vortex field volume is Float32 and precision-unrecoverable; implicit widening is forbidden"
        ),
        other => bail!("Vortex field volume must be non-nullable f64/u64/i64, got {other}"),
    }
}

fn validate_vortex_raw_volume(
    path: &Path,
    physical: VolumePhysicalType,
    row_range: std::ops::Range<u64>,
    expected_rows: usize,
) -> Result<()> {
    let array = read_vortex_projection_range(path, &["volume"], row_range)?;
    let structure = array.to_struct();
    let volume = structure
        .unmasked_field_by_name("volume")
        .context("projected Vortex raw volume is missing")?;
    if !volume
        .all_valid()
        .context("inspect projected Vortex raw volume validity")?
    {
        bail!("projected Vortex raw volume contains null values");
    }
    let actual_rows = match physical {
        VolumePhysicalType::UInt64 => volume.to_primitive().as_slice::<u64>().len(),
        VolumePhysicalType::Int64 => {
            let primitive = volume.to_primitive();
            let values = primitive.as_slice::<i64>();
            if let Some(value) = values.iter().find(|value| **value < 0) {
                bail!("raw Int64 volume {value} is negative");
            }
            values.len()
        }
        VolumePhysicalType::Float64 => unreachable!("caller validates only integer volume"),
    };
    if actual_rows != expected_rows {
        bail!("Vortex raw volume row-count mismatch: expected {expected_rows}, got {actual_rows}");
    }
    Ok(())
}

fn preflight_vortex_footer(path: &Path, limits: &ImportLimits) -> Result<()> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let eof_size = u64::try_from(vortex_file::EOF_SIZE)?;
    if file_size < eof_size {
        bail!("Vortex source is truncated before its EOF marker");
    }
    file.seek(SeekFrom::End(-i64::try_from(vortex_file::EOF_SIZE)?))?;
    let mut eof = [0_u8; vortex_file::EOF_SIZE];
    file.read_exact(&mut eof)?;
    let version = u16::from_le_bytes(eof[0..2].try_into()?);
    if version != vortex_file::VERSION {
        bail!("unsupported Vortex file version {version}");
    }
    if eof[4..8] != vortex_file::MAGIC_BYTES {
        bail!("Vortex source has invalid EOF magic");
    }
    let postscript_bytes = usize::from(u16::from_le_bytes(eof[2..4].try_into()?));
    let postscript_with_eof = postscript_bytes
        .checked_add(vortex_file::EOF_SIZE)
        .context("Vortex postscript size overflow")?;
    limits.check_vortex_metadata_bytes(postscript_with_eof)?;
    let postscript_with_eof_u64 = u64::try_from(postscript_with_eof)?;
    if postscript_with_eof_u64 > file_size {
        bail!("Vortex postscript extends before the start of the staged source");
    }
    let postscript_offset = file_size - postscript_with_eof_u64;
    file.seek(SeekFrom::Start(postscript_offset))?;
    let mut postscript_data = vec![0_u8; postscript_bytes];
    file.read_exact(&mut postscript_data)?;
    let postscript =
        flatbuffers::root::<vortex_flatbuffers::footer::Postscript<'_>>(&postscript_data)
            .context("decode bounded Vortex postscript")?;
    let dtype = postscript
        .dtype()
        .context("canonical Vortex source has no embedded dtype segment")?;
    let layout = postscript
        .layout()
        .context("Vortex postscript has no layout segment")?;
    let footer = postscript
        .footer()
        .context("Vortex postscript has no footer segment")?;
    let mut metadata_bytes = postscript_with_eof;
    for (label, segment) in [("dtype", dtype), ("layout", layout), ("footer", footer)]
        .into_iter()
        .chain(
            postscript
                .statistics()
                .map(|segment| ("statistics", segment)),
        )
    {
        let length = usize::try_from(segment.length())?;
        limits
            .check_vortex_metadata_bytes(length)
            .with_context(|| format!("Vortex {label} segment exceeds metadata limit"))?;
        metadata_bytes = metadata_bytes
            .checked_add(length)
            .context("Vortex cumulative metadata size overflow")?;
        limits.check_vortex_metadata_bytes(metadata_bytes)?;
        let end = segment
            .offset()
            .checked_add(u64::from(segment.length()))
            .context("Vortex metadata segment range overflow")?;
        if end > postscript_offset {
            bail!("Vortex {label} segment extends into the postscript or past EOF");
        }
    }
    Ok(())
}

fn require_vortex_primitive(
    fields: &vortex_array::dtype::StructFields,
    field: &str,
    required: PType,
) -> Result<()> {
    let dtype = fields
        .field(field)
        .with_context(|| format!("Vortex source is missing canonical field {field}"))?;
    match dtype {
        DType::Primitive(actual, nullability)
            if actual == required && !nullability.is_nullable() =>
        {
            Ok(())
        }
        DType::Primitive(PType::F32, _) => bail!(
            "Vortex field {field} is Float32 and precision-unrecoverable; implicit widening is forbidden"
        ),
        other => bail!("Vortex field {field} must be non-nullable {required}, got {other}"),
    }
}

fn copy_verified_vortex_candidate(
    source_path: &Path,
    verified: VerifiedVortexSource,
    limits: &ImportLimits,
    candidate_path: &Path,
) -> Result<CandidateWriteOutcome> {
    let buffer_bytes = limits.max_vortex_chunk_bytes().min(1024 * 1024).max(1);
    let mut buffer = vec![0_u8; buffer_bytes];
    let mut source = File::open(source_path)?;
    let mut candidate = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(candidate_path)
        .with_context(|| {
            format!(
                "create independent Vortex candidate {}",
                candidate_path.display()
            )
        })?;
    let mut copied = 0_u64;
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        copied = copied
            .checked_add(u64::try_from(count)?)
            .context("Vortex source copy byte-count overflow")?;
        limits.check_candidate_bytes(copied)?;
        candidate.write_all(&buffer[..count])?;
    }
    if copied != verified.file_size {
        bail!(
            "verified Vortex source size changed: expected {}, copied {copied}",
            verified.file_size
        );
    }
    candidate.flush()?;
    candidate.sync_all()?;
    drop(candidate);
    let copied_metadata = read_vortex_file_metadata(candidate_path)
        .context("reopen independently copied Vortex candidate")?;
    if copied_metadata.row_count() != verified.row_count {
        bail!(
            "copied Vortex row-count mismatch: expected {}, got {}",
            verified.row_count,
            copied_metadata.row_count()
        );
    }
    Ok(CandidateWriteOutcome {
        write_stats: VortexWriteStats {
            row_count: verified.row_count,
            file_size: copied,
            max_buffered_bytes: u64::try_from(buffer_bytes)?,
        },
        timestamp_range: verified.timestamp_range,
    })
}

fn inspect_json_volume_mapping(
    path: &Path,
    format: ImportSourceFormat,
    limits: &ImportLimits,
) -> Result<VolumeMappingV1> {
    let mut records = JsonRecordReader::open(path, format, limits)?;
    let mut rows = 0_u64;
    let mut rows_with_volume = 0_u64;
    let mut classifier = TextVolumeClassifier::default();
    while let Some(record) = records.next_record()? {
        rows = rows.checked_add(1).context("JSON row-count overflow")?;
        limits.check_total_rows(rows)?;
        if let Some(volume) = parse_json_market_row(&record, rows)?.volume {
            classifier.observe(volume);
            rows_with_volume = rows_with_volume
                .checked_add(1)
                .context("JSON volume row-count overflow")?;
        }
    }
    if rows == 0 {
        bail!("JSON source contains no market rows");
    }
    if rows_with_volume != 0 && rows_with_volume != rows {
        bail!(
            "JSON volume is present on {rows_with_volume} of {rows} rows; mixed missing volume cannot become a canonical column"
        );
    }
    classifier.finish(rows_with_volume != 0, format.as_str())
}

fn stream_json_to_candidate(
    path: &Path,
    format: ImportSourceFormat,
    fixed_duration_ms: i64,
    volume_mapping: VolumeMappingV1,
    limits: &ImportLimits,
    candidate_path: &Path,
) -> Result<CandidateWriteOutcome> {
    let records = JsonRecordReader::open(path, format, limits)?;
    let mut chunks =
        JsonChunkReader::new(records, format, fixed_duration_ms, volume_mapping, limits);
    let write_stats = write_vortex_chunks_fallible_limited(
        candidate_path,
        &mut chunks,
        limits.max_candidate_bytes(),
    )?;
    let timestamp_range = chunks.timestamp_range()?;
    if write_stats.row_count != chunks.row_count {
        bail!(
            "JSON reader row-count mismatch: parser {}, Vortex writer {}",
            chunks.row_count,
            write_stats.row_count
        );
    }
    Ok(CandidateWriteOutcome {
        write_stats,
        timestamp_range,
    })
}

#[derive(Clone, Copy, Debug)]
enum JsonArrayState {
    Start,
    FirstValueOrEnd,
    ValueAfterComma,
    SeparatorOrEnd,
    Finished,
}

struct JsonRecordReader<'a> {
    reader: BufReader<File>,
    format: ImportSourceFormat,
    limits: &'a ImportLimits,
    array_state: JsonArrayState,
}

impl<'a> JsonRecordReader<'a> {
    fn open(path: &Path, format: ImportSourceFormat, limits: &'a ImportLimits) -> Result<Self> {
        if !matches!(
            format,
            ImportSourceFormat::JsonArray | ImportSourceFormat::JsonLines
        ) {
            bail!("{} is not a JSON input format", format.as_str());
        }
        Ok(Self {
            reader: BufReader::with_capacity(64 * 1024, File::open(path)?),
            format,
            limits,
            array_state: JsonArrayState::Start,
        })
    }

    fn next_record(&mut self) -> Result<Option<Vec<u8>>> {
        match self.format {
            ImportSourceFormat::JsonLines => self.next_json_line(),
            ImportSourceFormat::JsonArray => self.next_json_array_record(),
            _ => unreachable!("constructor rejects non-JSON formats"),
        }
    }

    fn next_json_line(&mut self) -> Result<Option<Vec<u8>>> {
        loop {
            let Some(line) = self.read_bounded_line()? else {
                return Ok(None);
            };
            let trimmed = trim_ascii_whitespace(&line);
            if trimmed.is_empty() {
                continue;
            }
            validate_json_object_structure(trimmed, self.limits)?;
            return Ok(Some(trimmed.to_vec()));
        }
    }

    fn read_bounded_line(&mut self) -> Result<Option<Vec<u8>>> {
        let mut line = Vec::new();
        loop {
            let available = self.reader.fill_buf()?;
            if available.is_empty() {
                return if line.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(line))
                };
            }
            let take = available
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(available.len(), |index| index + 1);
            let next_len = line
                .len()
                .checked_add(take)
                .context("JSON line byte-count overflow")?;
            if next_len > self.limits.max_json_object_bytes() {
                self.limits
                    .check_json_object_bytes(next_len)
                    .context("JSON line exceeds the bounded object size")?;
            }
            line.extend_from_slice(&available[..take]);
            let has_newline = available[take - 1] == b'\n';
            self.reader.consume(take);
            if has_newline {
                return Ok(Some(line));
            }
        }
    }

    fn next_json_array_record(&mut self) -> Result<Option<Vec<u8>>> {
        loop {
            match self.array_state {
                JsonArrayState::Start => {
                    let byte = self
                        .next_non_whitespace_byte()?
                        .context("JSON array source is empty")?;
                    if byte != b'[' {
                        bail!("JSON array source must begin with '['");
                    }
                    self.array_state = JsonArrayState::FirstValueOrEnd;
                }
                JsonArrayState::FirstValueOrEnd => {
                    let byte = self
                        .next_non_whitespace_byte()?
                        .context("JSON array is truncated before its closing ']'")?;
                    if byte == b']' {
                        self.require_only_trailing_whitespace()?;
                        self.array_state = JsonArrayState::Finished;
                        return Ok(None);
                    }
                    if byte != b'{' {
                        bail!("JSON array market rows must be objects");
                    }
                    self.array_state = JsonArrayState::SeparatorOrEnd;
                    return self.read_array_object(byte).map(Some);
                }
                JsonArrayState::ValueAfterComma => {
                    let byte = self
                        .next_non_whitespace_byte()?
                        .context("JSON array is truncated after a comma")?;
                    if byte == b']' {
                        bail!("JSON array has a trailing comma");
                    }
                    if byte != b'{' {
                        bail!("JSON array market rows must be objects");
                    }
                    self.array_state = JsonArrayState::SeparatorOrEnd;
                    return self.read_array_object(byte).map(Some);
                }
                JsonArrayState::SeparatorOrEnd => {
                    let byte = self
                        .next_non_whitespace_byte()?
                        .context("JSON array is truncated after an object")?;
                    match byte {
                        b',' => self.array_state = JsonArrayState::ValueAfterComma,
                        b']' => {
                            self.require_only_trailing_whitespace()?;
                            self.array_state = JsonArrayState::Finished;
                            return Ok(None);
                        }
                        _ => bail!("JSON array object must be followed by ',' or ']'"),
                    }
                }
                JsonArrayState::Finished => return Ok(None),
            }
        }
    }

    fn read_array_object(&mut self, first: u8) -> Result<Vec<u8>> {
        let mut record = vec![first];
        let mut stack = vec![b'{'];
        let mut in_string = false;
        let mut escaped = false;
        let mut string_bytes = 0_usize;
        while !stack.is_empty() {
            let byte = self
                .next_byte()?
                .context("JSON array object is truncated")?;
            let next_len = record
                .len()
                .checked_add(1)
                .context("JSON object byte-count overflow")?;
            self.limits.check_json_object_bytes(next_len)?;
            record.push(byte);
            scan_json_structure_byte(
                byte,
                &mut stack,
                &mut in_string,
                &mut escaped,
                &mut string_bytes,
                self.limits,
            )?;
        }
        validate_json_object_structure(&record, self.limits)?;
        Ok(record)
    }

    fn next_non_whitespace_byte(&mut self) -> Result<Option<u8>> {
        loop {
            match self.next_byte()? {
                Some(byte) if byte.is_ascii_whitespace() => {}
                other => return Ok(other),
            }
        }
    }

    fn next_byte(&mut self) -> Result<Option<u8>> {
        let available = self.reader.fill_buf()?;
        let Some(&byte) = available.first() else {
            return Ok(None);
        };
        self.reader.consume(1);
        Ok(Some(byte))
    }

    fn require_only_trailing_whitespace(&mut self) -> Result<()> {
        while let Some(byte) = self.next_byte()? {
            if !byte.is_ascii_whitespace() {
                bail!("JSON array has trailing non-whitespace bytes");
            }
        }
        Ok(())
    }
}

fn trim_ascii_whitespace(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

fn validate_json_object_structure(bytes: &[u8], limits: &ImportLimits) -> Result<()> {
    limits.check_json_object_bytes(bytes.len())?;
    if bytes.first() != Some(&b'{') || bytes.last() != Some(&b'}') {
        bail!("JSON market row must be exactly one object");
    }
    let mut stack = Vec::with_capacity(limits.max_json_nesting().min(64));
    let mut in_string = false;
    let mut escaped = false;
    let mut string_bytes = 0_usize;
    for &byte in bytes {
        scan_json_structure_byte(
            byte,
            &mut stack,
            &mut in_string,
            &mut escaped,
            &mut string_bytes,
            limits,
        )?;
    }
    if in_string || escaped || !stack.is_empty() {
        bail!("JSON market row is structurally incomplete");
    }
    Ok(())
}

fn scan_json_structure_byte(
    byte: u8,
    stack: &mut Vec<u8>,
    in_string: &mut bool,
    escaped: &mut bool,
    string_bytes: &mut usize,
    limits: &ImportLimits,
) -> Result<()> {
    if *in_string {
        *string_bytes = string_bytes
            .checked_add(1)
            .context("JSON string byte-count overflow")?;
        limits.check_json_string_bytes(*string_bytes)?;
        if *escaped {
            *escaped = false;
        } else if byte == b'\\' {
            *escaped = true;
        } else if byte == b'"' {
            *in_string = false;
            *string_bytes = 0;
        } else if byte < 0x20 {
            bail!("JSON string contains an unescaped control byte");
        }
        return Ok(());
    }

    match byte {
        b'"' => {
            *in_string = true;
            *string_bytes = 0;
        }
        b'{' | b'[' => {
            stack.push(byte);
            limits.check_json_nesting(stack.len())?;
        }
        b'}' => {
            if stack.pop() != Some(b'{') {
                bail!("JSON object has mismatched closing brace");
            }
        }
        b']' => {
            if stack.pop() != Some(b'[') {
                bail!("JSON object has mismatched closing bracket");
            }
        }
        _ => {}
    }
    Ok(())
}

#[derive(Debug)]
struct JsonMarketRowWire {
    timestamp: serde_json::Number,
    open: serde_json::Number,
    high: serde_json::Number,
    low: serde_json::Number,
    close: serde_json::Number,
    volume: Option<serde_json::Number>,
}

impl<'de> Deserialize<'de> for JsonMarketRowWire {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MarketRowVisitor;

        impl<'de> Visitor<'de> for MarketRowVisitor {
            type Value = JsonMarketRowWire;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("one OHLCV JSON object")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut timestamp = None;
                let mut open = None;
                let mut high = None;
                let mut low = None;
                let mut close = None;
                let mut volume = None;
                while let Some(key) = map.next_key::<String>()? {
                    let canonical = key.trim().to_ascii_lowercase();
                    let slot = match canonical.as_str() {
                        "timestamp" | "time" => Some(("timestamp", &mut timestamp)),
                        "open" | "o" => Some(("open", &mut open)),
                        "high" | "h" => Some(("high", &mut high)),
                        "low" | "l" => Some(("low", &mut low)),
                        "close" | "c" => Some(("close", &mut close)),
                        "volume" | "vol" | "v" => Some(("volume", &mut volume)),
                        _ => None,
                    };
                    if let Some((label, slot)) = slot {
                        let value = map.next_value::<serde_json::Number>()?;
                        if slot.replace(value).is_some() {
                            return Err(de::Error::custom(format!(
                                "ambiguous duplicate {label} alias in JSON object"
                            )));
                        }
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(JsonMarketRowWire {
                    timestamp: timestamp
                        .ok_or_else(|| de::Error::missing_field("timestamp/time"))?,
                    open: open.ok_or_else(|| de::Error::missing_field("open/o"))?,
                    high: high.ok_or_else(|| de::Error::missing_field("high/h"))?,
                    low: low.ok_or_else(|| de::Error::missing_field("low/l"))?,
                    close: close.ok_or_else(|| de::Error::missing_field("close/c"))?,
                    volume,
                })
            }
        }

        deserializer.deserialize_map(MarketRowVisitor)
    }
}

#[derive(Clone, Copy, Debug)]
struct JsonMarketRow {
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: Option<TextVolumeValue>,
}

fn parse_json_market_row(record: &[u8], row: u64) -> Result<JsonMarketRow> {
    let wire: JsonMarketRowWire = serde_json::from_slice(record)
        .with_context(|| format!("parse bounded JSON market row {row}"))?;
    Ok(JsonMarketRow {
        timestamp: wire
            .timestamp
            .as_i64()
            .with_context(|| format!("JSON row {row} timestamp is not an exact i64"))?,
        open: json_number_to_f64(&wire.open, row, "open")?,
        high: json_number_to_f64(&wire.high, row, "high")?,
        low: json_number_to_f64(&wire.low, row, "low")?,
        close: json_number_to_f64(&wire.close, row, "close")?,
        volume: wire
            .volume
            .as_ref()
            .map(|number| json_volume_value(number, row))
            .transpose()?,
    })
}

fn json_number_to_f64(number: &serde_json::Number, row: u64, field: &str) -> Result<f64> {
    let value = number
        .as_f64()
        .with_context(|| format!("JSON row {row} field {field} is not representable as f64"))?;
    if !value.is_finite() {
        bail!("JSON row {row} field {field} is non-finite");
    }
    Ok(value)
}

fn json_volume_value(number: &serde_json::Number, row: u64) -> Result<TextVolumeValue> {
    if let Some(value) = number.as_u64() {
        return Ok(TextVolumeValue::UInt64(value));
    }
    if let Some(value) = number.as_i64() {
        if value < 0 {
            bail!("JSON row {row} volume is negative");
        }
        return Ok(TextVolumeValue::UInt64(value as u64));
    }
    let value = json_number_to_f64(number, row, "volume")?;
    if value < 0.0 {
        bail!("JSON row {row} volume is negative");
    }
    Ok(TextVolumeValue::Float64(value))
}

struct JsonChunkReader<'a> {
    records: JsonRecordReader<'a>,
    format: ImportSourceFormat,
    fixed_duration_ms: i64,
    volume_mapping: VolumeMappingV1,
    limits: &'a ImportLimits,
    row_count: u64,
    previous_timestamp: Option<i64>,
    first_timestamp: Option<i64>,
    last_timestamp: Option<i64>,
    exhausted: bool,
}

impl<'a> JsonChunkReader<'a> {
    fn new(
        records: JsonRecordReader<'a>,
        format: ImportSourceFormat,
        fixed_duration_ms: i64,
        volume_mapping: VolumeMappingV1,
        limits: &'a ImportLimits,
    ) -> Self {
        Self {
            records,
            format,
            fixed_duration_ms,
            volume_mapping,
            limits,
            row_count: 0,
            previous_timestamp: None,
            first_timestamp: None,
            last_timestamp: None,
            exhausted: false,
        }
    }

    fn timestamp_range(&self) -> Result<DatasetTimestampRange> {
        DatasetTimestampRange::new(
            self.first_timestamp.context("JSON source has no rows")?,
            self.last_timestamp.context("JSON source has no rows")?,
        )
    }

    fn read_chunk(&mut self) -> Result<Option<ArrayRef>> {
        if self.exhausted {
            return Ok(None);
        }
        let capacity = self.limits.max_arrow_batch_rows();
        let mut timestamps = Vec::with_capacity(capacity);
        let mut open = Vec::with_capacity(capacity);
        let mut high = Vec::with_capacity(capacity);
        let mut low = Vec::with_capacity(capacity);
        let mut close = Vec::with_capacity(capacity);
        let mut volume = match &self.volume_mapping {
            VolumeMappingV1::Absent => CanonicalVolumeChunk::Absent,
            VolumeMappingV1::SourceFloat64 => {
                CanonicalVolumeChunk::Float64(Vec::with_capacity(capacity))
            }
            VolumeMappingV1::ExactUnsignedInteger { .. } => {
                CanonicalVolumeChunk::UInt64(Vec::with_capacity(capacity))
            }
            other => bail!("unsupported JSON volume mapping {other:?}"),
        };

        while timestamps.len() < capacity {
            let Some(record) = self.records.next_record()? else {
                self.exhausted = true;
                break;
            };
            let source_row = self
                .row_count
                .checked_add(1)
                .context("JSON source row-number overflow")?;
            let row = parse_json_market_row(&record, source_row)?;
            if row.volume.is_some() != !matches!(&self.volume_mapping, VolumeMappingV1::Absent) {
                bail!("JSON volume presence changed between validation and streaming");
            }
            validate_grid_timestamp(
                row.timestamp,
                self.previous_timestamp,
                self.fixed_duration_ms,
                source_row,
                self.format.as_str(),
            )?;
            timestamps.push(row.timestamp);
            open.push(row.open);
            high.push(row.high);
            low.push(row.low);
            close.push(row.close);
            append_text_volume(&mut volume, row.volume, source_row, self.format.as_str())?;
            self.row_count = source_row;
            self.limits.check_total_rows(self.row_count)?;
            self.previous_timestamp = Some(row.timestamp);
            self.first_timestamp.get_or_insert(row.timestamp);
            self.last_timestamp = Some(row.timestamp);
        }

        if timestamps.is_empty() {
            return Ok(None);
        }
        canonical_market_chunk_to_vortex_array(timestamps, open, high, low, close, volume).map(Some)
    }
}

impl Iterator for JsonChunkReader<'_> {
    type Item = Result<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_chunk().transpose()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TextSourceDetection {
    format: ImportSourceFormat,
    delimiter: Option<u8>,
}

fn detect_text_source_format(path: &Path, limits: &ImportLimits) -> Result<TextSourceDetection> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(16 * 1024, file);
    let first_non_whitespace = first_non_whitespace_text_byte(&mut reader, limits)?;
    match first_non_whitespace {
        Some(b'[') => {
            return Ok(TextSourceDetection {
                format: ImportSourceFormat::JsonArray,
                delimiter: None,
            });
        }
        Some(b'{') => {
            return Ok(TextSourceDetection {
                format: ImportSourceFormat::JsonLines,
                delimiter: None,
            });
        }
        None => bail!("text import source is empty"),
        _ => {}
    }

    reader.seek(SeekFrom::Start(0))?;
    let header = read_bounded_delimited_header(&mut reader, limits)?;
    let mut commas = 0_usize;
    let mut semicolons = 0_usize;
    let mut tabs = 0_usize;
    let mut in_quotes = false;
    let mut index = 0_usize;
    while index < header.len() {
        match header[index] {
            b'"' if in_quotes && header.get(index + 1) == Some(&b'"') => index += 1,
            b'"' => in_quotes = !in_quotes,
            b',' if !in_quotes => commas += 1,
            b';' if !in_quotes => semicolons += 1,
            b'\t' if !in_quotes => tabs += 1,
            _ => {}
        }
        index += 1;
    }
    if in_quotes {
        bail!("text header has an unterminated quoted field");
    }
    if tabs > 0 && commas == 0 && semicolons == 0 {
        return Ok(TextSourceDetection {
            format: ImportSourceFormat::Tsv,
            delimiter: Some(b'\t'),
        });
    }
    if tabs > 0 {
        bail!("text delimiter is ambiguous between TSV and CSV bytes");
    }
    let delimiter = match (commas, semicolons) {
        (0, 0) => bail!("text header has no supported delimiter"),
        (comma, semicolon) if comma == semicolon => {
            bail!("CSV delimiter is ambiguous between comma and semicolon")
        }
        (comma, semicolon) if comma > semicolon => b',',
        _ => b';',
    };
    Ok(TextSourceDetection {
        format: ImportSourceFormat::Csv,
        delimiter: Some(delimiter),
    })
}

fn first_non_whitespace_text_byte<R: BufRead>(
    reader: &mut R,
    limits: &ImportLimits,
) -> Result<Option<u8>> {
    let mut inspected = 0_u64;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(None);
        }
        let token_index = available
            .iter()
            .position(|byte| !byte.is_ascii_whitespace());
        let inspected_now = token_index.map_or(available.len(), |index| index + 1);
        inspected = inspected
            .checked_add(
                u64::try_from(inspected_now)
                    .context("text detection prefix byte-count exceeds u64")?,
            )
            .context("text detection prefix byte-count overflow")?;
        limits.check_source_bytes(inspected)?;
        if let Some(index) = token_index {
            return Ok(Some(available[index]));
        }
        reader.consume(inspected_now);
    }
}

fn read_bounded_delimited_header<R: BufRead>(
    reader: &mut R,
    limits: &ImportLimits,
) -> Result<Vec<u8>> {
    let mut header = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(header);
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        let next_len = header
            .len()
            .checked_add(take)
            .context("delimited header byte-count overflow")?;
        limits.check_csv_record_bytes(next_len)?;
        header.extend_from_slice(&available[..take]);
        let has_newline = available[take - 1] == b'\n';
        reader.consume(take);
        if has_newline {
            return Ok(header);
        }
    }
}

fn validate_delimited_structure(path: &Path, delimiter: u8, limits: &ImportLimits) -> Result<()> {
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut record_bytes = 0_usize;
    let mut field_bytes = 0_usize;
    let mut field_count = 1_usize;
    let mut in_quotes = false;
    let mut quote_maybe_close = false;
    let mut saw_any = false;

    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        for &byte in &buffer[..count] {
            saw_any = true;
            record_bytes = record_bytes
                .checked_add(1)
                .context("CSV record byte overflow")?;
            limits.check_csv_record_bytes(record_bytes)?;

            let mut structural = false;
            if in_quotes {
                if quote_maybe_close {
                    if byte == b'"' {
                        quote_maybe_close = false;
                    } else {
                        in_quotes = false;
                        quote_maybe_close = false;
                        structural = byte == delimiter || byte == b'\n';
                    }
                } else if byte == b'"' {
                    quote_maybe_close = true;
                }
            } else if byte == b'"' && field_bytes == 0 {
                in_quotes = true;
            } else {
                structural = byte == delimiter || byte == b'\n';
            }

            if structural && byte == delimiter {
                limits.check_csv_field_bytes(field_bytes)?;
                field_count = field_count
                    .checked_add(1)
                    .context("CSV field-count overflow")?;
                limits.check_csv_field_count(field_count)?;
                field_bytes = 0;
            } else if structural && byte == b'\n' {
                limits.check_csv_field_bytes(field_bytes)?;
                limits.check_csv_field_count(field_count)?;
                record_bytes = 0;
                field_bytes = 0;
                field_count = 1;
            } else {
                field_bytes = field_bytes
                    .checked_add(1)
                    .context("CSV field byte overflow")?;
                limits.check_csv_field_bytes(field_bytes)?;
            }
        }
    }
    if !saw_any {
        bail!("delimited source is empty");
    }
    if record_bytes != 0 {
        limits.check_csv_field_bytes(field_bytes)?;
        limits.check_csv_field_count(field_count)?;
    }
    Ok(())
}

fn detect_volume_mapping(
    path: &Path,
    delimiter: u8,
    limits: &ImportLimits,
) -> Result<VolumeMappingV1> {
    let (mut reader, columns) = open_delimited_reader(path, delimiter, limits)?;
    let Some(volume_index) = columns.volume else {
        return Ok(VolumeMappingV1::Absent);
    };
    let mut classifier = TextVolumeClassifier::default();
    let mut record = ByteRecord::new();
    let mut row = 2_u64;
    while reader
        .read_byte_record(&mut record)
        .with_context(|| format!("classify CSV volume at row {row}"))?
    {
        classifier.observe(parse_volume_cell(&record, volume_index, row)?);
        row = row.checked_add(1).context("CSV row-number overflow")?;
        limits.check_total_rows(row.saturating_sub(2))?;
    }
    classifier.finish(true, "CSV")
}

fn stream_delimited_to_candidate(
    path: &Path,
    delimiter: u8,
    fixed_duration_ms: i64,
    volume_mapping: VolumeMappingV1,
    limits: &ImportLimits,
    candidate_path: &Path,
) -> Result<CandidateWriteOutcome> {
    let (reader, columns) = open_delimited_reader(path, delimiter, limits)?;
    let mut chunks =
        DelimitedChunkReader::new(reader, columns, fixed_duration_ms, volume_mapping, limits);
    let write_stats = write_vortex_chunks_fallible_limited(
        candidate_path,
        &mut chunks,
        limits.max_candidate_bytes(),
    )?;
    let timestamp_range = chunks.timestamp_range()?;
    if write_stats.row_count != chunks.row_count {
        bail!(
            "streaming CSV row-count mismatch: parser {}, Vortex writer {}",
            chunks.row_count,
            write_stats.row_count
        );
    }
    Ok(CandidateWriteOutcome {
        write_stats,
        timestamp_range,
    })
}

fn open_delimited_reader(
    path: &Path,
    delimiter: u8,
    limits: &ImportLimits,
) -> Result<(Reader<File>, ColumnMap)> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .flexible(false)
        .from_path(path)
        .with_context(|| format!("open staged delimited source {}", path.display()))?;
    let headers = reader.headers().context("read delimited header")?.clone();
    limits.check_csv_field_count(headers.len())?;
    let columns = ColumnMap::from_headers(&headers)?;
    Ok((reader, columns))
}

#[derive(Clone, Copy, Debug)]
struct ColumnMap {
    timestamp: usize,
    open: usize,
    high: usize,
    low: usize,
    close: usize,
    volume: Option<usize>,
}

impl ColumnMap {
    fn from_headers(headers: &csv::StringRecord) -> Result<Self> {
        let mut timestamp = None;
        let mut open = None;
        let mut high = None;
        let mut low = None;
        let mut close = None;
        let mut volume = None;
        for (index, header) in headers.iter().enumerate() {
            let canonical = match header.trim().to_ascii_lowercase().as_str() {
                "timestamp" | "time" => Some(("timestamp", &mut timestamp)),
                "open" | "o" => Some(("open", &mut open)),
                "high" | "h" => Some(("high", &mut high)),
                "low" | "l" => Some(("low", &mut low)),
                "close" | "c" => Some(("close", &mut close)),
                "volume" | "vol" | "v" => Some(("volume", &mut volume)),
                _ => None,
            };
            if let Some((name, slot)) = canonical {
                if slot.replace(index).is_some() {
                    bail!("ambiguous duplicate {name} alias in source header");
                }
            }
        }
        Ok(Self {
            timestamp: timestamp.context("source header has no timestamp/time column")?,
            open: open.context("source header has no open/o column")?,
            high: high.context("source header has no high/h column")?,
            low: low.context("source header has no low/l column")?,
            close: close.context("source header has no close/c column")?,
            volume,
        })
    }
}

struct DelimitedChunkReader<'a> {
    reader: Reader<File>,
    columns: ColumnMap,
    fixed_duration_ms: i64,
    volume_mapping: VolumeMappingV1,
    limits: &'a ImportLimits,
    row_count: u64,
    previous_timestamp: Option<i64>,
    first_timestamp: Option<i64>,
    last_timestamp: Option<i64>,
    exhausted: bool,
}

impl<'a> DelimitedChunkReader<'a> {
    fn new(
        reader: Reader<File>,
        columns: ColumnMap,
        fixed_duration_ms: i64,
        volume_mapping: VolumeMappingV1,
        limits: &'a ImportLimits,
    ) -> Self {
        Self {
            reader,
            columns,
            fixed_duration_ms,
            volume_mapping,
            limits,
            row_count: 0,
            previous_timestamp: None,
            first_timestamp: None,
            last_timestamp: None,
            exhausted: false,
        }
    }

    fn timestamp_range(&self) -> Result<DatasetTimestampRange> {
        DatasetTimestampRange::new(
            self.first_timestamp
                .context("import source has no data rows")?,
            self.last_timestamp
                .context("import source has no data rows")?,
        )
    }

    fn read_chunk(&mut self) -> Result<Option<ArrayRef>> {
        if self.exhausted {
            return Ok(None);
        }
        let capacity = self.limits.max_arrow_batch_rows();
        let mut timestamps = Vec::with_capacity(capacity);
        let mut open = Vec::with_capacity(capacity);
        let mut high = Vec::with_capacity(capacity);
        let mut low = Vec::with_capacity(capacity);
        let mut close = Vec::with_capacity(capacity);
        let mut volume = match &self.volume_mapping {
            VolumeMappingV1::Absent => CanonicalVolumeChunk::Absent,
            VolumeMappingV1::SourceFloat64 => {
                CanonicalVolumeChunk::Float64(Vec::with_capacity(capacity))
            }
            VolumeMappingV1::ExactUnsignedInteger { .. } => {
                CanonicalVolumeChunk::UInt64(Vec::with_capacity(capacity))
            }
            other => bail!("unsupported delimited volume mapping {other:?}"),
        };
        let mut record = ByteRecord::new();

        while timestamps.len() < capacity {
            let has_row = self
                .reader
                .read_byte_record(&mut record)
                .with_context(|| format!("parse CSV data row {}", self.row_count + 2))?;
            if !has_row {
                self.exhausted = true;
                break;
            }
            let source_row = self.row_count + 2;
            let timestamp =
                parse_i64_cell(&record, self.columns.timestamp, source_row, "timestamp")?;
            self.validate_timestamp(timestamp, source_row)?;
            let row_open = parse_f64_cell(&record, self.columns.open, source_row, "open")?;
            let row_high = parse_f64_cell(&record, self.columns.high, source_row, "high")?;
            let row_low = parse_f64_cell(&record, self.columns.low, source_row, "low")?;
            let row_close = parse_f64_cell(&record, self.columns.close, source_row, "close")?;
            let row_volume = self
                .columns
                .volume
                .map(|index| parse_volume_cell(&record, index, source_row))
                .transpose()?;

            timestamps.push(timestamp);
            open.push(row_open);
            high.push(row_high);
            low.push(row_low);
            close.push(row_close);
            append_text_volume(&mut volume, row_volume, source_row, "CSV")?;
            self.row_count = self
                .row_count
                .checked_add(1)
                .context("CSV row-count overflow")?;
            self.limits.check_total_rows(self.row_count)?;
            self.first_timestamp.get_or_insert(timestamp);
            self.last_timestamp = Some(timestamp);
            self.previous_timestamp = Some(timestamp);
        }

        if timestamps.is_empty() {
            return Ok(None);
        }
        canonical_market_chunk_to_vortex_array(timestamps, open, high, low, close, volume).map(Some)
    }

    fn validate_timestamp(&self, timestamp: i64, source_row: u64) -> Result<()> {
        if !(MIN_CANONICAL_MARKET_TIMESTAMP_MS..=MAX_CANONICAL_MARKET_TIMESTAMP_MS)
            .contains(&timestamp)
        {
            bail!(
                "CSV row {source_row} timestamp is not an in-range i64 Unix millisecond value: {timestamp}"
            );
        }
        if timestamp.rem_euclid(self.fixed_duration_ms) != 0 {
            bail!(
                "CSV row {source_row} timestamp {timestamp} is off the declared {} ms bar-open grid",
                self.fixed_duration_ms
            );
        }
        if let Some(previous) = self.previous_timestamp {
            if timestamp <= previous {
                bail!(
                    "CSV row {source_row} timestamp {timestamp} is not strictly after {previous}"
                );
            }
            let gap = timestamp - previous;
            if gap.rem_euclid(self.fixed_duration_ms) != 0 {
                bail!(
                    "CSV row {source_row} timestamp gap {gap} is not a legal multiple of {} ms",
                    self.fixed_duration_ms
                );
            }
        }
        Ok(())
    }
}

impl Iterator for DelimitedChunkReader<'_> {
    type Item = Result<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_chunk().transpose()
    }
}

fn parse_i64_cell(record: &ByteRecord, index: usize, row: u64, field: &str) -> Result<i64> {
    let bytes = record
        .get(index)
        .with_context(|| format!("CSV row {row} has no field {field}"))?;
    let text = std::str::from_utf8(bytes)
        .with_context(|| format!("CSV row {row} field {field} is not UTF-8"))?
        .trim();
    if text.is_empty() {
        bail!("CSV row {row} field {field} is empty");
    }
    text.parse::<i64>()
        .with_context(|| format!("CSV row {row} field {field} is not an exact i64"))
}

fn parse_f64_cell(record: &ByteRecord, index: usize, row: u64, field: &str) -> Result<f64> {
    let bytes = record
        .get(index)
        .with_context(|| format!("CSV row {row} has no field {field}"))?;
    let text = std::str::from_utf8(bytes)
        .with_context(|| format!("CSV row {row} field {field} is not UTF-8"))?
        .trim();
    if text.is_empty() {
        bail!("CSV row {row} field {field} is empty");
    }
    let value = text
        .parse::<f64>()
        .with_context(|| format!("CSV row {row} field {field} is not a direct Rust f64"))?;
    if !value.is_finite() {
        bail!("CSV row {row} field {field} is non-finite");
    }
    Ok(value)
}

fn parse_volume_cell(record: &ByteRecord, index: usize, row: u64) -> Result<TextVolumeValue> {
    let bytes = record
        .get(index)
        .with_context(|| format!("CSV row {row} has no field volume"))?;
    let text = std::str::from_utf8(bytes)
        .with_context(|| format!("CSV row {row} field volume is not UTF-8"))?
        .trim();
    if text.is_empty() {
        bail!("CSV row {row} field volume is empty");
    }

    let unsigned = text.strip_prefix('+').unwrap_or(text);
    if !unsigned.is_empty() && unsigned.bytes().all(|byte| byte.is_ascii_digit()) {
        let value = unsigned
            .parse::<u64>()
            .with_context(|| format!("CSV row {row} raw integer volume is outside UInt64"))?;
        return Ok(TextVolumeValue::UInt64(value));
    }
    if let Some(magnitude) = text.strip_prefix('-')
        && !magnitude.is_empty()
        && magnitude.bytes().all(|byte| byte.is_ascii_digit())
    {
        bail!("CSV row {row} field volume is negative");
    }

    let value = text
        .parse::<f64>()
        .with_context(|| format!("CSV row {row} field volume is not a direct Rust f64"))?;
    if !value.is_finite() {
        bail!("CSV row {row} field volume is non-finite");
    }
    if value < 0.0 {
        bail!("CSV row {row} field volume is negative");
    }
    Ok(TextVolumeValue::Float64(value))
}

fn unix_time_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_millis()
        .try_into()
        .context("import timestamp does not fit u64")?)
}
