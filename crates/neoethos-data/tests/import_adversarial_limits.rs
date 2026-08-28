use std::fs::{self, File};
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_ipc::writer::{FileWriter as IpcFileWriter, StreamWriter as IpcStreamWriter};
use arrow_schema::{DataType, Field, Schema};
use flatbuffers::FlatBufferBuilder;
use neoethos_data::core::dataset_manifest::read_current_manifest;
use neoethos_data::core::import_limits::{ImportLimitError, ImportLimitKind, ImportLimits};
use neoethos_data::core::import_provenance::ImportSourceFormat;
use neoethos_data::core::import_service::{ImportRequest, import_path_to_vortex};
use neoethos_data::{BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

mod common;

fn assert_route_limit(
    source: &std::path::Path,
    root: &std::path::Path,
    format: ImportSourceFormat,
    expected: ImportLimitKind,
) {
    assert_route_limit_with_limits(
        source,
        root,
        format,
        expected,
        &ImportLimits::conservative_for_tests(),
    );
}

fn assert_route_limit_with_limits(
    source: &std::path::Path,
    root: &std::path::Path,
    format: ImportSourceFormat,
    expected: ImportLimitKind,
    limits: &ImportLimits,
) {
    let identity = CanonicalDatasetIdentity::external(
        format!("adversarial-{}", format.as_str()),
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let grant = common::import_grant();
    let error = import_path_to_vortex(ImportRequest {
        source_path: source,
        configured_root: root,
        identity: &identity,
        declared_format: format,
        expected_generation: None,
        limits,
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect_err("adversarial source must fail before publication");
    let limit = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<ImportLimitError>())
        .unwrap_or_else(|| panic!("expected typed {expected:?} limit, got {error:#}"));
    assert_eq!(limit.kind(), expected, "{error:#}");
    assert!(read_current_manifest(root, &identity).is_err());
}

#[test]
fn checked_layout_arithmetic_rejects_overflow_before_allocation() {
    let limits = ImportLimits::conservative_for_tests();
    let error = limits
        .checked_layout_bytes(usize::MAX, 8, 8)
        .expect_err("overflow must fail");
    assert_eq!(error.kind(), ImportLimitKind::CheckedArithmetic);
}

#[test]
fn oversized_single_records_tokens_and_binary_headers_name_the_boundary() {
    let limits = ImportLimits::conservative_for_tests();
    let cases = [
        (
            limits.check_csv_record_bytes(limits.max_csv_record_bytes() + 1),
            ImportLimitKind::CsvRecordBytes,
        ),
        (
            limits.check_json_string_bytes(limits.max_json_string_bytes() + 1),
            ImportLimitKind::JsonStringBytes,
        ),
        (
            limits.check_arrow_message_bytes(limits.max_arrow_message_bytes() + 1),
            ImportLimitKind::ArrowMessageBytes,
        ),
        (
            limits.check_parquet_footer_bytes(limits.max_parquet_footer_bytes() + 1),
            ImportLimitKind::ParquetFooterBytes,
        ),
        (
            limits.check_vortex_metadata_bytes(limits.max_vortex_metadata_bytes() + 1),
            ImportLimitKind::VortexMetadataBytes,
        ),
    ];

    for (result, expected_kind) in cases {
        let error = result.expect_err("limit must reject one byte over its ceiling");
        assert_eq!(error.kind(), expected_kind);
    }
}

#[test]
fn oversized_csv_and_json_values_fail_through_the_real_import_route() {
    let temp = tempfile::tempdir().expect("tempdir");
    let limits = ImportLimits::conservative_for_tests();

    let csv = temp.path().join("oversized.csv");
    let oversized_field = "1".repeat(limits.max_csv_field_bytes() + 1);
    fs::write(
        &csv,
        format!(
            "timestamp,open,high,low,close,volume\n1700000040000,{oversized_field},1.2,1.0,1.1,0\n"
        ),
    )
    .expect("write CSV bomb");
    assert_route_limit(
        &csv,
        &temp.path().join("csv-root"),
        ImportSourceFormat::Csv,
        ImportLimitKind::CsvFieldBytes,
    );

    let csv_record = temp.path().join("oversized-record.csv");
    let maximum_field = "1".repeat(limits.max_csv_field_bytes());
    fs::write(
        &csv_record,
        format!(
            "timestamp,open,high,low,close,volume\n1700000040000,{maximum_field},{maximum_field},{maximum_field},{maximum_field},{maximum_field}\n"
        ),
    )
    .expect("write CSV record bomb");
    assert_route_limit(
        &csv_record,
        &temp.path().join("csv-record-root"),
        ImportSourceFormat::Csv,
        ImportLimitKind::CsvRecordBytes,
    );

    let json_string = temp.path().join("oversized.jsonl");
    let giant = "x".repeat(limits.max_json_string_bytes() + 1);
    fs::write(
        &json_string,
        format!(
            "{{\"timestamp\":1700000040000,\"open\":1.1,\"high\":1.2,\"low\":1.0,\"close\":1.1,\"volume\":0,\"unknown\":\"{giant}\"}}\n"
        ),
    )
    .expect("write JSON string bomb");
    assert_route_limit(
        &json_string,
        &temp.path().join("json-string-root"),
        ImportSourceFormat::JsonLines,
        ImportLimitKind::JsonStringBytes,
    );

    let nested_json = temp.path().join("nested.json");
    let nesting = "[".repeat(limits.max_json_nesting() + 1);
    let closing = "]".repeat(limits.max_json_nesting() + 1);
    fs::write(
        &nested_json,
        format!(
            "[{{\"timestamp\":1700000040000,\"open\":1.1,\"high\":1.2,\"low\":1.0,\"close\":1.1,\"volume\":0,\"unknown\":{nesting}0{closing}}}]"
        ),
    )
    .expect("write JSON nesting bomb");
    assert_route_limit(
        &nested_json,
        &temp.path().join("json-nesting-root"),
        ImportSourceFormat::JsonArray,
        ImportLimitKind::JsonNesting,
    );
}

#[test]
fn oversized_parquet_footer_and_ipc_message_fail_before_reader_allocation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let limits = ImportLimits::conservative_for_tests();

    let parquet = temp.path().join("oversized-footer.parquet");
    let footer_bytes = limits.max_parquet_footer_bytes() + 1;
    let mut parquet_bytes = vec![0_u8; footer_bytes + 8];
    parquet_bytes[..4].copy_from_slice(b"PAR1");
    let trailer = parquet_bytes.len() - 8;
    parquet_bytes[trailer..trailer + 4].copy_from_slice(
        &u32::try_from(footer_bytes)
            .expect("test footer size")
            .to_le_bytes(),
    );
    parquet_bytes[trailer + 4..].copy_from_slice(b"PAR1");
    fs::write(&parquet, parquet_bytes).expect("write Parquet footer bomb");
    assert_route_limit(
        &parquet,
        &temp.path().join("parquet-root"),
        ImportSourceFormat::Parquet,
        ImportLimitKind::ParquetFooterBytes,
    );

    let ipc = temp.path().join("oversized.arrow-stream");
    let metadata_bytes = limits.max_arrow_metadata_bytes() + 1;
    fs::write(
        &ipc,
        i32::try_from(metadata_bytes)
            .expect("test metadata size")
            .to_le_bytes(),
    )
    .expect("write IPC metadata bomb");
    assert_route_limit(
        &ipc,
        &temp.path().join("ipc-root"),
        ImportSourceFormat::ArrowIpcStream,
        ImportLimitKind::ArrowMetadataBytes,
    );
}

#[test]
fn oversized_vortex_segment_fails_before_the_vortex_reader_allocates() {
    use vortex_flatbuffers::footer::{
        Postscript, PostscriptArgs, PostscriptSegment, PostscriptSegmentArgs,
    };

    let temp = tempfile::tempdir().expect("tempdir");
    let limits = ImportLimits::conservative_for_tests();
    let oversized_length = u32::try_from(limits.max_vortex_metadata_bytes() + 1)
        .expect("test Vortex metadata limit fits u32");

    let mut builder = FlatBufferBuilder::new();
    let oversized_dtype = PostscriptSegment::create(
        &mut builder,
        &PostscriptSegmentArgs {
            length: oversized_length,
            ..Default::default()
        },
    );
    let empty_layout = PostscriptSegment::create(&mut builder, &PostscriptSegmentArgs::default());
    let empty_footer = PostscriptSegment::create(&mut builder, &PostscriptSegmentArgs::default());
    let postscript = Postscript::create(
        &mut builder,
        &PostscriptArgs {
            dtype: Some(oversized_dtype),
            layout: Some(empty_layout),
            footer: Some(empty_footer),
            ..Default::default()
        },
    );
    builder.finish(postscript, None);
    let postscript_bytes = builder.finished_data();
    let postscript_len = u16::try_from(postscript_bytes.len()).expect("bounded test postscript");
    let mut vortex_bytes = Vec::with_capacity(postscript_bytes.len() + vortex_file::EOF_SIZE);
    vortex_bytes.extend_from_slice(postscript_bytes);
    vortex_bytes.extend_from_slice(&vortex_file::VERSION.to_le_bytes());
    vortex_bytes.extend_from_slice(&postscript_len.to_le_bytes());
    vortex_bytes.extend_from_slice(&vortex_file::MAGIC_BYTES);

    let source = temp.path().join("oversized-metadata.vortex");
    fs::write(&source, vortex_bytes).expect("write Vortex metadata bomb");
    assert_route_limit(
        &source,
        &temp.path().join("vortex-root"),
        ImportSourceFormat::Vortex,
        ImportLimitKind::VortexMetadataBytes,
    );
}

fn large_ipc_batch(rows: usize) -> RecordBatch {
    let base = 1_700_000_040_000_i64;
    let timestamps = (0..rows)
        .map(|row| base + i64::try_from(row).expect("row index") * 60_000)
        .collect::<Vec<_>>();
    let prices = vec![1.1_f64; rows];
    let fields = vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(timestamps)),
        Arc::new(Float64Array::from(prices.clone())),
        Arc::new(Float64Array::from(vec![1.2; rows])),
        Arc::new(Float64Array::from(vec![1.0; rows])),
        Arc::new(Float64Array::from(prices)),
        Arc::new(Float64Array::from(vec![0.0; rows])),
    ];
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("large IPC batch")
}

#[test]
fn ipc_record_batch_rows_and_body_are_bounded_before_decode() {
    let temp = tempfile::tempdir().expect("tempdir");
    let limits = ImportLimits::conservative_for_tests();

    let oversized_rows = large_ipc_batch(limits.max_arrow_batch_rows() + 1);
    let file_path = temp.path().join("oversized-rows.arrow");
    let file = File::create(&file_path).expect("create IPC file");
    let mut file_writer =
        IpcFileWriter::try_new(file, oversized_rows.schema().as_ref()).expect("IPC file writer");
    file_writer
        .write(&oversized_rows)
        .expect("write IPC file batch");
    file_writer.finish().expect("finish IPC file");
    drop(file_writer);
    assert_route_limit(
        &file_path,
        &temp.path().join("ipc-row-root"),
        ImportSourceFormat::ArrowIpcFile,
        ImportLimitKind::ArrowBatchRows,
    );

    let body_rows = limits
        .max_arrow_body_bytes()
        .checked_div(6 * std::mem::size_of::<f64>())
        .expect("positive Arrow row width")
        + 1_024;
    let oversized_body = large_ipc_batch(body_rows);
    let stream_path = temp.path().join("oversized-body.stream");
    let file = File::create(&stream_path).expect("create IPC stream");
    let mut stream_writer = IpcStreamWriter::try_new(file, oversized_body.schema().as_ref())
        .expect("IPC stream writer");
    stream_writer
        .write(&oversized_body)
        .expect("write IPC stream batch");
    stream_writer.finish().expect("finish IPC stream");
    drop(stream_writer);
    assert_route_limit(
        &stream_path,
        &temp.path().join("ipc-body-root"),
        ImportSourceFormat::ArrowIpcStream,
        ImportLimitKind::ArrowBodyBytes,
    );
}

fn write_parquet_with_properties(
    path: &std::path::Path,
    batch: &RecordBatch,
    properties: WriterProperties,
) {
    let file = File::create(path).expect("create adversarial Parquet");
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(properties))
        .expect("adversarial Parquet writer");
    writer
        .write(batch)
        .expect("write adversarial Parquet batch");
    writer.close().expect("close adversarial Parquet");
}

#[test]
fn parquet_page_dictionary_and_compression_ratio_are_preflight_bounded() {
    let temp = tempfile::tempdir().expect("tempdir");
    let page_limits = ImportLimits::conservative_for_tests().with_parquet_decode_bounds(
        16 * 1024 * 1024,
        256 * 1024,
        4 * 1024 * 1024,
        32 * 1024 * 1024,
        100,
    );

    let page_rows =
        page_limits.max_parquet_page_bytes() as usize / std::mem::size_of::<f64>() + 4_096;
    let page_batch = large_ipc_batch(page_rows);
    let page_path = temp.path().join("oversized-page.parquet");
    write_parquet_with_properties(
        &page_path,
        &page_batch,
        WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_data_page_size_limit(1024 * 1024)
            .set_max_row_group_row_count(Some(page_rows))
            .build(),
    );
    assert_route_limit_with_limits(
        &page_path,
        &temp.path().join("parquet-page-root"),
        ImportSourceFormat::Parquet,
        ImportLimitKind::ParquetPageBytes,
        &page_limits,
    );

    let dictionary_limits = ImportLimits::conservative_for_tests().with_parquet_decode_bounds(
        16 * 1024 * 1024,
        2 * 1024 * 1024,
        512 * 1024,
        32 * 1024 * 1024,
        100,
    );
    let dictionary_rows = 8_000;
    let base_batch = large_ipc_batch(dictionary_rows);
    let mut fields = base_batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields.push(Field::new("untrusted_dictionary", DataType::Utf8, false));
    let mut columns = base_batch.columns().to_vec();
    let padding = "x".repeat(120);
    let dictionary_values = (0..dictionary_rows)
        .map(|row| format!("{row:08}-{padding}"))
        .collect::<Vec<_>>();
    columns.push(Arc::new(StringArray::from(dictionary_values)) as ArrayRef);
    let dictionary_batch =
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("dictionary batch");
    let dictionary_path = temp.path().join("oversized-dictionary.parquet");
    write_parquet_with_properties(
        &dictionary_path,
        &dictionary_batch,
        WriterProperties::builder()
            .set_dictionary_enabled(true)
            .set_dictionary_page_size_limit(
                dictionary_limits.max_parquet_dictionary_bytes() as usize + 256 * 1024,
            )
            .set_data_page_size_limit(1024 * 1024)
            .set_max_row_group_row_count(Some(dictionary_rows))
            .build(),
    );
    assert_route_limit_with_limits(
        &dictionary_path,
        &temp.path().join("parquet-dictionary-root"),
        ImportSourceFormat::Parquet,
        ImportLimitKind::ParquetDictionaryBytes,
        &dictionary_limits,
    );

    let compression_limits = ImportLimits::conservative_for_tests().with_parquet_decode_bounds(
        16 * 1024 * 1024,
        2 * 1024 * 1024,
        2 * 1024 * 1024,
        32 * 1024 * 1024,
        100,
    );
    let compressed_rows = 100_000;
    let compressed_batch = large_ipc_batch(compressed_rows);
    let compressed_path = temp.path().join("compression-bomb.parquet");
    write_parquet_with_properties(
        &compressed_path,
        &compressed_batch,
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .set_dictionary_enabled(false)
            .set_max_row_group_row_count(Some(compressed_rows))
            .build(),
    );
    assert_route_limit_with_limits(
        &compressed_path,
        &temp.path().join("parquet-compression-root"),
        ImportSourceFormat::Parquet,
        ImportLimitKind::ParquetCompressionRatio,
        &compression_limits,
    );
}
