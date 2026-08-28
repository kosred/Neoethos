use std::fs::{self, File};
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, UInt64Array};
use arrow_ipc::writer::{FileWriter as IpcFileWriter, StreamWriter as IpcStreamWriter};
use arrow_schema::{DataType, Field, Schema};
use neoethos_data::core::dataset_manifest::read_current_manifest;
use neoethos_data::core::import_limits::ImportLimits;
use neoethos_data::core::import_provenance::{ImportProvenanceV1, ImportSourceFormat};
use neoethos_data::core::import_service::{ImportRequest, import_path_to_vortex};
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe, load_vortex,
    read_vortex_array,
};
use parquet::arrow::ArrowWriter;
use vortex_array::arrays::{PrimitiveArray, StructArray};
use vortex_array::{IntoArray, ToCanonical};

mod common;

#[test]
fn csv_high_precision_round_trip_publishes_an_independent_verified_generation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("operator.csv");
    let root = temp.path().join("canonical");
    let rows = [
        (1_700_000_040_000_i64, 1.000_000_000_000_000_2_f64),
        (1_700_000_100_000_i64, 1.234_567_890_123_456_7_f64),
        (1_700_000_160_000_i64, 1.234_567_990_123_456_8_f64),
    ];
    let csv = format!(
        "time,o,h,l,c,vol\n{},{:.17},{:.17},{:.17},{:.17},0\n{},{:.17},{:.17},{:.17},{:.17},16777217\n{},{:.17},{:.17},{:.17},{:.17},3\n",
        rows[0].0,
        rows[0].1,
        rows[0].1 + 0.000_2,
        rows[0].1 - 0.000_2,
        rows[0].1 + 0.000_1,
        rows[1].0,
        rows[1].1,
        rows[1].1 + 0.000_2,
        rows[1].1 - 0.000_2,
        rows[1].1 + 0.000_1,
        rows[2].0,
        rows[2].1,
        rows[2].1 + 0.000_2,
        rows[2].1 - 0.000_2,
        rows[2].1 + 0.000_1,
    );
    fs::write(&source, csv.as_bytes()).expect("write source");
    let identity = CanonicalDatasetIdentity::external(
        "operator-upload",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");

    let grant = common::import_grant();
    let result = import_path_to_vortex(ImportRequest {
        source_path: &source,
        configured_root: &root,
        identity: &identity,
        declared_format: ImportSourceFormat::Csv,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect("strict import");
    assert_eq!(result.row_count(), rows.len() as u64);

    let manifest = read_current_manifest(&root, &identity).expect("published manifest");
    let provenance =
        ImportProvenanceV1::from_envelope(manifest.provenance()).expect("typed import provenance");
    assert_eq!(provenance.source_size(), csv.len() as u64);
    assert_eq!(provenance.detected_format(), ImportSourceFormat::Csv);

    let loaded = load_vortex(manifest.generation_path()).expect("reopen generation");
    for (actual, (_, expected)) in loaded.open.iter().zip(rows) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(
        loaded.volume.as_deref(),
        Some(&[0.0, 16_777_217.0, 3.0][..])
    );

    fs::write(&source, b"destroyed original").expect("overwrite source");
    fs::remove_file(&source).expect("remove source");
    let reopened = load_vortex(manifest.generation_path()).expect("canonical file is independent");
    assert_eq!(reopened.open, loaded.open);
}

#[test]
fn tsv_json_array_and_json_lines_round_trip_exact_f64_bits() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rows = [
        (1_700_000_040_000_i64, 1.000_000_000_000_000_2_f64),
        (1_700_000_100_000_i64, 1.234_567_890_123_456_7_f64),
        (1_700_000_160_000_i64, 1.234_567_990_123_456_8_f64),
    ];
    let object = |(timestamp, open): (i64, f64)| {
        format!(
            "{{\"timestamp\":{timestamp},\"open\":{open:.17},\"high\":{:.17},\"low\":{:.17},\"close\":{:.17},\"volume\":16777217}}",
            open + 0.000_2,
            open - 0.000_2,
            open + 0.000_1,
        )
    };
    let json_records = rows.map(object);
    let cases = [
        (
            "operator.tsv",
            ImportSourceFormat::Tsv,
            format!(
                "timestamp\topen\thigh\tlow\tclose\tvolume\n{}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t16777217\n{}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t16777217\n{}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t16777217\n",
                rows[0].0,
                rows[0].1,
                rows[0].1 + 0.000_2,
                rows[0].1 - 0.000_2,
                rows[0].1 + 0.000_1,
                rows[1].0,
                rows[1].1,
                rows[1].1 + 0.000_2,
                rows[1].1 - 0.000_2,
                rows[1].1 + 0.000_1,
                rows[2].0,
                rows[2].1,
                rows[2].1 + 0.000_2,
                rows[2].1 - 0.000_2,
                rows[2].1 + 0.000_1,
            ),
        ),
        (
            "operator.json",
            ImportSourceFormat::JsonArray,
            format!(
                "[\n  {},\n  {},\n  {}\n]\n",
                json_records[0], json_records[1], json_records[2]
            ),
        ),
        (
            "operator.jsonl",
            ImportSourceFormat::JsonLines,
            format!(
                "{}\n{}\n{}\n",
                json_records[0], json_records[1], json_records[2]
            ),
        ),
    ];

    let grant = common::import_grant();
    for (index, (name, format, contents)) in cases.into_iter().enumerate() {
        let source = temp.path().join(name);
        fs::write(&source, contents).expect("write source format fixture");
        let root = temp.path().join(format!("text-canonical-{index}"));
        let identity = CanonicalDatasetIdentity::external(
            format!("text-{index}"),
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("identity");
        import_path_to_vortex(ImportRequest {
            source_path: &source,
            configured_root: &root,
            identity: &identity,
            declared_format: format,
            expected_generation: None,
            limits: &ImportLimits::conservative_for_tests(),
            auxiliary_slot: grant
                .auxiliary_slot()
                .expect("import grant owns a source-seal slot"),
        })
        .unwrap_or_else(|error| panic!("{format:?} import failed: {error:#}"));

        let manifest = read_current_manifest(&root, &identity).expect("manifest");
        let provenance =
            ImportProvenanceV1::from_envelope(manifest.provenance()).expect("provenance");
        assert_eq!(provenance.detected_format(), format);
        let loaded = load_vortex(manifest.generation_path()).expect("reopen");
        assert_eq!(loaded.open.len(), rows.len());
        for (actual, (_, expected)) in loaded.open.iter().zip(rows) {
            assert_eq!(actual.to_bits(), expected.to_bits(), "format {format:?}");
        }
    }
}

fn binary_fixture_with_volume(
    float32_prices: bool,
    volume_type: DataType,
    volume: ArrayRef,
) -> RecordBatch {
    let timestamps = Arc::new(Int64Array::from(vec![
        1_700_000_040_000_i64,
        1_700_000_100_000_i64,
        1_700_000_160_000_i64,
    ])) as ArrayRef;
    let values = [
        1.000_000_000_000_000_2_f64,
        1.234_567_890_123_456_7_f64,
        1.234_567_990_123_456_8_f64,
    ];
    let mut fields = vec![Field::new("timestamp", DataType::Int64, false)];
    let mut columns = vec![timestamps];
    for (name, offset) in [
        ("open", 0.0),
        ("high", 0.000_2),
        ("low", -0.000_2),
        ("close", 0.000_1),
    ] {
        if float32_prices {
            fields.push(Field::new(name, DataType::Float32, false));
            columns.push(Arc::new(Float32Array::from(
                values.map(|value| (value + offset) as f32).to_vec(),
            )) as ArrayRef);
        } else {
            fields.push(Field::new(name, DataType::Float64, false));
            columns.push(Arc::new(Float64Array::from(
                values.map(|value| value + offset).to_vec(),
            )) as ArrayRef);
        }
    }
    fields.push(Field::new("volume", volume_type, false));
    columns.push(volume);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("binary fixture batch")
}

fn binary_fixture(float32_prices: bool) -> RecordBatch {
    binary_fixture_with_volume(
        float32_prices,
        DataType::Float64,
        Arc::new(Float64Array::from(vec![0.0, 16_777_217.0, 3.0])) as ArrayRef,
    )
}

fn write_parquet(path: &std::path::Path, batch: &RecordBatch) {
    let file = File::create(path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).expect("parquet writer");
    writer.write(batch).expect("write parquet batch");
    writer.close().expect("close parquet");
}

fn write_ipc_file(path: &std::path::Path, batch: &RecordBatch) {
    let file = File::create(path).expect("create IPC file");
    let mut writer = IpcFileWriter::try_new(file, batch.schema().as_ref()).expect("IPC writer");
    writer.write(batch).expect("write IPC batch");
    writer.finish().expect("finish IPC file");
}

fn write_ipc_stream(path: &std::path::Path, batch: &RecordBatch) {
    let file = File::create(path).expect("create IPC stream");
    let mut writer = IpcStreamWriter::try_new(file, batch.schema().as_ref()).expect("IPC writer");
    writer.write(batch).expect("write IPC batch");
    writer.finish().expect("finish IPC stream");
}

type BinaryFixtureWriter = fn(&std::path::Path, &RecordBatch);

#[test]
fn parquet_and_arrow_ipc_float64_round_trip_exact_bits() {
    let temp = tempfile::tempdir().expect("tempdir");
    let batch = binary_fixture(false);
    let cases: [(&str, ImportSourceFormat, BinaryFixtureWriter); 3] = [
        ("data.parquet", ImportSourceFormat::Parquet, write_parquet),
        (
            "data.arrow",
            ImportSourceFormat::ArrowIpcFile,
            write_ipc_file,
        ),
        (
            "data.stream",
            ImportSourceFormat::ArrowIpcStream,
            write_ipc_stream,
        ),
    ];

    let grant = common::import_grant();
    for (index, (name, format, write)) in cases.into_iter().enumerate() {
        let source = temp.path().join(name);
        write(&source, &batch);
        let root = temp.path().join(format!("canonical-{index}"));
        let identity = CanonicalDatasetIdentity::external(
            format!("binary-{index}"),
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("identity");
        import_path_to_vortex(ImportRequest {
            source_path: &source,
            configured_root: &root,
            identity: &identity,
            declared_format: format,
            expected_generation: None,
            limits: &ImportLimits::conservative_for_tests(),
            auxiliary_slot: grant
                .auxiliary_slot()
                .expect("import grant owns a source-seal slot"),
        })
        .expect("binary import");
        let manifest = read_current_manifest(&root, &identity).expect("manifest");
        let loaded = load_vortex(manifest.generation_path()).expect("reopen");
        let source_open = batch
            .column_by_name("open")
            .expect("open")
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("f64 open");
        for (actual, expected) in loaded.open.iter().zip(source_open.values()) {
            assert_eq!(actual.to_bits(), expected.to_bits(), "format {format:?}");
        }
    }
}

#[test]
fn parquet_and_arrow_ipc_float32_prices_are_precision_unrecoverable() {
    let temp = tempfile::tempdir().expect("tempdir");
    let batch = binary_fixture(true);
    let cases: [(&str, ImportSourceFormat, BinaryFixtureWriter); 3] = [
        ("bad.parquet", ImportSourceFormat::Parquet, write_parquet),
        (
            "bad.arrow",
            ImportSourceFormat::ArrowIpcFile,
            write_ipc_file,
        ),
        (
            "bad.stream",
            ImportSourceFormat::ArrowIpcStream,
            write_ipc_stream,
        ),
    ];

    let grant = common::import_grant();
    for (index, (name, format, write)) in cases.into_iter().enumerate() {
        let source = temp.path().join(name);
        write(&source, &batch);
        let root = temp.path().join(format!("rejected-{index}"));
        let identity = CanonicalDatasetIdentity::external(
            format!("bad-binary-{index}"),
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("identity");
        let error = import_path_to_vortex(ImportRequest {
            source_path: &source,
            configured_root: &root,
            identity: &identity,
            declared_format: format,
            expected_generation: None,
            limits: &ImportLimits::conservative_for_tests(),
            auxiliary_slot: grant
                .auxiliary_slot()
                .expect("import grant owns a source-seal slot"),
        })
        .expect_err("Float32 binary prices must fail");
        let detail = format!("{error:#}");
        assert!(
            detail.contains("Float32") && detail.contains("precision"),
            "{detail}"
        );
        assert!(read_current_manifest(&root, &identity).is_err());
    }
}

fn vortex_fixture(float32_prices: bool, float32_volume: bool) -> vortex_array::ArrayRef {
    let timestamps = vec![
        1_700_000_040_000_i64,
        1_700_000_100_000_i64,
        1_700_000_160_000_i64,
    ];
    let values = vec![
        1.000_000_000_000_000_2_f64,
        1.234_567_890_123_456_7_f64,
        1.234_567_990_123_456_8_f64,
    ];
    let mut fields = vec![(
        "timestamp",
        PrimitiveArray::from_iter(timestamps).into_array(),
    )];
    for (name, offset) in [
        ("open", 0.0),
        ("high", 0.000_2),
        ("low", -0.000_2),
        ("close", 0.000_1),
    ] {
        let array = if float32_prices {
            PrimitiveArray::from_iter(values.iter().map(|value| (*value + offset) as f32))
                .into_array()
        } else {
            PrimitiveArray::from_iter(values.iter().map(|value| *value + offset)).into_array()
        };
        fields.push((name, array));
    }
    let volume = if float32_volume {
        PrimitiveArray::from_iter([0.0_f32, 16_777_217.0, 3.0]).into_array()
    } else {
        PrimitiveArray::from_iter([0.0_f64, 16_777_217.0, 3.0]).into_array()
    };
    fields.push(("volume", volume));
    StructArray::from_fields(&fields)
        .expect("Vortex fixture")
        .into_array()
}

#[test]
fn vortex_input_is_independently_copied_and_float32_prices_are_rejected() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("operator.vortex");
    neoethos_data::write_vortex_array(&source, vortex_fixture(false, false))
        .expect("write f64 Vortex");
    let exact_source_bytes = fs::read(&source).expect("read source bytes");
    let root = temp.path().join("vortex-canonical");
    let identity = CanonicalDatasetIdentity::external(
        "vortex-input",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let grant = common::import_grant();
    import_path_to_vortex(ImportRequest {
        source_path: &source,
        configured_root: &root,
        identity: &identity,
        declared_format: ImportSourceFormat::Vortex,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect("strict Vortex registration");
    let manifest = read_current_manifest(&root, &identity).expect("manifest");
    assert_eq!(
        fs::read(manifest.generation_path()).expect("generation bytes"),
        exact_source_bytes
    );
    fs::write(&source, b"mutated after publication").expect("mutate original");
    assert_eq!(
        fs::read(manifest.generation_path()).expect("independent generation"),
        exact_source_bytes
    );

    let float32_source = temp.path().join("float32.vortex");
    neoethos_data::write_vortex_array(&float32_source, vortex_fixture(true, false))
        .expect("write f32 Vortex");
    let rejected_root = temp.path().join("vortex-rejected");
    let rejected_identity = CanonicalDatasetIdentity::external(
        "vortex-float32",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let error = import_path_to_vortex(ImportRequest {
        source_path: &float32_source,
        configured_root: &rejected_root,
        identity: &rejected_identity,
        declared_format: ImportSourceFormat::Vortex,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect_err("Float32 Vortex prices must fail");
    let detail = format!("{error:#}");
    assert!(
        detail.contains("Float32") && detail.contains("precision"),
        "{detail}"
    );
    assert!(read_current_manifest(&rejected_root, &rejected_identity).is_err());
}

#[test]
fn parquet_arrow_and_vortex_float32_volume_are_precision_unrecoverable() {
    let temp = tempfile::tempdir().expect("tempdir");
    let batch = binary_fixture_with_volume(
        false,
        DataType::Float32,
        Arc::new(Float32Array::from(vec![0.0, 16_777_217.0, 3.0])) as ArrayRef,
    );
    let binary_cases: [(&str, ImportSourceFormat, BinaryFixtureWriter); 3] = [
        ("volume.parquet", ImportSourceFormat::Parquet, write_parquet),
        (
            "volume.arrow",
            ImportSourceFormat::ArrowIpcFile,
            write_ipc_file,
        ),
        (
            "volume.stream",
            ImportSourceFormat::ArrowIpcStream,
            write_ipc_stream,
        ),
    ];
    let grant = common::import_grant();
    for (index, (name, format, write)) in binary_cases.into_iter().enumerate() {
        let source = temp.path().join(name);
        write(&source, &batch);
        let root = temp.path().join(format!("float32-volume-{index}"));
        let identity = CanonicalDatasetIdentity::external(
            format!("float32-volume-{index}"),
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("identity");
        let error = import_path_to_vortex(ImportRequest {
            source_path: &source,
            configured_root: &root,
            identity: &identity,
            declared_format: format,
            expected_generation: None,
            limits: &ImportLimits::conservative_for_tests(),
            auxiliary_slot: grant
                .auxiliary_slot()
                .expect("import grant owns a source-seal slot"),
        })
        .expect_err("Float32 binary volume must fail");
        let detail = format!("{error:#}");
        assert!(
            detail.contains("Float32") && detail.contains("precision"),
            "{detail}"
        );
        assert!(read_current_manifest(&root, &identity).is_err());
    }

    let vortex_source = temp.path().join("volume.vortex");
    neoethos_data::write_vortex_array(&vortex_source, vortex_fixture(false, true))
        .expect("write f32-volume Vortex");
    let root = temp.path().join("float32-volume-vortex");
    let identity = CanonicalDatasetIdentity::external(
        "float32-volume-vortex",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let error = import_path_to_vortex(ImportRequest {
        source_path: &vortex_source,
        configured_root: &root,
        identity: &identity,
        declared_format: ImportSourceFormat::Vortex,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect_err("Float32 Vortex volume must fail");
    let detail = format!("{error:#}");
    assert!(
        detail.contains("Float32") && detail.contains("precision"),
        "{detail}"
    );
    assert!(read_current_manifest(&root, &identity).is_err());
}

#[test]
fn uint64_volume_is_preserved_raw_and_only_exact_values_enter_f64_features() {
    let temp = tempfile::tempdir().expect("tempdir");
    let grant = common::import_grant();
    let largest_exact_below_two_to_64 = u64::MAX - 2_047;
    let exact_values = [0_u64, 1_u64 << 53, largest_exact_below_two_to_64];
    let exact_batch = binary_fixture_with_volume(
        false,
        DataType::UInt64,
        Arc::new(UInt64Array::from(exact_values.to_vec())) as ArrayRef,
    );
    let writers: [(&str, ImportSourceFormat, BinaryFixtureWriter); 3] = [
        ("exact.parquet", ImportSourceFormat::Parquet, write_parquet),
        (
            "exact.arrow",
            ImportSourceFormat::ArrowIpcFile,
            write_ipc_file,
        ),
        (
            "exact.stream",
            ImportSourceFormat::ArrowIpcStream,
            write_ipc_stream,
        ),
    ];
    for (index, (name, format, write)) in writers.into_iter().enumerate() {
        let source = temp.path().join(name);
        write(&source, &exact_batch);
        let root = temp.path().join(format!("exact-volume-{index}"));
        let identity = CanonicalDatasetIdentity::external(
            format!("exact-volume-{index}"),
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("identity");
        import_path_to_vortex(ImportRequest {
            source_path: &source,
            configured_root: &root,
            identity: &identity,
            declared_format: format,
            expected_generation: None,
            limits: &ImportLimits::conservative_for_tests(),
            auxiliary_slot: grant
                .auxiliary_slot()
                .expect("import grant owns a source-seal slot"),
        })
        .expect("exact UInt64 volume import");
        let manifest = read_current_manifest(&root, &identity).expect("manifest");
        let provenance =
            ImportProvenanceV1::from_envelope(manifest.provenance()).expect("provenance");
        assert!(matches!(
            provenance.volume_mapping(),
            neoethos_data::core::import_provenance::VolumeMappingV1::ExactUnsignedInteger {
                bit_width: 64,
                ..
            }
        ));
        let loaded = load_vortex(manifest.generation_path()).expect("reopen");
        let expected = exact_values.map(|value| value as f64);
        assert_eq!(loaded.volume.as_deref(), Some(expected.as_slice()));
    }

    for (case_index, inexact) in [(1_u64 << 53) + 1, u64::MAX].into_iter().enumerate() {
        let batch = binary_fixture_with_volume(
            false,
            DataType::UInt64,
            Arc::new(UInt64Array::from(vec![0, inexact, 3])) as ArrayRef,
        );
        let source = temp.path().join(format!("inexact-{case_index}.arrow"));
        write_ipc_file(&source, &batch);
        let root = temp.path().join(format!("inexact-volume-{case_index}"));
        let identity = CanonicalDatasetIdentity::external(
            format!("inexact-volume-{case_index}"),
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("identity");
        import_path_to_vortex(ImportRequest {
            source_path: &source,
            configured_root: &root,
            identity: &identity,
            declared_format: ImportSourceFormat::ArrowIpcFile,
            expected_generation: None,
            limits: &ImportLimits::conservative_for_tests(),
            auxiliary_slot: grant
                .auxiliary_slot()
                .expect("import grant owns a source-seal slot"),
        })
        .expect("inexact raw UInt64 volume remains canonical");
        let manifest = read_current_manifest(&root, &identity).expect("manifest");
        let provenance =
            ImportProvenanceV1::from_envelope(manifest.provenance()).expect("provenance");
        assert!(matches!(
            provenance.volume_mapping(),
            neoethos_data::core::import_provenance::VolumeMappingV1::ExactUnsignedInteger {
                bit_width: 64,
                ..
            }
        ));
        assert_raw_u64_volume(&manifest.generation_path(), &[0, inexact, 3]);
        let error = load_vortex(manifest.generation_path())
            .expect_err("inexact raw volume must not enter f64 feature input");
        assert!(
            format!("{error:#}").contains("no exact f64 mapping"),
            "{error:#}"
        );
        if case_index == 0 {
            let source_generation = manifest.generation_path();
            let copied_root = temp.path().join("raw-volume-vortex-copy");
            let copied_identity = CanonicalDatasetIdentity::external(
                "raw-volume-vortex-copy",
                "EURUSD",
                CanonicalTimeframe::M1,
                BarTimestampConvention::BarOpen,
            )
            .expect("copy identity");
            import_path_to_vortex(ImportRequest {
                source_path: &source_generation,
                configured_root: &copied_root,
                identity: &copied_identity,
                declared_format: ImportSourceFormat::Vortex,
                expected_generation: None,
                limits: &ImportLimits::conservative_for_tests(),
                auxiliary_slot: grant
                    .auxiliary_slot()
                    .expect("import grant owns a source-seal slot"),
            })
            .expect("raw UInt64 Vortex re-import");
            let copied_manifest =
                read_current_manifest(&copied_root, &copied_identity).expect("copied manifest");
            assert_raw_u64_volume(&copied_manifest.generation_path(), &[0, inexact, 3]);
        }
    }
}

#[test]
fn text_integer_volume_that_cannot_map_exactly_to_f64_remains_raw() {
    let temp = tempfile::tempdir().expect("tempdir");
    let grant = common::import_grant();
    let cases = [
        (
            "inexact.csv",
            ImportSourceFormat::Csv,
            "timestamp,open,high,low,close,volume\n1700000040000,1.1,1.2,1.0,1.1,9007199254740993\n",
        ),
        (
            "saturated.jsonl",
            ImportSourceFormat::JsonLines,
            "{\"timestamp\":1700000040000,\"open\":1.1,\"high\":1.2,\"low\":1.0,\"close\":1.1,\"volume\":18446744073709551615}\n",
        ),
    ];
    for (index, (name, format, contents)) in cases.into_iter().enumerate() {
        let source = temp.path().join(name);
        fs::write(&source, contents).expect("write inexact text volume");
        let root = temp.path().join(format!("inexact-text-volume-{index}"));
        let identity = CanonicalDatasetIdentity::external(
            format!("inexact-text-volume-{index}"),
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("identity");
        import_path_to_vortex(ImportRequest {
            source_path: &source,
            configured_root: &root,
            identity: &identity,
            declared_format: format,
            expected_generation: None,
            limits: &ImportLimits::conservative_for_tests(),
            auxiliary_slot: grant
                .auxiliary_slot()
                .expect("import grant owns a source-seal slot"),
        })
        .expect("inexact text integer volume remains canonical raw data");
        let manifest = read_current_manifest(&root, &identity).expect("manifest");
        let provenance =
            ImportProvenanceV1::from_envelope(manifest.provenance()).expect("provenance");
        assert!(matches!(
            provenance.volume_mapping(),
            neoethos_data::core::import_provenance::VolumeMappingV1::ExactUnsignedInteger {
                bit_width: 64,
                ..
            }
        ));
        let expected = if index == 0 {
            9_007_199_254_740_993
        } else {
            u64::MAX
        };
        assert_raw_u64_volume(&manifest.generation_path(), &[expected]);
        let error = load_vortex(manifest.generation_path())
            .expect_err("inexact raw text volume must not enter f64 feature input");
        assert!(
            format!("{error:#}").contains("no exact f64 mapping"),
            "{error:#}"
        );
    }
}

#[test]
fn signed_broker_style_integer_volume_remains_raw_and_nonnegative() {
    let temp = tempfile::tempdir().expect("tempdir");
    let grant = common::import_grant();
    let inexact = (1_i64 << 53) + 1;
    let batch = binary_fixture_with_volume(
        false,
        DataType::Int64,
        Arc::new(Int64Array::from(vec![0, inexact, 3])) as ArrayRef,
    );
    let source = temp.path().join("signed-volume.arrow");
    write_ipc_file(&source, &batch);
    let root = temp.path().join("signed-volume");
    let identity = CanonicalDatasetIdentity::external(
        "signed-volume",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    import_path_to_vortex(ImportRequest {
        source_path: &source,
        configured_root: &root,
        identity: &identity,
        declared_format: ImportSourceFormat::ArrowIpcFile,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect("signed raw volume import");
    let manifest = read_current_manifest(&root, &identity).expect("manifest");
    let provenance = ImportProvenanceV1::from_envelope(manifest.provenance()).expect("provenance");
    assert!(matches!(
        provenance.volume_mapping(),
        neoethos_data::core::import_provenance::VolumeMappingV1::ExactSignedInteger {
            bit_width: 64,
            ..
        }
    ));
    assert_raw_i64_volume(&manifest.generation_path(), &[0, inexact, 3]);
    let error = load_vortex(manifest.generation_path())
        .expect_err("non-exact signed raw volume must not enter f64 features");
    assert!(
        format!("{error:#}").contains("no exact f64 mapping"),
        "{error:#}"
    );
}

#[test]
fn declared_text_format_must_match_the_sealed_source_bytes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("mislabelled.csv");
    fs::write(
        &source,
        concat!(
            "timestamp\topen\thigh\tlow\tclose\tvolume\n",
            "1700000040000\t1.1\t1.2\t1.0\t1.1\t0\n",
            "1700000100000\t1.2\t1.3\t1.1\t1.2\t1\n",
        ),
    )
    .expect("write TSV bytes behind a CSV-looking path");
    let root = temp.path().join("canonical");
    let identity = CanonicalDatasetIdentity::external(
        "declared-format-mismatch",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let grant = common::import_grant();

    let error = import_path_to_vortex(ImportRequest {
        source_path: &source,
        configured_root: &root,
        identity: &identity,
        declared_format: ImportSourceFormat::Csv,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect_err("TSV bytes declared as CSV must fail before publication");

    let message = format!("{error:#}");
    assert!(
        message.contains("declared csv") && message.contains("detected tsv"),
        "format mismatch must be explicit: {message}"
    );
    assert!(
        read_current_manifest(&root, &identity).is_err(),
        "a format-mismatched source must publish no canonical generation"
    );
}

fn assert_raw_u64_volume(path: &std::path::Path, expected: &[u64]) {
    let array = read_vortex_array(path).expect("read raw canonical Vortex");
    let structure = array.to_struct();
    let volume = structure
        .unmasked_field_by_name("volume")
        .expect("raw volume field");
    assert!(volume.all_valid().expect("raw volume validity"));
    assert_eq!(volume.to_primitive().as_slice::<u64>(), expected);
}

fn assert_raw_i64_volume(path: &std::path::Path, expected: &[i64]) {
    let array = read_vortex_array(path).expect("read raw canonical Vortex");
    let structure = array.to_struct();
    let volume = structure
        .unmasked_field_by_name("volume")
        .expect("raw volume field");
    assert!(volume.all_valid().expect("raw volume validity"));
    assert_eq!(volume.to_primitive().as_slice::<i64>(), expected);
}
