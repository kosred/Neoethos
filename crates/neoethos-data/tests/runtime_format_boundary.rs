use neoethos_data::core::import_limits::ImportLimits;
use neoethos_data::core::import_provenance::ImportSourceFormat;
use neoethos_data::core::import_service::{ImportRequest, import_path_to_vortex};
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe, DataVerificationStatus,
    DatasetDiscovery, ImportDiscovery, discover_symbols, discover_timeframes,
    load_dataset_for_identity, load_symbol_timeframe,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

mod common;

fn external_identity() -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::external(
        "runtime-format-boundary-test",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid external dataset identity")
}

#[test]
fn explicit_import_publishes_the_only_runtime_loadable_generation() {
    let temporary = tempfile::tempdir().expect("temporary runtime-boundary root");
    let source = temporary.path().join("EURUSD_M1.csv");
    let canonical_root = temporary.path().join("canonical");
    fs::write(
        &source,
        concat!(
            "timestamp,open,high,low,close,volume\n",
            "1700000040000,1.1234567890123,1.2234567890123,1.0234567890123,1.1734567890123,0\n",
            "1700000100000,1.1734567890123,1.2734567890123,1.0734567890123,1.2234567890123,16777217\n",
            "1700000160000,1.2234567890123,1.3234567890123,1.1234567890123,1.2734567890123,42\n",
        ),
    )
    .expect("write exact-f64 CSV fixture");

    let before = load_symbol_timeframe(&canonical_root, "EURUSD", "M1")
        .expect_err("a raw source file is never a runtime dataset");
    assert!(
        before.to_string().contains("explicit import"),
        "runtime rejection must tell the operator how to proceed: {before:#}"
    );

    let identity = external_identity();
    let grant = common::import_grant();
    let imported = import_path_to_vortex(ImportRequest {
        source_path: &source,
        configured_root: &canonical_root,
        identity: &identity,
        declared_format: ImportSourceFormat::Csv,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect("explicit CSV import publishes a verified canonical generation");
    assert_eq!(imported.manifest().generation_id(), imported.generation());
    assert_eq!(
        imported.provenance().dataset_identity(),
        imported.manifest().identity()
    );

    assert_eq!(
        discover_timeframes(&canonical_root, "EURUSD").expect("discover canonical runtime data"),
        vec!["M1"],
        "runtime discovery must resolve manifest-backed canonical identities"
    );
    assert_eq!(
        discover_symbols(&canonical_root).expect("discover canonical symbols"),
        vec!["EURUSD"],
        "runtime symbol discovery must decode canonical identities"
    );

    let loaded = load_symbol_timeframe(&canonical_root, "EURUSD", "M1")
        .expect("runtime resolves the verified canonical generation");
    let expected = [1.1734567890123_f64, 1.2234567890123, 1.2734567890123];
    assert_eq!(loaded.close.len(), expected.len());
    for (actual, expected) in loaded.close.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn retired_human_layout_is_not_a_runtime_compatibility_fallback() {
    let temporary = tempfile::tempdir().expect("temporary legacy-layout root");
    let legacy_path = temporary
        .path()
        .join("symbol=EURUSD")
        .join("timeframe=M1")
        .join("data.vortex");
    fs::create_dir_all(legacy_path.parent().expect("legacy parent"))
        .expect("create retired layout fixture");
    fs::write(&legacy_path, b"retired-layout-is-never-decoded")
        .expect("write retired-layout fixture");

    assert!(
        discover_timeframes(temporary.path(), "EURUSD")
            .expect("scan runtime datasets")
            .is_empty(),
        "retired directories must not be registered as runnable datasets"
    );
    assert!(
        discover_symbols(temporary.path())
            .expect("scan runtime symbols")
            .is_empty(),
        "retired directories must not register a runtime symbol"
    );

    let error = load_symbol_timeframe(temporary.path(), "EURUSD", "M1")
        .expect_err("retired layout must require an explicit offline migration");
    let message = format!("{error:#}");
    assert!(message.contains("explicit offline migration"), "{message}");
}

#[test]
fn import_discovery_never_registers_raw_sources_as_runtime_data() {
    let temporary = tempfile::tempdir().expect("temporary discovery root");
    let fixtures = [
        ("EURUSD_M1.csv", ImportSourceFormat::Csv),
        ("EURUSD_M1.tsv", ImportSourceFormat::Tsv),
        ("EURUSD_M1.json", ImportSourceFormat::JsonArray),
        ("EURUSD_M1.jsonl", ImportSourceFormat::JsonLines),
        ("EURUSD_M1.parquet", ImportSourceFormat::Parquet),
        ("EURUSD_M1.arrow", ImportSourceFormat::ArrowIpcFile),
        ("EURUSD_M1.arrows", ImportSourceFormat::ArrowIpcStream),
        ("EURUSD_M1.vortex", ImportSourceFormat::Vortex),
    ];
    for (name, _) in fixtures {
        fs::write(temporary.path().join(name), b"source-discovery-only")
            .expect("write source-discovery fixture");
    }

    let import = ImportDiscovery::scan(temporary.path()).expect("scan import candidates");
    let actual = import
        .entries
        .iter()
        .map(|entry| entry.format)
        .collect::<BTreeSet<_>>();
    let expected = fixtures
        .into_iter()
        .map(|(_, format)| format)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);

    let runtime = DatasetDiscovery::scan(temporary.path()).expect("scan runtime datasets");
    assert!(
        runtime.entries.is_empty(),
        "raw import candidates are not runnable until verified canonical publication"
    );
    assert_eq!(
        runtime
            .skipped
            .iter()
            .filter(|skipped| matches!(
                skipped.reason,
                neoethos_data::SkipReason::ImportRequired(_)
            ))
            .count(),
        fixtures.len(),
        "every loose source file must produce an explicit import-required diagnostic"
    );
}

#[test]
fn metadata_inventory_is_exact_without_claiming_generation_verification() {
    let temporary = tempfile::tempdir().expect("temporary runtime-boundary root");
    let source = temporary.path().join("EURUSD_M1.csv");
    let canonical_root = temporary.path().join("canonical");
    fs::write(
        &source,
        concat!(
            "timestamp,open,high,low,close\n",
            "1700000040000,1.1,1.2,1.0,1.15\n",
            "1700000100000,1.15,1.25,1.05,1.2\n",
        ),
    )
    .expect("write source fixture");
    let identity = external_identity();
    let grant = common::import_grant();
    let imported = import_path_to_vortex(ImportRequest {
        source_path: &source,
        configured_root: &canonical_root,
        identity: &identity,
        declared_format: ImportSourceFormat::Csv,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect("publish verified dataset");

    let metadata = DatasetDiscovery::scan_metadata(&canonical_root).expect("metadata inventory");
    assert_eq!(metadata.entries.len(), 1);
    let entry = &metadata.entries[0];
    assert_eq!(entry.dataset_identity, identity.to_path_component());
    assert_eq!(entry.generation, imported.generation());
    assert_eq!(
        entry.manifest_binding_sha256,
        imported.manifest().manifest_binding_sha256()
    );
    assert_eq!(entry.verification, DataVerificationStatus::ManifestOnly);

    // Corrupting immutable data proves the metadata inventory does not perform
    // the multi-gigabyte generation hash/read performed by the runtime loader.
    // It must remain explicitly labelled manifest-only, never verified.
    let generation = imported.manifest().generation_path();
    let mut bytes = fs::read(&generation).expect("read generation fixture");
    bytes[0] ^= 0xff;
    fs::write(&generation, bytes).expect("corrupt generation fixture");

    let metadata_after_corruption =
        DatasetDiscovery::scan_metadata(&canonical_root).expect("metadata inventory after damage");
    assert_eq!(metadata_after_corruption.entries.len(), 1);
    assert_eq!(
        metadata_after_corruption.entries[0].verification,
        DataVerificationStatus::ManifestOnly
    );

    let verified = DatasetDiscovery::scan(&canonical_root).expect("verified runtime scan");
    assert!(verified.entries.is_empty());
    assert!(verified.skipped.iter().any(|skipped| matches!(
        skipped.reason,
        neoethos_data::SkipReason::UnverifiedGeneration(_)
    )));
}

#[test]
fn exact_identity_selection_loads_one_source_without_global_symbol_ambiguity() {
    let temporary = tempfile::tempdir().expect("temporary runtime-boundary root");
    let canonical_root = temporary.path().join("canonical");
    let grant = common::import_grant();
    let mut identities = Vec::new();
    for (namespace, price) in [("source-a", "1.15"), ("source-b", "2.15")] {
        let source = temporary.path().join(format!("{namespace}.csv"));
        fs::write(
            &source,
            format!(
                "timestamp,open,high,low,close\n\
                 1700000040000,{price},{price},{price},{price}\n\
                 1700000100000,{price},{price},{price},{price}\n"
            ),
        )
        .expect("write source fixture");
        let identity = CanonicalDatasetIdentity::external(
            namespace,
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("exact source identity");
        import_path_to_vortex(ImportRequest {
            source_path: &source,
            configured_root: &canonical_root,
            identity: &identity,
            declared_format: ImportSourceFormat::Csv,
            expected_generation: None,
            limits: &ImportLimits::conservative_for_tests(),
            auxiliary_slot: grant
                .auxiliary_slot()
                .expect("import grant owns a source-seal slot"),
        })
        .expect("publish exact source");
        identities.push((identity, price.parse::<f64>().expect("price")));
    }

    let ambiguous = load_symbol_timeframe(&canonical_root, "EURUSD", "M1")
        .expect_err("symbol-only loading must fail rather than guess a source");
    assert!(ambiguous.to_string().contains("found 2"));

    for (identity, expected_price) in identities {
        let selected = load_dataset_for_identity(&canonical_root, &identity)
            .expect("load exact source/account series");
        assert_eq!(selected.symbol, "EURUSD");
        assert_eq!(selected.frames["M1"].close, vec![expected_price; 2]);
        assert_eq!(
            selected.source_artifacts["M1"].identity(),
            &identity,
            "selected bytes must remain bound to the requested exact identity"
        );
    }
}

#[test]
fn production_import_adapters_use_only_the_shared_explicit_import_service() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cli = fs::read_to_string(repository.join("crates/neoethos-cli/src/main.rs"))
        .expect("read CLI import adapter");
    let cli_import = cli
        .split_once("fn cmd_import(args: &[String]) -> Result<()> {")
        .expect("CLI keeps an explicit import command")
        .1
        .split_once("\nfn cmd_stop_target")
        .expect("isolate CLI import command")
        .0;
    for required in [
        "--source",
        "--format",
        "--source-namespace",
        "--symbol",
        "--timeframe",
        "--bar-timestamps",
        "import_path_to_vortex",
        "imported.manifest()",
        "imported.provenance()",
        "CompositeAdmissionAuthority",
    ] {
        assert!(
            cli_import.contains(required),
            "CLI import adapter is missing explicit/shared boundary `{required}`"
        );
    }
    for retired in [
        "universal_importer",
        "import_directory_recursive",
        "--force",
        "--data-path",
        "dataset_manifest::read_current_manifest",
    ] {
        assert!(
            !cli_import.contains(retired),
            "CLI import adapter still reaches retired implicit conversion `{retired}`"
        );
    }

    let app = fs::read_to_string(repository.join("crates/neoethos-app/src/server/data_control.rs"))
        .expect("read app import adapter");
    assert!(app.contains("import_path_to_vortex"));
    assert!(app.contains("imported.manifest()"));
    assert!(app.contains("imported.provenance()"));
    assert!(app.contains("admit_import"));
    assert!(!app.contains("dataset_manifest::read_current_manifest"));
    assert!(!app.contains("universal_importer"));
    assert!(!app.contains("to_vortex::"));

    let tui = fs::read_to_string(repository.join("crates/neoethos-cli/src/tui/pages/symbols.rs"))
        .expect("read TUI import adapter");
    assert!(
        !tui.contains("auto-detected"),
        "TUI must never claim that a mutable source format is inferred for publication"
    );
    for required in [
        "--source",
        "--format",
        "--source-namespace",
        "--symbol",
        "--timeframe",
        "--bar-timestamps",
        "DatasetDiscovery::scan",
    ] {
        assert!(
            tui.contains(required),
            "TUI import/inventory adapter is missing `{required}`"
        );
    }
    assert!(
        !tui.contains("strip_prefix(\"symbol=\")"),
        "TUI inventory must not scan the retired human directory layout"
    );
}

#[test]
fn superseded_conversion_engines_and_runtime_format_variants_are_deleted() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let retired_files = [
        "crates/neoethos-data/src/core/universal_importer.rs",
        "crates/neoethos-data/src/core/to_vortex.rs",
        "crates/neoethos-data/src/core/parquet_migration.rs",
    ];
    for relative in retired_files {
        assert!(
            !repository.join(relative).exists(),
            "superseded conversion engine still exists: {relative}"
        );
    }

    let modules = fs::read_to_string(repository.join("crates/neoethos-data/src/core/mod.rs"))
        .expect("read data core module registry");
    let root = fs::read_to_string(repository.join("crates/neoethos-data/src/lib.rs"))
        .expect("read data crate exports");
    let cli = fs::read_to_string(repository.join("crates/neoethos-cli/src/main.rs"))
        .expect("read CLI commands");
    for retired_module in [
        "pub mod universal_importer;",
        "pub mod to_vortex;",
        "pub mod parquet_migration;",
    ] {
        assert!(
            !modules.contains(retired_module),
            "module registry retains {retired_module}"
        );
    }
    for retired_export in ["core::parquet_migration", "core::universal_importer"] {
        assert!(
            !root.contains(retired_export),
            "crate exports retain {retired_export}"
        );
    }
    for retired_command in [
        "\"migrate-data\" =>",
        "fn cmd_migrate_data",
        "migrate_legacy_parquet_tree",
    ] {
        assert!(
            !cli.contains(retired_command),
            "CLI retains {retired_command}"
        );
    }

    let discovery =
        fs::read_to_string(repository.join("crates/neoethos-data/src/core/discover.rs"))
            .expect("read runtime discovery type");
    let data_format = discovery
        .split_once("pub enum DataFormat {")
        .expect("runtime DataFormat enum")
        .1
        .split_once('}')
        .expect("runtime DataFormat enum closes")
        .0;
    for non_vortex in ["Parquet", "Arrow", "Csv", "Tsv", "Json", "JsonLines"] {
        assert!(
            !data_format.contains(&format!("{non_vortex},")),
            "runtime DataFormat still exposes non-Vortex variant {non_vortex}"
        );
    }
}

#[test]
fn canonical_data_runtime_has_no_polars_dependency() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workspace_manifest =
        fs::read_to_string(repository.join("Cargo.toml")).expect("read workspace manifest");
    let data_manifest = fs::read_to_string(repository.join("crates/neoethos-data/Cargo.toml"))
        .expect("read data manifest");

    assert!(
        !workspace_manifest
            .lines()
            .any(|line| line.trim_start().starts_with("polars")),
        "the Vortex-only workspace boundary still declares an unused Polars dependency"
    );
    assert!(
        !data_manifest
            .lines()
            .any(|line| line.trim_start().starts_with("polars")),
        "neoethos-data still links Polars even though production data is Vortex-only"
    );
    assert!(
        !workspace_manifest
            .lines()
            .any(|line| line.trim_start().starts_with("arrow-csv")),
        "the workspace still declares unused arrow-csv after CSV parsing moved behind the shared importer"
    );
    assert!(
        !data_manifest
            .lines()
            .any(|line| line.trim_start().starts_with("arrow-csv")),
        "neoethos-data still links unused arrow-csv"
    );
}

#[test]
fn canonical_data_runtime_exposes_no_retired_human_layout_api() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let data = fs::read_to_string(repository.join("crates/neoethos-data/src/lib.rs"))
        .expect("read data runtime source");
    for retired in [
        "pub enum VortexIntegrity",
        "pub fn vortex_integrity",
        "pub fn symbol_timeframe_vortex_path",
        "pub fn write_symbol_timeframe_vortex",
        "pub fn write_symbol_timeframe_vortex_with_volume",
    ] {
        assert!(
            !data.contains(retired),
            "retired human-layout production API remains reachable: {retired}"
        );
    }
}

#[test]
fn incomplete_new_identity_never_blocks_existing_verified_data() {
    let temporary = tempfile::tempdir().expect("temporary runtime-boundary root");
    let source = temporary.path().join("EURUSD_M1.csv");
    let canonical_root = temporary.path().join("canonical");
    fs::write(
        &source,
        concat!(
            "timestamp,open,high,low,close\n",
            "1700000040000,1.1,1.2,1.0,1.15\n",
            "1700000100000,1.15,1.25,1.05,1.2\n",
        ),
    )
    .expect("write source fixture");
    let identity = external_identity();
    let grant = common::import_grant();
    import_path_to_vortex(ImportRequest {
        source_path: &source,
        configured_root: &canonical_root,
        identity: &identity,
        declared_format: ImportSourceFormat::Csv,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect("publish existing verified dataset");

    let incomplete = CanonicalDatasetIdentity::external(
        "concurrent-first-import",
        "GBPUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid second identity");
    fs::create_dir_all(canonical_root.join(incomplete.to_path_component()))
        .expect("simulate an in-flight first publication");

    let loaded = load_symbol_timeframe(&canonical_root, "EURUSD", "M1")
        .expect("unrelated verified data remains loadable");
    assert_eq!(loaded.len(), 2);
    let discovery = DatasetDiscovery::scan(&canonical_root).expect("scan with incomplete identity");
    assert_eq!(discovery.entries.len(), 1);
    assert!(discovery.skipped.iter().any(|skipped| matches!(
        skipped.reason,
        neoethos_data::SkipReason::UnverifiedGeneration(_)
    )));
}
