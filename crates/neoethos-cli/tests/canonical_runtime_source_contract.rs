const CLI_MAIN: &str = include_str!("../src/main.rs");
const TUI_CHART: &str = include_str!("../src/tui/pages/chart.rs");
const TUI_SYMBOLS: &str = include_str!("../src/tui/pages/symbols.rs");
const TUI_DASHBOARD: &str = include_str!("../src/tui/pages/dashboard.rs");
const TUI_FORM: &str = include_str!("../src/tui/form.rs");
const TUI_CONFIG: &str = include_str!("../src/tui/pages/config_view.rs");
const TUI_STRATEGIES: &str = include_str!("../src/tui/pages/strategies.rs");
const GPU_BENCH_PREPARE: &str = include_str!("../src/gpu_bench_prepare.rs");
const GPU_BENCH_RUNNER: &str = include_str!("../../../scripts/gpu-bench/run_rented.sh");
const GPU_BENCH_README: &str = include_str!("../../../scripts/gpu-bench/README.md");
const STAGE1_WORKFLOW: &str = include_str!("../../../.github/workflows/agent-stage1.yml");

#[test]
fn higher_timeframes_use_verified_canonical_identities() {
    assert!(
        !CLI_MAIN.contains("symbol_timeframe_vortex_path"),
        "CLI runtime must not probe the retired symbol=/timeframe= layout"
    );
    assert!(
        CLI_MAIN.contains("DatasetDiscovery::scan_metadata")
            && CLI_MAIN.contains("load_canonical_timeframe"),
        "CLI may inventory manifests cheaply, but runtime consumption must fully verify the exact identity"
    );
}

#[test]
fn every_cli_runtime_consumer_requires_an_exact_dataset_identity() {
    for (label, source) in [("main", CLI_MAIN), ("gpu-bench", GPU_BENCH_PREPARE)] {
        for retired in [
            "load_symbol_timeframe(",
            "load_symbol_dataset(",
            "load_symbol_timeframe_tail(",
            "discover_timeframes(",
        ] {
            assert!(
                !source.contains(retired),
                "{label} still uses ambiguous runtime boundary {retired}"
            );
        }
    }
    assert!(
        CLI_MAIN.contains("--dataset-identity")
            && CLI_MAIN.contains("select_exact_runtime_identity")
            && GPU_BENCH_PREPARE.contains("CanonicalDatasetIdentity")
            && GPU_BENCH_PREPARE.contains("load_canonical_timeframe"),
        "CLI commands and paid GPU snapshot preparation must carry and fully verify an opaque exact identity"
    );
}

#[test]
fn paid_gpu_preparation_reopens_shared_imported_vortex_only() {
    for retired in ["--csv", "parse_canonical_csv", "struct CanonicalSeries"] {
        assert!(
            !GPU_BENCH_PREPARE.contains(retired),
            "GPU benchmark preparation retains private input path {retired}"
        );
    }
    for (label, source) in [
        ("runner", GPU_BENCH_RUNNER),
        ("README", GPU_BENCH_README),
        ("workflow", STAGE1_WORKFLOW),
    ] {
        assert!(
            !source.contains("prepare_snapshot.py")
                && !source.contains("bench-prepare --csv")
                && !source.contains("input-csv-dir"),
            "{label} still drives or advertises a retired direct CSV snapshot path"
        );
    }
    assert!(
        GPU_BENCH_RUNNER.contains(" import ")
            && GPU_BENCH_RUNNER.contains("--dataset-identity")
            && GPU_BENCH_README.contains("canonical Vortex"),
        "the paid runner must use the shared importer and reopen its exact canonical Vortex identity"
    );
    assert!(
        GPU_BENCH_PREPARE.contains("parse_usize(args, \"--max-features\", usize::MAX)"),
        "paid preparation must exercise the complete connected feature frame unless the operator explicitly narrows it"
    );
}

#[test]
fn cli_never_manufactures_a_timeframe() {
    assert!(
        !CLI_MAIN.to_ascii_lowercase().contains("resample"),
        "CLI production code/help must not expose or call timeframe synthesis"
    );
    assert!(
        !CLI_MAIN.contains("REQUIRED_DIRECT_TIMEFRAMES")
            && CLI_MAIN.contains("require_direct_timeframes"),
        "discovery/search must verify only the explicitly consumed direct generations"
    );
    assert!(
        CLI_MAIN.contains("import/download required"),
        "a missing direct timeframe must tell the operator to import or download it"
    );
    assert!(
        CLI_MAIN.contains("No timeframe is manufactured"),
        "active CLI help must state the direct-generation contract"
    );
}

#[test]
fn discover_dry_run_still_requires_direct_canonical_generations() {
    let discover = CLI_MAIN
        .split_once("fn cmd_discover(args: &[String])")
        .expect("discover command")
        .1
        .split_once("fn cmd_batch_discover(args: &[String])")
        .expect("batch-discover command boundary")
        .0;
    let selection = discover
        .find("select_runtime_timeframe_identities")
        .expect("direct canonical selection preflight");
    let dry_run_exit = discover
        .find("if has_flag(args, \"--dry-run\")")
        .expect("discover dry-run exit");

    assert!(
        selection < dry_run_exit,
        "discover --dry-run must fail on missing/ambiguous direct generations before reporting success"
    );
}

#[test]
fn cli_financial_and_batch_paths_keep_the_exact_selected_identity() {
    assert!(
        !CLI_MAIN.contains("set_store_root")
            && CLI_MAIN.contains("set_store_selection")
            && CLI_MAIN.contains("selection.base_identity.clone()"),
        "financial lookups must be anchored to the exact selected canonical series"
    );
    assert!(
        !CLI_MAIN.contains("run_batch(&symbols, &tfs)") && CLI_MAIN.contains("run_batch(&anchors)"),
        "batch discovery must pass exact canonical anchors, not display labels"
    );
    assert!(
        CLI_MAIN.contains("faithful_oos_eval("),
        "forward/OOS evaluation must delegate exact reopening to the v2 portfolio authority"
    );
    let forward_test = CLI_MAIN
        .split_once("fn cmd_forward_test(args: &[String])")
        .expect("forward-test command")
        .1
        .split_once("fn cmd_blend_test(args: &[String])")
        .expect("blend-test command boundary")
        .0;
    assert!(
        !forward_test.contains("unique_canonical_identity")
            && !forward_test.contains("inventory_canonical_identities")
            && !forward_test.contains("selected_identity"),
        "forward-test must not replace the v2 portfolio receipt with a current symbol/timeframe lookup"
    );
}

#[test]
fn schedule_uses_manifest_row_count_without_a_compression_guess() {
    assert!(
        !CLI_MAIN.contains("--bytes-per-bar"),
        "compressed file size is not a truthful row-count estimate"
    );
    assert!(
        CLI_MAIN.contains("read_current_manifest") && CLI_MAIN.contains("manifest.row_count()"),
        "schedule must read the verified canonical manifest row count"
    );
}

#[test]
fn slice_publishes_a_typed_canonical_generation() {
    assert!(
        !CLI_MAIN.contains("write_symbol_timeframe_vortex"),
        "slice-dataset must not write the retired loose layout"
    );
    assert!(
        CLI_MAIN.contains("publish_canonical_ohlcv_generation"),
        "slice-dataset must use the canonical immutable-generation publisher"
    );
    assert!(
        CLI_MAIN.contains("neoethos.cli.slice-dataset-provenance.v1"),
        "slice-dataset must attach typed, versioned derivation provenance"
    );
}

#[test]
fn cli_inventory_is_manifest_only_and_prints_exact_entry_identity() {
    assert!(
        CLI_MAIN.matches("DatasetDiscovery::scan_metadata").count() >= 2,
        "CLI identity resolution and discovery summary must use bounded manifest inventory"
    );
    assert!(
        CLI_MAIN.contains("entry.dataset_identity")
            && CLI_MAIN.contains("entry.generation")
            && CLI_MAIN.contains("entry.manifest_binding_sha256")
            && CLI_MAIN.contains("entry.verification"),
        "CLI inventory output must expose exact identity, generation, binding and verification"
    );
    assert!(
        !CLI_MAIN.contains("Files found:") && CLI_MAIN.contains("Canonical identities:"),
        "manifest inventory must not describe canonical identities as loose files"
    );
}

#[test]
fn tui_inventory_uses_metadata_and_surfaces_rejections() {
    for (label, source) in [("chart", TUI_CHART), ("symbols", TUI_SYMBOLS)] {
        assert!(
            source.contains("DatasetDiscovery::scan_metadata"),
            "{label} inventory must use manifest-only scan_metadata"
        );
        assert!(
            source.contains("reason.category()"),
            "{label} inventory must surface every rejected category"
        );
        assert!(
            source.contains("dataset_identity") && source.contains("generation"),
            "{label} inventory must retain exact identity and generation"
        );
    }
}

#[test]
fn cli_default_symbol_inventory_is_manifest_backed_and_exactly_reported() {
    assert!(
        !CLI_MAIN.contains("neoethos_data::discover_symbols"),
        "batch and auto-loop defaults must not hide exact canonical identities behind grouped discovery"
    );
    assert!(
        CLI_MAIN.contains("metadata_inventory_symbols"),
        "shared CLI symbol inventory must preserve exact entry reporting"
    );
    assert!(
        CLI_MAIN.contains("metadata inventory found no canonical dataset identities"),
        "batch/auto-loop inventory must fail closed instead of reporting empty work as success"
    );
}

#[test]
fn dashboard_inventory_uses_manifests_and_surfaces_exact_entries_and_skips() {
    assert!(
        TUI_DASHBOARD.contains("DatasetDiscovery::scan_metadata"),
        "dashboard inventory must use bounded canonical manifest discovery"
    );
    for field in [
        "dataset_identity",
        "generation",
        "manifest_binding_sha256",
        "verification",
        "reason.category()",
    ] {
        assert!(
            TUI_DASHBOARD.contains(field),
            "dashboard inventory must surface {field}"
        );
    }
    assert!(
        !TUI_DASHBOARD.contains("starts_with(\"symbol=\")"),
        "dashboard must not scan the retired symbol=/timeframe= layout"
    );
    assert!(
        TUI_DASHBOARD.contains("InventoryCache")
            && TUI_DASHBOARD.contains("Duration::from_secs(2)"),
        "per-frame dashboard draws must not rescan every canonical manifest"
    );
}

#[test]
fn active_cli_and_tui_help_does_not_advertise_the_retired_layout() {
    assert!(
        !CLI_MAIN.contains("auto-discover dataset layout")
            && !CLI_MAIN.contains("Hive-style or flat"),
        "CLI help must not advertise loose layouts that canonical runtime rejects"
    );
    for (name, source) in [("form", TUI_FORM), ("config", TUI_CONFIG)] {
        assert!(
            !source.contains("symbol=*/timeframe=*"),
            "{name} help must describe canonical manifest-backed roots"
        );
    }
}

#[test]
fn weekly_promotion_uses_the_selected_v3_portfolio_as_its_only_authority() {
    let command = CLI_MAIN
        .split_once("fn cmd_discovery_promote_weekly(args: &[String])")
        .expect("weekly promotion command")
        .1
        .split_once("fn cmd_")
        .map(|(body, _)| body)
        .unwrap_or_default();
    for required in [
        "--portfolio",
        "load_live_portfolio_json",
        "artifact.search_scope",
        "artifact.search_config_hash",
        "load_prior_ledger(\n        &cache_dir,\n        &symbol,\n        &tf,\n        &search_receipt,",
        "CanonicalSearchArtifactEnvelopeV2::new(",
    ] {
        assert!(
            command.contains(required),
            "weekly promotion is missing exact authority `{required}`"
        );
    }
    assert!(
        !command.contains("--search-receipt")
            && !command.contains("--config-hash")
            && !command.contains("ledger_path(&cache_dir, &symbol, &tf)"),
        "weekly promotion still accepts competing or display-derived authority"
    );
    assert!(
        TUI_STRATEGIES.contains("--portfolio")
            && TUI_STRATEGIES.contains("ends_with(\".live_portfolio.json\")")
            && !TUI_STRATEGIES.contains("parse_symbol_tf")
            && !TUI_STRATEGIES.contains("sidecar.push(\".live_portfolio.json\")")
            && !TUI_STRATEGIES.contains("--symbol")
            && !TUI_STRATEGIES.contains("--tf"),
        "TUI promotion must select and pass an exact v3 artifact, never reconstruct its path or identity"
    );
}
