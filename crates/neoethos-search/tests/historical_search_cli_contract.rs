use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
    CanonicalTimeframe, CanonicalVolumeRef, Ohlcv, publish_canonical_ohlcv_generation,
};
use neoethos_search::ExactCanonicalSeries;

static NEXT_TEMP_ROOT: AtomicU64 = AtomicU64::new(0);

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        let unique = NEXT_TEMP_ROOT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "neoethos-historical-search-cli-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create isolated historical-search root");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!(
                "ERROR historical-search CLI cleanup failed for {}: {error}",
                self.0.display()
            );
        }
    }
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn lightweight_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_neoethos-historical-search")
        .expect("the model-free neoethos-historical-search binary target must exist")
}

fn publish_fixture(root: &Path, namespace: &str) -> CanonicalDatasetIdentity {
    let identity = CanonicalDatasetIdentity::external(
        namespace,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid direct M1 identity");
    let rows = 192_usize;
    let timestamp = (0..rows)
        .map(|row| 1_704_067_200_000_i64 + row as i64 * 60_000)
        .collect::<Vec<_>>();
    let close = (0..rows)
        .map(|row| 1.10 + row as f64 * 0.00001 + (row as f64 / 9.0).sin() * 0.0004)
        .collect::<Vec<_>>();
    let open = (0..rows)
        .map(|row| {
            row.checked_sub(1)
                .map_or(close[0], |previous| close[previous])
        })
        .collect::<Vec<_>>();
    let high = open
        .iter()
        .zip(&close)
        .map(|(open, close)| open.max(*close) + 0.0003)
        .collect::<Vec<_>>();
    let low = open
        .iter()
        .zip(&close)
        .map(|(open, close)| open.min(*close) - 0.0003)
        .collect::<Vec<_>>();
    let ohlcv = Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: Some((0..rows).map(|row| 100.0 + row as f64).collect()),
    };
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.historical-search-cli-fixture.v1",
        identity.canonical_bytes(),
    )
    .expect("fixture provenance");
    publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity: &identity,
        expected_generation: None,
        provenance: &provenance,
        ohlcv: &ohlcv,
        volume: CanonicalVolumeRef::Float64(ohlcv.volume.as_deref().expect("fixture volume")),
        rows_per_chunk: 64,
    })
    .expect("publish direct canonical fixture");
    identity
}

#[test]
fn cli_and_lightweight_bin_share_one_strict_adapter_without_legacy_search() {
    let repository = repository_root();
    let main = fs::read_to_string(repository.join("crates/neoethos-cli/src/main.rs"))
        .expect("read CLI main");
    let adapter =
        fs::read_to_string(repository.join("crates/neoethos-search/src/historical_search_cli.rs"))
            .expect("read shared strict historical-search adapter");
    let selected_frame_builder = fs::read_to_string(
        repository.join("crates/neoethos-search/src/historical_search_receipt_prep.rs"),
    )
    .expect("read shared exact selected-frame builder");
    let bin = fs::read_to_string(
        repository.join("crates/neoethos-search/src/bin/neoethos-historical-search.rs"),
    )
    .expect("read model-free historical-search binary");

    let search_startup = main
        .find("if subcommand == \"search\"")
        .expect("strict search must branch before configuration loading");
    let settings_load = main
        .find("neoethos_core::Settings::load()")
        .expect("non-search commands still load settings");
    assert!(
        search_startup < settings_load,
        "strict search must branch before Settings::load"
    );
    let strict_startup = &main[search_startup..settings_load];
    assert!(
        strict_startup.contains("historical_search_cli::install_historical_search_process_budget(")
            && strict_startup.contains("&raw_args")
            && strict_startup
                .contains("return neoethos_search::historical_search_cli::run(&raw_args[2..]);"),
        "the legacy CLI must use the shared detected-host-plus-parent startup then delegate"
    );
    for forbidden in [
        "Settings::load",
        "from_settings_and_parent",
        "install_search_runtime_overrides_from_settings",
        "install_hardware_runtime_overrides_from_settings",
        "install_data_runtime_overrides",
        "neoethos_models::",
    ] {
        assert!(
            !strict_startup.contains(forbidden),
            "strict pre-config search startup reaches forbidden settings surface: {forbidden}"
        );
    }
    assert!(
        !main.contains("\"search\" => neoethos_search::historical_search_cli::run(&args[2..])"),
        "post-config dispatch must not retain a second search entrypoint"
    );
    assert!(!main.contains("fn cmd_search(args: &[String])"));
    assert!(!main.contains("neoethos_search::evolve_search(&features"));
    assert!(
        bin.contains("historical_search_cli::install_historical_search_process_budget(&args)?;")
            && bin.contains("neoethos_search::historical_search_cli::run"),
        "the standalone binary must use the same strict CPU startup helper"
    );

    let runtime_install = main
        .split_once("setup_logging(false)?;")
        .expect("runtime install start")
        .1
        .split_once("startup_trace.record(StartupEvent::RuntimeSettingsInstalled)?;")
        .expect("runtime install end")
        .0;
    assert!(
        !runtime_install.contains("if subcommand != \"search\"")
            && runtime_install
                .contains("neoethos_search::install_search_runtime_overrides_from_settings",)
            && runtime_install.contains("neoethos_models::"),
        "search must be unreachable before ordinary config-driven runtime installation"
    );

    for forbidden in [
        "neoethos_models::",
        "install_search_runtime_overrides_from_settings(",
        "neoethos_search::evolve_search",
        "broker_truth::",
        "live_portfolio::",
        "net_profit",
        "pnl",
        "load_canonical_timeframe(",
        "exactcanonicalseries",
        "resample",
    ] {
        assert!(
            !adapter.to_ascii_lowercase().contains(forbidden),
            "shared adapter reaches a forbidden fallback/financial path: {forbidden}"
        );
    }
    for required in [
        "SelectedDatasetGenerationV1",
        "build_exact_selected_feature_input",
        "CanonicalSearchRunInputV2",
        "scan_historical_candidates_v2",
    ] {
        assert!(
            adapter.contains(required),
            "shared adapter omits exact production boundary {required}"
        );
    }
    let selected_frame_builder = selected_frame_builder
        .split_once("pub fn build_exact_selected_feature_input(")
        .expect("shared exact selected-frame builder")
        .1
        .split_once("/// Prepare one create-new canonical search-input receipt")
        .expect("shared exact selected-frame builder boundary")
        .0;
    assert!(
        selected_frame_builder.contains("load_exact_canonical_timeframe("),
        "shared selected-frame builder must open every generation through the exact loader"
    );

    let help = main
        .split_once("fn print_help()")
        .expect("CLI help function")
        .1
        .split_once("fn cli_record(")
        .expect("CLI help boundary")
        .0;
    assert!(help.contains("search --expected-input-receipt"));
    let historical_search_help = help
        .lines()
        .filter(|line| line.contains("  search ") || line.contains("ResearchOnly"))
        .collect::<Vec<_>>()
        .join("\n");
    for retired in [
        "search --symbol",
        "--genes 64",
        "--generations 5 --max-indicators",
    ] {
        assert!(
            !historical_search_help.contains(retired),
            "retired search help: {retired}"
        );
    }
    assert!(help.contains("ResearchOnly"));
    assert!(help.contains("NotPromotionEligible"));
    assert!(help.contains("gross-R"));
}

#[test]
fn lightweight_search_requires_exact_receipt_before_feature_computation() {
    let temp = TempRoot::new("missing-receipt");
    let missing = temp.path().join("missing-receipt.json");
    let output = Command::new(lightweight_binary())
        .args([
            "--expected-input-receipt",
            missing.to_str().expect("UTF-8 receipt path"),
            "--seed",
            "7",
            "--candidates",
            "2",
            "--max-indicators",
            "1",
            "--stop-multiple",
            "1.0",
            "--target-multiple",
            "2.0",
            "--out",
            temp.path()
                .join("unused.json")
                .to_str()
                .expect("UTF-8 output path"),
            "--root",
            temp.path().to_str().expect("UTF-8 root path"),
        ])
        .output()
        .expect("run model-free historical search preflight");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--expected-input-receipt")
            && stderr.contains("before feature computation"),
        "missing receipt must fail at the exact boundary, stderr:\n{stderr}"
    );
    assert!(!stderr.contains("symbol"));
}

#[test]
fn lightweight_search_writes_a_complete_receipt_bound_research_only_artifact() {
    let temp = TempRoot::new("end-to-end");
    let identity = publish_fixture(temp.path(), "historical-search-cli-e2e");
    let selected = ExactCanonicalSeries::open(temp.path(), identity)
        .expect("select fixture series")
        .load_search_input(&[])
        .expect("build exact fixture receipt");
    let receipt = selected.receipt().expect("canonical input receipt");
    let expected_receipt_sha256 = receipt.identity_sha256().expect("receipt identity");
    let receipt_path = temp.path().join("expected-receipt.json");
    fs::write(
        &receipt_path,
        receipt.to_json_bytes().expect("serialize exact receipt"),
    )
    .expect("write expected receipt fixture");
    let output_path = temp.path().join("historical-search.json");

    let output = Command::new(lightweight_binary())
        .args([
            "--expected-input-receipt",
            receipt_path.to_str().expect("UTF-8 receipt path"),
            "--seed",
            "42",
            "--candidates",
            "2",
            "--max-indicators",
            "1",
            "--stop-multiple",
            "1.0",
            "--target-multiple",
            "2.0",
            "--out",
            output_path.to_str().expect("UTF-8 output path"),
            "--root",
            temp.path().to_str().expect("UTF-8 root path"),
        ])
        .output()
        .expect("run model-free historical candidate search");
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = fs::read(&output_path).expect("read historical-search artifact");
    let artifact: serde_json::Value =
        serde_json::from_slice(&bytes).expect("decode historical-search artifact");
    assert_eq!(artifact["schema_version"], 2);
    assert_eq!(artifact["input_receipt_sha256"], expected_receipt_sha256);
    assert_eq!(artifact["candidate_generation"]["candidate_count"], 2);
    assert_eq!(
        artifact["candidate_generation"]["signal_rules"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(artifact["search"]["artifact_class"], "research_only");
    assert_eq!(
        artifact["search"]["promotion_eligibility"],
        "not_promotion_eligible"
    );
    assert_eq!(artifact["search"]["backend"], "cpu_only");
    assert_eq!(
        artifact["search"]["contract"]["scope"]["receipt_sha256"],
        expected_receipt_sha256
    );
    assert!(
        artifact["search"]["search_identity_sha256"]
            .as_str()
            .is_some_and(|identity| identity.len() == 64)
    );
    assert!(
        artifact["search"]["contract"]["ranking_policy_id"]
            .as_str()
            .is_some_and(|identity| !identity.is_empty())
    );
    assert_eq!(
        artifact["search"]["contract"]["ordered_candidate_identities"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(
        artifact["candidate_generation"]["signal_rules"][0]["candidate_identity_sha256"],
        artifact["search"]["contract"]["ordered_candidate_identities"][0]
    );
    let lower = String::from_utf8(bytes)
        .expect("artifact is UTF-8")
        .to_ascii_lowercase();
    assert!(!lower.contains("net_profit"));
    assert!(!lower.contains("\"pnl"));

    let stdout = String::from_utf8_lossy(&output.stdout);
    for required in [
        "receipt_sha256=",
        "search_identity_sha256=",
        "ranking_policy_id=",
        "artifact_class=ResearchOnly",
        "promotion_eligibility=NotPromotionEligible",
        "accounting=GrossReferenceR",
    ] {
        assert!(
            stdout.contains(required),
            "stdout omits {required}:\n{stdout}"
        );
    }
}

#[test]
fn lightweight_search_artifact_is_byte_identical_across_parent_cpu_widths() {
    let temp = TempRoot::new("cpu-width-parity");
    let identity = publish_fixture(temp.path(), "historical-search-cli-width-parity");
    let selected = ExactCanonicalSeries::open(temp.path(), identity)
        .expect("select width-parity fixture")
        .load_search_input(&[])
        .expect("build width-parity exact receipt");
    let receipt_path = temp.path().join("expected-receipt.json");
    fs::write(
        &receipt_path,
        selected
            .receipt()
            .expect("width-parity receipt")
            .to_json_bytes()
            .expect("serialize width-parity receipt"),
    )
    .expect("write width-parity receipt");
    let width_one_path = temp.path().join("width-one.json");
    let automatic_path = temp.path().join("automatic.json");

    let common = [
        "--expected-input-receipt",
        receipt_path.to_str().expect("UTF-8 receipt path"),
        "--seed",
        "91",
        "--candidates",
        "2",
        "--max-indicators",
        "1",
        "--stop-multiple",
        "1.0",
        "--target-multiple",
        "2.0",
        "--root",
        temp.path().to_str().expect("UTF-8 root path"),
    ];
    let width_one = Command::new(lightweight_binary())
        .args(common)
        .args([
            "--cpu-threads",
            "1",
            "--out",
            width_one_path.to_str().expect("UTF-8 width-one path"),
        ])
        .output()
        .expect("run one-worker historical search");
    assert!(
        width_one.status.success(),
        "one-worker stderr:\n{}",
        String::from_utf8_lossy(&width_one.stderr)
    );
    let automatic = Command::new(lightweight_binary())
        .args(common)
        .args([
            "--out",
            automatic_path.to_str().expect("UTF-8 automatic path"),
        ])
        .output()
        .expect("run automatic-width historical search");
    assert!(
        automatic.status.success(),
        "automatic-width stderr:\n{}",
        String::from_utf8_lossy(&automatic.stderr)
    );

    assert_eq!(
        fs::read(width_one_path).expect("read one-worker artifact"),
        fs::read(automatic_path).expect("read automatic-width artifact"),
        "parent CPU assignment must not change canonical search evidence"
    );
}
