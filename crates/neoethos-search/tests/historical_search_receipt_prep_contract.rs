use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity,
    CanonicalOhlcvPublishRequest, CanonicalTimeframe, CanonicalVolumeRef, Ohlcv,
    SelectedDatasetGenerationV1, publish_canonical_ohlcv_generation,
};
use neoethos_search::CanonicalSearchInputReceiptV2;

static NEXT_TEMP_ROOT: AtomicU64 = AtomicU64::new(0);

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        let unique = NEXT_TEMP_ROOT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "neoethos-historical-search-receipt-prep-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create isolated receipt-preparation root");
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
                "ERROR historical-search receipt-preparation cleanup failed for {}: {error}",
                self.0.display()
            );
        }
    }
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn preparation_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_neoethos-prepare-historical-search-receipt").expect(
        "the model-free neoethos-prepare-historical-search-receipt binary target must exist",
    )
}

fn historical_search_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_neoethos-historical-search")
        .expect("the model-free neoethos-historical-search binary target must exist")
}

fn ctrader_identity(
    account_id: i64,
    symbol_id: i64,
    symbol: &str,
    timeframe: CanonicalTimeframe,
) -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "Broker-Demo",
        account_id,
        symbol_id,
        symbol,
        timeframe,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid cTrader fixture identity")
}

fn publish(
    root: &Path,
    identity: &CanonicalDatasetIdentity,
    expected_generation: Option<&str>,
    close_offset: f64,
    provenance_label: &str,
) -> SelectedDatasetGenerationV1 {
    let period_ms = identity
        .timeframe()
        .fixed_duration_ms()
        .expect("fixture uses a fixed-duration timeframe");
    let rows = 512_i64;
    let start_ms = 1_704_067_200_000_i64;
    let timestamp = (0..rows)
        .map(|row| start_ms + row * period_ms)
        .collect::<Vec<_>>();
    let close = (0..rows)
        .map(|row| close_offset + row as f64 * 0.000_01 + (row as f64 / 11.0).sin() * 0.000_3)
        .collect::<Vec<_>>();
    let open = std::iter::once(close[0])
        .chain(close.iter().take(close.len().saturating_sub(1)).cloned())
        .collect::<Vec<_>>();
    let high = open
        .iter()
        .zip(&close)
        .map(|(open, close)| open.max(*close) + 0.000_2)
        .collect();
    let low = open
        .iter()
        .zip(&close)
        .map(|(open, close)| open.min(*close) - 0.000_2)
        .collect();
    let ohlcv = Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: Some((0..rows).map(|row| 100.0 + row as f64).collect()),
    };
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.historical-search-receipt-prep-fixture.v1",
        provenance_label.as_bytes().to_vec(),
    )
    .expect("valid fixture provenance");
    let published = publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity,
        expected_generation,
        provenance: &provenance,
        ohlcv: &ohlcv,
        volume: CanonicalVolumeRef::Float64(
            ohlcv.volume.as_deref().expect("fixture volume is present"),
        ),
        rows_per_chunk: 128,
    })
    .expect("publish canonical fixture generation");
    SelectedDatasetGenerationV1::from_manifest(published.manifest())
        .expect("select the exact published generation")
}

fn fake_selected(
    identity: CanonicalDatasetIdentity,
    discriminator: u64,
) -> SelectedDatasetGenerationV1 {
    SelectedDatasetGenerationV1::new(
        identity,
        format!("g1-{discriminator:064x}.vortex"),
        format!("{:064x}", discriminator + 10_000),
    )
    .expect("valid synthetic selected generation")
}

fn write_selected(path: &Path, selected: &SelectedDatasetGenerationV1) {
    fs::write(
        path,
        selected
            .to_json_bytes()
            .expect("serialize selected generation"),
    )
    .expect("write selected-generation fixture");
}

fn run_preparation(
    root: &Path,
    anchor_path: &Path,
    direct_paths: &[&Path],
    output_path: &Path,
) -> Output {
    let mut command = Command::new(preparation_binary());
    command.args([
        "--root",
        root.to_str().expect("UTF-8 canonical root"),
        "--anchor-selected-generation",
        anchor_path.to_str().expect("UTF-8 anchor path"),
    ]);
    for direct in direct_paths {
        command.args([
            "--direct-selected-generation",
            direct.to_str().expect("UTF-8 direct path"),
        ]);
    }
    command
        .args(["--out", output_path.to_str().expect("UTF-8 output path")])
        .output()
        .expect("run model-free receipt preparation")
}

fn assert_failed_without_output(output: &Output, path: &Path, expected: &str) {
    assert!(!output.status.success(), "command unexpectedly succeeded");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_ascii_lowercase().contains(expected),
        "stderr omits expected `{expected}`:\n{stderr}"
    );
    assert!(
        !path.exists(),
        "failure must occur before creating feature/receipt output {}",
        path.display()
    );
}

#[test]
fn preparation_and_replay_share_one_exact_selected_frame_recipe_without_fallbacks() {
    let repository = repository_root();
    let prep_path = repository.join("crates/neoethos-search/src/historical_search_receipt_prep.rs");
    let prep = fs::read_to_string(&prep_path).expect("read receipt-preparation source");
    let replay =
        fs::read_to_string(repository.join("crates/neoethos-search/src/historical_search_cli.rs"))
            .expect("read historical replay source");
    let lib = fs::read_to_string(repository.join("crates/neoethos-search/src/lib.rs"))
        .expect("read search library root");
    let bin = fs::read_to_string(
        repository
            .join("crates/neoethos-search/src/bin/neoethos-prepare-historical-search-receipt.rs"),
    )
    .expect("read model-free receipt-preparation binary");

    for required in [
        "SelectedDatasetGenerationV1::from_json_bytes",
        "CanonicalDatasetSeriesReceiptV1::new",
        "load_exact_canonical_timeframe",
        "pub fn build_exact_selected_feature_input(",
        "CanonicalSearchInputReceiptV2::from_feature_frame",
        "atomic_write_create_new",
    ] {
        assert!(
            prep.contains(required),
            "receipt preparation omits strict boundary `{required}`"
        );
    }
    assert_eq!(
        prep.matches("prepare_multitimeframe_features(").count(),
        1,
        "there must be exactly one selected-frame feature recipe"
    );
    assert!(
        replay.contains("build_exact_selected_feature_input("),
        "historical replay does not reuse the exact selected-frame builder"
    );
    assert!(
        !replay.contains("prepare_multitimeframe_features("),
        "historical replay retains a second feature recipe"
    );
    for forbidden in [
        "load_canonical_timeframe(",
        "load_dataset_for_identity_with_timeframes(",
        "ExactCanonicalSeries",
        "inventory_for_symbol(",
        "open_current_dataset_generation(",
        "resample",
    ] {
        assert!(
            !prep.contains(forbidden),
            "preparation reaches forbidden fallback `{forbidden}`"
        );
    }
    assert!(lib.contains("pub mod historical_search_receipt_prep;"));
    assert!(bin.contains("initialize_source_seal_before_runtime"));
    assert!(bin.contains("install_historical_search_process_budget(&args)?;"));
    assert!(bin.contains("historical_search_receipt_prep::run(&args[1..])"));
}

#[test]
fn m15_and_direct_h1_prepare_an_exact_receipt_consumed_by_research_only_replay() {
    let temp = TempRoot::new("m15-h1-replay");
    let anchor_identity = ctrader_identity(42, 1001, "EURUSD", CanonicalTimeframe::M15);
    let higher_identity = ctrader_identity(42, 1001, "EURUSD", CanonicalTimeframe::H1);
    let anchor = publish(temp.path(), &anchor_identity, None, 1.10, "m15-generation");
    let higher = publish(temp.path(), &higher_identity, None, 1.20, "h1-generation");
    let anchor_path = temp.path().join("selected-m15.json");
    let higher_path = temp.path().join("selected-h1.json");
    write_selected(&anchor_path, &anchor);
    write_selected(&higher_path, &higher);
    let receipt_path = temp.path().join("historical-search-receipt.json");

    let prepared = run_preparation(
        temp.path(),
        &anchor_path,
        &[higher_path.as_path()],
        &receipt_path,
    );
    assert!(
        prepared.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&prepared.stdout),
        String::from_utf8_lossy(&prepared.stderr)
    );
    let receipt_bytes = fs::read(&receipt_path).expect("read prepared receipt");
    let receipt = CanonicalSearchInputReceiptV2::from_json_bytes(&receipt_bytes)
        .expect("decode prepared exact receipt");
    assert_eq!(
        receipt.anchor_dataset_identity(),
        anchor.identity().to_path_component()
    );
    assert_eq!(receipt.source_bindings().len(), 2);
    for selected in [&anchor, &higher] {
        let binding = receipt
            .source_bindings()
            .iter()
            .find(|binding| binding.dataset_identity() == selected.identity().to_path_component())
            .expect("selected generation has one receipt source binding");
        assert_eq!(binding.generation_id(), selected.generation_id());
        assert_eq!(
            binding.manifest_sha256(),
            selected.manifest_binding_sha256()
        );
    }
    let stdout = String::from_utf8_lossy(&prepared.stdout);
    assert!(stdout.contains("receipt_sha256="), "stdout:\n{stdout}");
    assert!(stdout.contains("output="), "stdout:\n{stdout}");

    let artifact_path = temp.path().join("historical-search.json");
    let replay = Command::new(historical_search_binary())
        .args([
            "--expected-input-receipt",
            receipt_path.to_str().expect("UTF-8 receipt path"),
            "--seed",
            "17",
            "--candidates",
            "1",
            "--max-indicators",
            "1",
            "--stop-multiple",
            "1.0",
            "--target-multiple",
            "2.0",
            "--root",
            temp.path().to_str().expect("UTF-8 root"),
            "--out",
            artifact_path.to_str().expect("UTF-8 artifact path"),
        ])
        .output()
        .expect("run exact receipt-bound historical replay");
    assert!(
        replay.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&replay.stdout),
        String::from_utf8_lossy(&replay.stderr)
    );
    let artifact: serde_json::Value =
        serde_json::from_slice(&fs::read(&artifact_path).expect("read historical-search artifact"))
            .expect("decode historical-search artifact");
    assert_eq!(
        artifact["input_receipt"],
        serde_json::to_value(&receipt).unwrap()
    );
    assert_eq!(artifact["search"]["artifact_class"], "research_only");
    assert_eq!(
        artifact["search"]["promotion_eligibility"],
        "not_promotion_eligible"
    );
}

#[test]
fn stale_selected_generation_fails_preparation_and_replay_without_falling_forward() {
    let temp = TempRoot::new("stale-selection");
    let anchor_identity = ctrader_identity(42, 1001, "EURUSD", CanonicalTimeframe::M15);
    let higher_identity = ctrader_identity(42, 1001, "EURUSD", CanonicalTimeframe::H1);
    let anchor = publish(
        temp.path(),
        &anchor_identity,
        None,
        1.10,
        "m15-generation-n",
    );
    let higher = publish(temp.path(), &higher_identity, None, 1.20, "h1-generation-n");
    let anchor_path = temp.path().join("selected-m15.json");
    let higher_path = temp.path().join("selected-h1.json");
    write_selected(&anchor_path, &anchor);
    write_selected(&higher_path, &higher);
    let original_receipt_path = temp.path().join("receipt-before-current-advance.json");
    let initial = run_preparation(
        temp.path(),
        &anchor_path,
        &[higher_path.as_path()],
        &original_receipt_path,
    );
    assert!(
        initial.status.success(),
        "initial preparation stderr:\n{}",
        String::from_utf8_lossy(&initial.stderr)
    );

    let newer_anchor = publish(
        temp.path(),
        &anchor_identity,
        Some(anchor.generation_id()),
        2.10,
        "m15-generation-n-plus-one",
    );
    assert_ne!(newer_anchor.generation_id(), anchor.generation_id());

    let stale_output = temp.path().join("stale-preparation.json");
    let stale_preparation = run_preparation(
        temp.path(),
        &anchor_path,
        &[higher_path.as_path()],
        &stale_output,
    );
    assert_failed_without_output(&stale_preparation, &stale_output, "current");

    let replay_output = temp.path().join("stale-replay.json");
    let stale_replay = Command::new(historical_search_binary())
        .args([
            "--expected-input-receipt",
            original_receipt_path
                .to_str()
                .expect("UTF-8 original receipt path"),
            "--seed",
            "19",
            "--candidates",
            "1",
            "--max-indicators",
            "1",
            "--stop-multiple",
            "1.0",
            "--target-multiple",
            "2.0",
            "--root",
            temp.path().to_str().expect("UTF-8 root"),
            "--out",
            replay_output.to_str().expect("UTF-8 replay output"),
        ])
        .output()
        .expect("run stale historical replay");
    assert_failed_without_output(&stale_replay, &replay_output, "current");
}

#[test]
fn malformed_selected_inputs_fail_before_feature_or_receipt_output() {
    let temp = TempRoot::new("invalid-inputs");
    let anchor_identity = ctrader_identity(42, 1001, "EURUSD", CanonicalTimeframe::M15);
    let higher_identity = ctrader_identity(42, 1001, "EURUSD", CanonicalTimeframe::H1);
    let anchor = publish(temp.path(), &anchor_identity, None, 1.10, "valid-m15");
    let higher = publish(temp.path(), &higher_identity, None, 1.20, "valid-h1");
    let anchor_path = temp.path().join("selected-m15.json");
    let higher_path = temp.path().join("selected-h1.json");
    write_selected(&anchor_path, &anchor);
    write_selected(&higher_path, &higher);

    let missing_anchor_output = temp.path().join("missing-anchor.json");
    let missing_anchor = Command::new(preparation_binary())
        .args([
            "--root",
            temp.path().to_str().expect("UTF-8 root"),
            "--direct-selected-generation",
            higher_path.to_str().expect("UTF-8 direct path"),
            "--out",
            missing_anchor_output.to_str().expect("UTF-8 output"),
        ])
        .output()
        .expect("run missing-anchor preparation");
    assert_failed_without_output(&missing_anchor, &missing_anchor_output, "anchor");

    let duplicate_output = temp.path().join("duplicate.json");
    let duplicate = run_preparation(
        temp.path(),
        &anchor_path,
        &[anchor_path.as_path()],
        &duplicate_output,
    );
    assert_failed_without_output(&duplicate, &duplicate_output, "repeats direct timeframe");

    for (label, foreign) in [
        (
            "foreign-account",
            fake_selected(
                ctrader_identity(99, 1001, "EURUSD", CanonicalTimeframe::H1),
                1,
            ),
        ),
        (
            "foreign-symbol",
            fake_selected(
                ctrader_identity(42, 2002, "GBPUSD", CanonicalTimeframe::H1),
                2,
            ),
        ),
    ] {
        let foreign_path = temp.path().join(format!("{label}.json"));
        write_selected(&foreign_path, &foreign);
        let output_path = temp.path().join(format!("{label}-receipt.json"));
        let output = run_preparation(
            temp.path(),
            &anchor_path,
            &[foreign_path.as_path()],
            &output_path,
        );
        assert_failed_without_output(&output, &output_path, "different source/account series");
    }

    let mut unknown: serde_json::Value = serde_json::from_slice(
        &higher
            .to_json_bytes()
            .expect("serialize strict selected generation"),
    )
    .expect("parse selected-generation fixture");
    unknown["legacy_current"] = serde_json::Value::Bool(true);
    let unknown_path = temp.path().join("unknown-field.json");
    fs::write(&unknown_path, serde_json::to_vec(&unknown).unwrap())
        .expect("write unknown-field fixture");
    let unknown_output = temp.path().join("unknown-field-receipt.json");
    let unknown_result = run_preparation(
        temp.path(),
        &anchor_path,
        &[unknown_path.as_path()],
        &unknown_output,
    );
    assert_failed_without_output(&unknown_result, &unknown_output, "unknown field");

    let mut tampered: serde_json::Value = serde_json::from_slice(
        &higher
            .to_json_bytes()
            .expect("serialize strict selected generation"),
    )
    .expect("parse selected-generation fixture");
    tampered["manifest_binding_sha256"] = serde_json::Value::String("0".repeat(64));
    let tampered_path = temp.path().join("tampered-binding.json");
    fs::write(&tampered_path, serde_json::to_vec(&tampered).unwrap())
        .expect("write tampered-binding fixture");
    let tampered_output = temp.path().join("tampered-binding-receipt.json");
    let tampered_result = run_preparation(
        temp.path(),
        &anchor_path,
        &[tampered_path.as_path()],
        &tampered_output,
    );
    assert_failed_without_output(&tampered_result, &tampered_output, "manifest");
}

#[test]
fn existing_output_is_refused_before_selected_input_reads() {
    let temp = TempRoot::new("overwrite");
    let output_path = temp.path().join("existing.json");
    let sentinel = b"do-not-overwrite";
    fs::write(&output_path, sentinel).expect("write output sentinel");
    let missing_anchor = temp.path().join("missing-anchor.json");
    let missing_direct = temp.path().join("missing-direct.json");

    let output = run_preparation(
        temp.path(),
        &missing_anchor,
        &[missing_direct.as_path()],
        &output_path,
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr).to_ascii_lowercase();
    assert!(stderr.contains("overwrite"), "stderr:\n{stderr}");
    assert!(
        !stderr.contains("read --anchor-selected-generation"),
        "unsafe output must be rejected before any selected input is read: {stderr}"
    );
    assert_eq!(fs::read(&output_path).expect("read sentinel"), sentinel);
}
