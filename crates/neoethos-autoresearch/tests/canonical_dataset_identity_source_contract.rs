use std::path::{Path, PathBuf};

fn repository_root() -> PathBuf {
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        return Path::new(manifest_dir)
            .parent()
            .and_then(Path::parent)
            .expect("neoethos-autoresearch manifest must be under <repo>/crates")
            .to_path_buf();
    }
    std::env::current_dir().expect("standalone source contract working directory")
}

fn read(relative: &str) -> String {
    let path = repository_root().join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

#[test]
fn autoresearch_consumes_one_exact_direct_canonical_series() {
    let runner = read("crates/neoethos-autoresearch/src/runner.rs");
    let streaming = read("crates/neoethos-autoresearch/src/runner/streaming.rs");
    let session = read("crates/neoethos-autoresearch/src/session.rs");
    let journal = read("crates/neoethos-autoresearch/src/journal.rs");
    let verdict = read("crates/neoethos-autoresearch/src/verdict.rs");
    let canonical_ohlcv = read("crates/neoethos-data/src/core/canonical_ohlcv.rs");
    let data = read("crates/neoethos-data/src/lib.rs");
    let cli = read("crates/neoethos-cli/src/main.rs");
    let autoresearch_cli = cli
        .split_once("fn cmd_autoresearch(args: &[String]) -> Result<()> {")
        .expect("autoresearch CLI entrypoint")
        .1
        .split_once("\nfn autoresearch_help()")
        .expect("autoresearch CLI boundary")
        .0;
    let promotion_portfolio = runner
        .split_once("pub struct PromotionPortfolio {")
        .expect("promotion portfolio declaration")
        .1
        .split_once("\n}\n\nimpl PromotionPortfolio")
        .expect("promotion portfolio boundary")
        .0;
    let scratch_manifest = streaming
        .split_once("struct ScratchLedgerManifestV1 {")
        .expect("scratch manifest declaration")
        .1
        .split_once("\n}\n\nimpl ScratchLedgerManifestV1")
        .expect("scratch manifest boundary")
        .0;
    let verdict_production = verdict
        .split_once("#[cfg(test)]")
        .expect("verdict production/test boundary")
        .0;

    assert!(
        runner.contains("pub dataset_identity: neoethos_data::CanonicalDatasetIdentity")
            && !runner.contains("pub symbol: String"),
        "RunArgs must carry the typed canonical identity instead of a bare symbol"
    );
    assert!(
        runner.contains("let symbol = dataset_identity.symbol_name().to_owned()")
            && runner
                .contains("let base_timeframe = dataset_identity.timeframe().as_str().to_owned()"),
        "autoresearch must derive its display symbol and base timeframe from the selected identity"
    );
    assert!(
        !runner.contains("args.symbol")
            && runner
                .contains("assert_identity_matches_config(&args.dataset_identity, &base_config)"),
        "both public run entry points must reject identity/config drift and no stale bare-symbol read may remain"
    );
    assert!(
        streaming.contains("pin_exact_canonical_series_v1")
            && streaming.contains("dispatch_canonical_discovery_data_preparation_v3")
            && streaming.contains("into_cpu_dataset_after_no_physical_gpu_v1")
            && streaming.contains("require_direct_timeframes")
            && !streaming.contains("load_dataset_for_identity_with_timeframes")
            && !streaming.contains("load_symbol_dataset("),
        "the executor must pin exact direct generations before admission and may decode only behind the selected CPU authority"
    );
    assert!(
        !streaming.contains("source_artifacts: full.source_artifacts.clone()")
            && !streaming.contains("in_sample: neoethos_data::SymbolDataset")
            && streaming.contains("prepare_multitimeframe_features_before_with_options"),
        "in-sample execution must use segment-aware canonical frames and must never attach full-generation artifacts to sliced OHLCV"
    );
    assert!(
        streaming.contains("CanonicalSearchInputReceiptV2::from_feature_frame(")
            && streaming.contains("validate_search_receipt_against_dataset_receipt(")
            && streaming.contains(
                "CanonicalSearchRunInputV2::new(search_receipt, features, &self.in_sample_base)",
            )
            && streaming
                .contains("run_discovery_cycle_with_holdout(\n                    &run_input,")
            && !streaming
                .contains("run_discovery_cycle_with_holdout(\n                    features,"),
        "autoresearch must cross-check its frozen session receipt and pass a receipt-bound typed input to search"
    );
    assert!(
        streaming.contains("last_search_receipt = Some(result.search_input_receipt.clone())")
            && streaming.contains("let (evidence_receipt, evidence_config_hash) = best")
            && streaming.contains("neoethos_search::trial_returns::load_manifest(")
            && streaming.contains("evidence_receipt,\n            evidence_config_hash,")
            && streaming.contains("neoethos_search::trial_returns::binary_path(")
            && streaming
                .matches("neoethos_search::deflated::read_matrix(")
                .count()
                == 2
            && !streaming.contains("neoethos_search::deflated::read_matrix_at("),
        "autoresearch must load/copy trial evidence only through the exact search receipt plus config address"
    );
    assert!(
        canonical_ohlcv.contains("source_segment: SourceSegmentV1")
            && canonical_ohlcv.contains("pub fn row_window(")
            && canonical_ohlcv.contains("pub fn prefix_before_timestamp_ms(")
            && canonical_ohlcv.contains("pub fn source_binding("),
        "canonical OHLCV windows must retain a typed exact consumed source segment"
    );
    assert!(
        !data.contains(".artifact().source_binding(")
            && data.contains("prepare_multitimeframe_features_before_with_options"),
        "every feature-plan source binding must come from the segment-aware frame"
    );
    assert!(
        data.contains("pub fn vortex_feature_run_root() -> PathBuf")
            && data.contains("let scratch_root = vortex_feature_run_root();"),
        "the Vortex feature-run scratch root must have one canonical public authority"
    );
    assert!(
        session.contains("pub struct DatasetReceiptV1")
            && session.contains("pub dataset_receipt: DatasetReceiptV1")
            && session.contains("DatasetReceiptChanged")
            && journal.contains("dataset_receipt: DatasetReceiptV1"),
        "the exact anchor/direct-generation/window receipt must be frozen in the header, journal fold, and resume gate"
    );
    assert!(
        promotion_portfolio.contains("pub session_id: SessionId")
            && promotion_portfolio.contains("pub dataset_receipt: DatasetReceiptV1")
            && promotion_portfolio.contains("pub batch_bindings: Vec<PromotionBatchBindingV5>")
            && !promotion_portfolio.contains("pub genes: Vec<")
            && runner.contains("expected_session_id: &SessionId")
            && runner.contains("if &self.session_id != expected_session_id")
            && runner.contains("expected_dataset_receipt: &DatasetReceiptV1")
            && streaming.contains("session_id: request.session_id.clone()"),
        "promotion evidence itself, its writer, and its reader must exact-bind both session and full receipt"
    );
    assert!(
        scratch_manifest.contains("session_id: crate::session::SessionId")
            && scratch_manifest.contains("dataset_receipt: DatasetReceiptV1")
            && streaming.contains("session_id: (*request.session_id).clone()")
            && streaming.contains("dataset_receipt: (*request.dataset_receipt).clone()")
            && streaming.contains("observed == *expected"),
        "the scratch manifest itself must carry and exact-compare session plus full receipt"
    );
    assert!(
        verdict_production
            .matches("pub dataset_receipt: DatasetReceiptV1")
            .count()
            == 2
            && verdict_production.contains("--dataset-identity {} --session {}")
            && !verdict_production.contains("--symbol {} --session {}"),
        "both verdict context and emitted verdict must carry the receipt, and reproduction must use its encoded identity"
    );
    assert!(
        streaming.contains("signals_for_gene(&features, &tagged.gene)")
            && streaming
                .contains(r#"context("building exact batch-bound OOS signals_for_gene")"#)
            && streaming.contains("to_dense_samples_major()")
            && streaming.contains(
                r#"context("materializing the exact f64/validity feature frame for the shuffle control")"#,
            )
            && streaming.contains("dense.validity")
            && streaming.contains("FeatureData::InMemory")
            && !streaming.contains("FeatureFrame::from_array"),
        "streaming must propagate typed API errors and shuffle f64 values with their validity without rebuilding away provenance"
    );
    assert!(
        autoresearch_cli.contains("inventory_canonical_identities")
            && autoresearch_cli.contains("select_exact_runtime_identity")
            && autoresearch_cli.contains("RunArgs::new(dataset_identity)"),
        "the CLI must inventory, select, and pass one exact canonical dataset identity"
    );
}
