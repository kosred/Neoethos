//! Compile-free guard for the Gate-1 search-state identity boundary.
//!
//! This test is deliberately runnable with `rustc --test`: the shared workspace
//! can have a long hardware/data test holding Cargo's build lane, but a source
//! regression must still go RED before production code changes.

#[test]
fn discovery_ledger_is_versioned_and_receipt_bound() {
    let source = include_str!("../src/discovery_ledger.rs");

    for required in [
        "neoethos.discovery_search_ledger.v3",
        "search_input_receipt: CanonicalSearchInputReceiptV2",
        "search_input_receipt_sha256: String",
        "pub config_hash: String",
        "#[serde(deny_unknown_fields)]",
        "expected_receipt: &CanonicalSearchInputReceiptV2",
        "expected_config_hash: &str",
        "Result<Option<DiscoverySearchLedger>>",
        "legacy unbound discovery ledger",
        "embedded trial-returns manifest does not match",
    ] {
        assert!(
            source.contains(required),
            "discovery ledger is missing strict receipt contract fragment `{required}`"
        );
    }

    assert!(
        !source.contains("pub fn load_prior_ledger(cache_dir: &str, symbol: &str, tf: &str)"),
        "the production loader still exposes symbol/TF-only authority"
    );
}

#[test]
fn trial_return_manifest_and_binary_are_receipt_bound() {
    let source = include_str!("../src/trial_returns.rs");

    for required in [
        "neoethos.trial_returns.v3",
        "search_input_receipt: CanonicalSearchInputReceiptV2",
        "search_input_receipt_sha256: String",
        "pub config_hash: String",
        "expected_receipt: &CanonicalSearchInputReceiptV2",
        "expected_config_hash: &str",
        "Result<Option<TrialReturnsManifest>>",
        "legacy unbound trial-returns manifest",
        "trial-returns binary hash mismatch",
        "trial-returns binary file mismatch",
        "orphaned receipt-bound trial-returns binary",
    ] {
        assert!(
            source.contains(required),
            "trial returns are missing strict receipt contract fragment `{required}`"
        );
    }

    assert!(
        !source.contains("pub config_hash: Option<String>"),
        "the receipt-bound manifest still permits an unattributable config"
    );
}

#[test]
fn discovery_passes_one_receipt_and_config_identity_to_all_state_edges() {
    let source = include_str!("../src/discovery.rs");

    for required in [
        "pub search_config_hash: String",
        "search_config_hash: search_state_config_hash.to_string()",
        "search_input_receipt,\n            &search_state_config_hash",
        "search_input_receipt,\n                search_state_config_hash,",
        "search_input_receipt,\n            &search_state_config_hash,",
    ] {
        assert!(
            source.contains(required),
            "discovery state edge is not bound to the run receipt/config: `{required}`"
        );
    }
    assert_eq!(
        source.matches("result.search_config_hash.clone()").count(),
        2,
        "portfolio and promotion envelopes must carry the exact result config identity"
    );
}

#[test]
fn trial_statistics_reader_has_no_symbol_timeframe_only_authority() {
    let source = include_str!("../src/deflated.rs");
    let lib_source = include_str!("../src/lib.rs");

    for required in [
        "expected_receipt: &CanonicalSearchInputReceiptV2",
        "expected_config_hash: &str",
        "load_manifest(\n        cache_dir,\n        symbol,\n        tf,\n        expected_receipt,\n        expected_config_hash,",
    ] {
        assert!(
            source.contains(required),
            "trial-statistics reader is missing receipt-bound fragment `{required}`"
        );
    }
    assert!(
        !source.contains("pub fn read_matrix(cache_dir: &str, symbol: &str, tf: &str)"),
        "trial-statistics reader still exposes symbol/TF-only binary authority"
    );
    assert!(
        !source.contains("pub fn read_matrix_at("),
        "trial-statistics reader still exposes a public raw-path bypass around receipt validation"
    );
    assert!(
        !lib_source.contains("read_matrix_at as read_trial_matrix_at"),
        "search lib still exports the legacy raw-path matrix authority"
    );
}
