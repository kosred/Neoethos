use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use neoethos_broker_truth::{
    BrokerFinancialOperationV1, BrokerFinancialTruthArtifactSourceV1,
    BrokerFinancialTruthBindingV1, BrokerFinancialTruthBundleManifestV1,
    BrokerFinancialTruthBundleStoreV1, BrokerFinancialTruthStoreErrorCodeV1,
    BrokerFinancialTruthVortexSchemaV1, EvidenceWindowV1, ExactCapturedEvidencePairV1,
    ExactConversionRouteEvidenceV1, ExactQuoteSideEvidenceV1, ImmutableVortexArtifactV1,
    QuoteSideV1, SynchronizedBidAskEvidenceV1, current_broker_financial_truth_capability_v1,
};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalTimeframe,
};

struct Fixture {
    root: PathBuf,
    store: BrokerFinancialTruthBundleStoreV1,
    binding: BrokerFinancialTruthBindingV1,
    manifest: BrokerFinancialTruthBundleManifestV1,
    sources: Vec<BrokerFinancialTruthArtifactSourceV1>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn unique_temp_root(test_name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "neoethos-broker-truth-{test_name}-{}-{nonce}",
        std::process::id()
    ))
}

fn captured_pair(
    source_root: &Path,
    stem: &str,
    raw_schema: BrokerFinancialTruthVortexSchemaV1,
    decoded_schema: BrokerFinancialTruthVortexSchemaV1,
) -> (
    ExactCapturedEvidencePairV1,
    Vec<BrokerFinancialTruthArtifactSourceV1>,
) {
    let mut sources = Vec::new();
    let artifact = |suffix: &str,
                    schema: BrokerFinancialTruthVortexSchemaV1,
                    sources: &mut Vec<BrokerFinancialTruthArtifactSourceV1>| {
        let relative_path = format!("{stem}-{suffix}.vortex");
        let source_path = source_root.join(&relative_path);
        std::fs::write(
            &source_path,
            format!("synthetic store-integrity fixture: {stem}/{suffix}"),
        )
        .expect("write source artifact");
        let reference =
            ImmutableVortexArtifactV1::from_file(relative_path.clone(), schema, 1, &source_path)
                .expect("inspect source artifact");
        sources.push(
            BrokerFinancialTruthArtifactSourceV1::new(relative_path, source_path)
                .expect("source mapping"),
        );
        reference
    };

    let raw = artifact("raw", raw_schema, &mut sources);
    let decoded = artifact("decoded", decoded_schema, &mut sources);
    (ExactCapturedEvidencePairV1::new(raw, decoded), sources)
}

fn fixture(test_name: &str) -> Fixture {
    let root = unique_temp_root(test_name);
    let source_root = root.join("sources");
    std::fs::create_dir_all(&source_root).expect("create source root");

    let window = EvidenceWindowV1::new(1_700_000_000_000, 1_700_000_060_000)
        .expect("valid half-open window");
    let identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        7,
        42,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("canonical cTrader identity");
    let binding = BrokerFinancialTruthBindingV1::new(
        &identity,
        "11".repeat(32),
        window,
        1,
        "EUR",
        2,
        "USD",
        2,
        "USD",
    )
    .expect("exact run binding");

    let (bid_pair, mut sources) = captured_pair(
        &source_root,
        "primary-bid",
        BrokerFinancialTruthVortexSchemaV1::CTraderTickPagesRawV1,
        BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV1,
    );
    let (ask_pair, ask_sources) = captured_pair(
        &source_root,
        "primary-ask",
        BrokerFinancialTruthVortexSchemaV1::CTraderTickPagesRawV1,
        BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV1,
    );
    sources.extend(ask_sources);
    let (synchronization_pair, synchronization_sources) = captured_pair(
        &source_root,
        "quote-synchronization",
        BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV1,
        BrokerFinancialTruthVortexSchemaV1::CTraderQuoteReplayRulesDecodedV1,
    );
    sources.extend(synchronization_sources);

    let bid = ExactQuoteSideEvidenceV1::new(
        QuoteSideV1::Bid,
        42,
        "EURUSD",
        1,
        2,
        window,
        window,
        bid_pair,
    )
    .expect("bid evidence contract");
    let ask = ExactQuoteSideEvidenceV1::new(
        QuoteSideV1::Ask,
        42,
        "EURUSD",
        1,
        2,
        window,
        window,
        ask_pair,
    )
    .expect("ask evidence contract");
    let primary_quotes = SynchronizedBidAskEvidenceV1::new(bid, ask, synchronization_pair)
        .expect("synchronized bid/ask contract");

    let conversion_route = ExactConversionRouteEvidenceV1::new(
        "primary_pnl_settlement",
        2,
        "USD",
        2,
        "USD",
        Vec::new(),
    )
    .expect("explicit identity conversion route");

    let (symbol_contracts, pair_sources) = captured_pair(
        &source_root,
        "symbol-contracts",
        BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV1,
        BrokerFinancialTruthVortexSchemaV1::CTraderSymbolContractsDecodedV1,
    );
    sources.extend(pair_sources);
    let (position_unrealized_pnl, pair_sources) = captured_pair(
        &source_root,
        "position-unrealized-pnl",
        BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlResponsesRawV1,
        BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlDecodedV1,
    );
    sources.extend(pair_sources);
    let (close_deal_reconciliation, pair_sources) = captured_pair(
        &source_root,
        "close-deal-reconciliation",
        BrokerFinancialTruthVortexSchemaV1::CTraderDealResponsesRawV1,
        BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV1,
    );
    sources.extend(pair_sources);

    let manifest = BrokerFinancialTruthBundleManifestV1::new(
        binding.clone(),
        primary_quotes,
        vec![conversion_route],
        symbol_contracts,
        position_unrealized_pnl,
        close_deal_reconciliation,
    )
    .expect("complete structural evidence contract");
    let store = BrokerFinancialTruthBundleStoreV1::new(root.join("store"));

    Fixture {
        root,
        store,
        binding,
        manifest,
        sources,
    }
}

#[test]
fn exact_reopen_rejects_artifact_tampering() {
    let fixture = fixture("tamper");
    let receipt = fixture
        .store
        .publish(&fixture.manifest, &fixture.sources)
        .expect("publish immutable mechanics fixture");
    fixture
        .store
        .open_exact(&receipt, &fixture.binding)
        .expect("exact immutable reopen before tampering");

    let tampered = fixture
        .store
        .bundle_path(&receipt)
        .join("primary-bid-decoded.vortex");
    let mut tampered_bytes = std::fs::read(&tampered).expect("read published fixture");
    tampered_bytes[0] ^= 1;
    std::fs::write(tampered, tampered_bytes).expect("same-length tamper fixture");

    let error = fixture
        .store
        .open_exact(&receipt, &fixture.binding)
        .expect_err("changed evidence must fail closed");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthStoreErrorCodeV1::ArtifactDigestMismatch
    );
}

#[test]
fn exact_reopen_rejects_a_different_search_receipt_binding() {
    let fixture = fixture("binding");
    let receipt = fixture
        .store
        .publish(&fixture.manifest, &fixture.sources)
        .expect("publish immutable mechanics fixture");
    let identity = fixture.binding.canonical_dataset_identity().clone();
    let wrong_binding = BrokerFinancialTruthBindingV1::new(
        &identity,
        "22".repeat(32),
        fixture.binding.evaluated_window(),
        fixture.binding.primary_base_asset_id(),
        fixture.binding.primary_base_asset_name(),
        fixture.binding.primary_quote_asset_id(),
        fixture.binding.primary_quote_asset_name(),
        fixture.binding.account_asset_id(),
        fixture.binding.account_asset_name(),
    )
    .expect("well-formed but different binding");

    let error = fixture
        .store
        .open_exact(&receipt, &wrong_binding)
        .expect_err("a foreign search receipt must not reuse evidence");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthStoreErrorCodeV1::BindingMismatch
    );
}

#[test]
fn storing_synthetic_bytes_cannot_create_or_install_a_financial_truth_permit() {
    let fixture = fixture("no-permit");
    let _receipt = fixture
        .store
        .publish(&fixture.manifest, &fixture.sources)
        .expect("exercise immutable store mechanics only");

    let error = current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("chunk 1 has no evidence-to-capability bridge");
    assert_eq!(
        error.to_string().split_whitespace().next(),
        Some("BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1")
    );
}
