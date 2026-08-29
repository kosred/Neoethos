use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use neoethos_broker_truth::{
    BROKER_TRUTH_ACQUISITION_AUTHORITY_MANIFEST_FILE_V1,
    BROKER_TRUTH_ACQUISITION_LINK_MANIFEST_FILE_V1, BrokerFinancialOperationV1,
    BrokerFinancialTruthArtifactSourceV1, BrokerFinancialTruthBindingV1,
    BrokerFinancialTruthBundleManifestV2, BrokerFinancialTruthBundleReceiptV2,
    BrokerFinancialTruthBundleStoreV1, BrokerFinancialTruthVortexSchemaV1,
    BrokerTruthAcquisitionArtifactRoleV1, BrokerTruthAcquisitionArtifactSourceV1,
    BrokerTruthAcquisitionArtifactV1, BrokerTruthAcquisitionAuthorityManifestV1,
    BrokerTruthAcquisitionAuthorityReceiptV1, BrokerTruthAcquisitionLinkManifestV1,
    BrokerTruthAcquisitionLinkReceiptV1, BrokerTruthAcquisitionPromotionEligibilityV1,
    BrokerTruthAcquisitionSemanticStatusV1, BrokerTruthAcquisitionStoreErrorCodeV1,
    BrokerTruthAcquisitionStoreV1, BrokerTruthReviewedSynchronizationBindingV1, EvidenceWindowV1,
    ExactBrokerRequestChunkV2, ExactBrokerRequestPageV2, ExactCapturedEvidencePairV1,
    ExactConversionRouteEvidenceV2, ExactDealReconciliationEvidenceV2, ExactQuoteSideEvidenceV2,
    ExactSymbolContractEvidenceV2, ImmutableVortexArtifactV1, QuoteSideV1,
    ReviewedQuoteReplayRuleEvidenceV2, ReviewedQuoteReplayRuleIdentityV2,
    SynchronizedBidAskEvidenceV2, current_broker_financial_truth_capability_v1,
};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use sha2::{Digest, Sha256};

const ACCOUNT_ID: i64 = 7;
const SYMBOL_ID: i64 = 42;
const WINDOW_FROM_MS: i64 = 1_700_000_000_000;
const WINDOW_TO_MS: i64 = WINDOW_FROM_MS + 60_000;

struct AcquisitionFixture {
    root: PathBuf,
    store: BrokerTruthAcquisitionStoreV1,
    manifest: BrokerTruthAcquisitionAuthorityManifestV1,
    sources: Vec<BrokerTruthAcquisitionArtifactSourceV1>,
}

impl Drop for AcquisitionFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn unique_temp_root(test_name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "neoethos-broker-truth-acquisition-{test_name}-{}-{nonce}",
        std::process::id()
    ))
}

fn binding() -> BrokerFinancialTruthBindingV1 {
    let identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        ACCOUNT_ID,
        SYMBOL_ID,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("canonical cTrader identity");
    BrokerFinancialTruthBindingV1::new(
        &identity,
        digest(0x51),
        EvidenceWindowV1::new(WINDOW_FROM_MS, WINDOW_TO_MS).expect("evidence window"),
        1,
        "EUR",
        2,
        "USD",
        2,
        "USD",
    )
    .expect("exact broker financial binding")
}

fn write_authority_artifact(
    source_root: &Path,
    role: BrokerTruthAcquisitionArtifactRoleV1,
    relative_path: &str,
    bytes: &[u8],
) -> (
    BrokerTruthAcquisitionArtifactV1,
    BrokerTruthAcquisitionArtifactSourceV1,
) {
    let source_path = source_root.join(relative_path);
    fs::write(&source_path, bytes).expect("write acquisition source");
    let artifact = BrokerTruthAcquisitionArtifactV1::new(
        role,
        relative_path,
        sha256(bytes),
        bytes.len() as u64,
    )
    .expect("immutable authority artifact");
    let source = BrokerTruthAcquisitionArtifactSourceV1::new(relative_path, source_path)
        .expect("exact acquisition source mapping");
    (artifact, source)
}

fn acquisition_fixture(test_name: &str) -> AcquisitionFixture {
    let root = unique_temp_root(test_name);
    let source_root = root.join("authority-sources");
    fs::create_dir_all(&source_root).expect("create acquisition source root");

    let specs = [
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchInputReceipt,
            "canonical-search-input-receipt.json",
            b"canonical receipt bytes".as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchArtifactScope,
            "canonical-search-artifact-scope.json",
            b"canonical scope bytes".as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalRootVerificationReceipt,
            "canonical-root-verification.json",
            b"exact generation root verification bytes".as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalScopeWindowBinding,
            "canonical-scope-window-binding.json",
            b"exact scope to half-open window bytes".as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CapturePlan,
            "broker-truth-capture-plan.json",
            b"exact non-secret capture plan bytes".as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::ReviewRecord,
            "quote-replay-review-record.json",
            b"immutable external review bytes".as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::ProtocolEvidence,
            "ctrader-protocol-evidence.bin",
            b"immutable reviewed protocol bytes".as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::TrustRoot,
            "quote-review-trust-root.pub",
            b"unverified trust root bytes".as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::QuoteSessionObservations { ordinal: 0 },
            "quote-session-observations-000.vortex",
            b"retained broker observation bytes".as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::ReviewedQuoteReplayRules { ordinal: 0 },
            "reviewed-quote-replay-rules-000.vortex",
            b"retained reviewed replay rule bytes".as_slice(),
        ),
    ];
    let mut artifacts = Vec::with_capacity(specs.len());
    let mut sources = Vec::with_capacity(specs.len());
    for (role, relative_path, bytes) in specs {
        let (artifact, source) = write_authority_artifact(&source_root, role, relative_path, bytes);
        artifacts.push(artifact);
        sources.push(source);
    }

    let review_identity = ReviewedQuoteReplayRuleIdentityV2::new(
        artifacts[5].sha256(),
        artifacts[6].sha256(),
        artifacts[8].sha256(),
    )
    .expect("reviewed replay identity");
    let synchronization = BrokerTruthReviewedSynchronizationBindingV1::new(
        0,
        ACCOUNT_ID,
        SYMBOL_ID,
        EvidenceWindowV1::new(WINDOW_FROM_MS, WINDOW_TO_MS).expect("evidence window"),
        review_identity,
        artifacts[9].sha256(),
    )
    .expect("reviewed synchronization");
    let canonical_root_verification_sha256 = artifacts[2].sha256().to_owned();
    let canonical_scope_window_binding_sha256 = artifacts[3].sha256().to_owned();
    let capture_plan_sha256 = artifacts[4].sha256().to_owned();
    let expected_trust_root_sha256 = artifacts[7].sha256().to_owned();
    let manifest = BrokerTruthAcquisitionAuthorityManifestV1::new(
        digest(0x51),
        digest(0x52),
        canonical_root_verification_sha256,
        canonical_scope_window_binding_sha256,
        capture_plan_sha256,
        expected_trust_root_sha256,
        artifacts,
        vec![synchronization],
    )
    .expect("complete acquisition authority");

    AcquisitionFixture {
        store: BrokerTruthAcquisitionStoreV1::new(root.join("store")),
        root,
        manifest,
        sources,
    }
}

fn write_vortex_descriptor(
    source_root: &Path,
    sources: &mut Vec<BrokerFinancialTruthArtifactSourceV1>,
    relative_path: &str,
    schema: BrokerFinancialTruthVortexSchemaV1,
) -> ImmutableVortexArtifactV1 {
    let source_path = source_root.join(relative_path);
    fs::write(
        &source_path,
        format!("opaque integrity-only V2 store fixture: {relative_path}"),
    )
    .expect("write opaque broker artifact");
    let artifact = ImmutableVortexArtifactV1::from_file(relative_path, schema, 1, &source_path)
        .expect("describe opaque broker artifact");
    sources.push(
        BrokerFinancialTruthArtifactSourceV1::new(relative_path, source_path)
            .expect("broker artifact source"),
    );
    artifact
}

fn quote_side(
    source_root: &Path,
    sources: &mut Vec<BrokerFinancialTruthArtifactSourceV1>,
    side: QuoteSideV1,
) -> ExactQuoteSideEvidenceV2 {
    let label = match side {
        QuoteSideV1::Bid => "bid",
        QuoteSideV1::Ask => "ask",
    };
    let window = EvidenceWindowV1::new(WINDOW_FROM_MS, WINDOW_TO_MS).expect("quote window");
    let page = ExactBrokerRequestPageV2::new(
        0,
        0,
        format!("primary-{label}-page"),
        window,
        Some(WINDOW_FROM_MS + 30_000),
        Some(WINDOW_FROM_MS + 30_000),
        1,
        false,
        None,
    )
    .expect("terminal quote page");
    let chunk = ExactBrokerRequestChunkV2::new(0, window, vec![page]).expect("quote chunk");
    let raw = write_vortex_descriptor(
        source_root,
        sources,
        &format!("primary-{label}-pages-raw.vortex"),
        BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2,
    );
    let decoded = write_vortex_descriptor(
        source_root,
        sources,
        &format!("primary-{label}-ticks-decoded.vortex"),
        BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV2,
    );
    ExactQuoteSideEvidenceV2::new(
        side,
        SYMBOL_ID,
        "EURUSD",
        1,
        2,
        window,
        vec![chunk],
        raw,
        decoded,
    )
    .expect("exact quote side")
}

fn publish_integrity_only_bft2(
    root: &Path,
) -> (
    BrokerFinancialTruthBundleReceiptV2,
    BrokerFinancialTruthBindingV1,
) {
    let source_root = root.join("bft2-sources");
    fs::create_dir_all(&source_root).expect("create BFT2 source root");
    let mut sources = Vec::new();
    let window = EvidenceWindowV1::new(WINDOW_FROM_MS, WINDOW_TO_MS).expect("bundle window");
    let exact_binding = binding();
    let bid = quote_side(&source_root, &mut sources, QuoteSideV1::Bid);
    let ask = quote_side(&source_root, &mut sources, QuoteSideV1::Ask);
    let observations = write_vortex_descriptor(
        &source_root,
        &mut sources,
        "quote-session-observations-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV2,
    );
    let rules = write_vortex_descriptor(
        &source_root,
        &mut sources,
        "reviewed-quote-replay-rules-decoded.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderReviewedQuoteReplayRulesDecodedV2,
    );
    let review_identity =
        ReviewedQuoteReplayRuleIdentityV2::new(digest(0xa1), digest(0xa2), observations.sha256())
            .expect("integrity-only review identity");
    let replay = ReviewedQuoteReplayRuleEvidenceV2::new(review_identity, observations, rules)
        .expect("reviewed replay evidence shape");
    let primary =
        SynchronizedBidAskEvidenceV2::new(bid, ask, replay).expect("primary quote evidence");

    let symbol_contracts = ExactSymbolContractEvidenceV2::new(
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "light-symbol-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderLightSymbolResponsesRawV2,
        ),
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "full-symbol-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV2,
        ),
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "account-asset-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderAccountAssetResponsesRawV2,
        ),
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "trader-account-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderTraderAccountResponsesRawV2,
        ),
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "symbol-money-contracts-decoded.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolMoneyContractsDecodedV2,
        ),
    )
    .expect("symbol contracts");
    let pnl = ExactCapturedEvidencePairV1::new(
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "position-unrealized-pnl-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlResponsesRawV2,
        ),
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "position-unrealized-pnl-decoded.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlDecodedV2,
        ),
    );
    let deal_page =
        ExactBrokerRequestPageV2::new(0, 0, "deal-page", window, None, None, 0, false, Some(100))
            .expect("terminal empty DealList page");
    let deal_chunk =
        ExactBrokerRequestChunkV2::new(0, window, vec![deal_page]).expect("DealList chunk");
    let close_deal = ExactDealReconciliationEvidenceV2::new(
        window,
        deal_chunk,
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "reconcile-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderReconcileResponsesRawV2,
        ),
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "deal-pages-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderDealPagesRawV2,
        ),
        write_vortex_descriptor(
            &source_root,
            &mut sources,
            "close-deal-reconciliation-decoded.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV2,
        ),
    )
    .expect("close/deal evidence");
    let settlement = ExactConversionRouteEvidenceV2::new(
        "primary_pnl_settlement",
        2,
        "USD",
        2,
        "USD",
        Vec::new(),
    )
    .expect("identity settlement route");
    let manifest = BrokerFinancialTruthBundleManifestV2::new(
        exact_binding.clone(),
        primary,
        vec![settlement],
        symbol_contracts,
        pnl,
        close_deal,
    )
    .expect("integrity-only BFT2 manifest");
    let receipt = BrokerFinancialTruthBundleStoreV1::new(root)
        .publish_v2(&manifest, &sources)
        .expect("publish integrity-only BFT2 bundle");
    (receipt, exact_binding)
}

fn symlink_file(source: &Path, destination: &Path) {
    #[cfg(unix)]
    std::os::unix::fs::symlink(source, destination).expect("create source symlink");
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(source, destination).expect("create source symlink");
}

#[test]
fn authority_publish_and_exact_reopen_are_content_addressed_and_evidence_only() {
    let fixture = acquisition_fixture("authority-green");
    let receipt = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect("publish immutable acquisition authority");

    assert!(receipt.authority_id().starts_with("bfta1-"));
    assert_eq!(
        BrokerTruthAcquisitionAuthorityReceiptV1::from_json_bytes(
            &receipt
                .canonical_json_bytes()
                .expect("canonical authority receipt")
        )
        .expect("strict authority receipt reopen"),
        receipt
    );
    assert_eq!(
        fixture.store.authority_path(&receipt),
        fixture.store.root().join(receipt.authority_id())
    );
    let reopened = fixture
        .store
        .open_authority(&receipt)
        .expect("exact authority reopen");
    assert_eq!(reopened.receipt(), &receipt);
    assert_eq!(reopened.manifest(), &fixture.manifest);
    for artifact in fixture.manifest.artifacts() {
        assert_eq!(
            reopened.artifact_path(artifact),
            reopened.root().join(artifact.relative_path())
        );
    }
    assert_eq!(
        fixture
            .store
            .publish_authority(&fixture.manifest, &fixture.sources)
            .expect("idempotent exact publication"),
        receipt
    );
    assert!(!fixture.store.root().join("current").exists());
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("an immutable acquisition receipt is not a financial permit");
}

#[test]
fn authority_source_set_modified_bytes_and_symlinks_are_refused() {
    let fixture = acquisition_fixture("missing-source");
    let error = fixture
        .store
        .publish_authority(
            &fixture.manifest,
            &fixture.sources[..fixture.sources.len() - 1],
        )
        .expect_err("partial source sets must fail");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch
    );

    let fixture = acquisition_fixture("modified-source");
    fs::write(fixture.sources[0].source_path(), b"changed after hashing")
        .expect("modify source bytes");
    let error = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect_err("changed source bytes must fail");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch
    );

    let error = BrokerTruthAcquisitionArtifactSourceV1::new(
        "../canonical-search-input-receipt.json",
        "ignored",
    )
    .expect_err("unsafe relative source names must fail");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::SourceMismatch
    );

    let fixture = acquisition_fixture("symlink-source");
    let source_path = fixture.sources[0].source_path().to_path_buf();
    let target_path = fixture.root.join("symlink-target.bin");
    fs::copy(&source_path, &target_path).expect("copy exact symlink target");
    fs::remove_file(&source_path).expect("remove original source");
    symlink_file(&target_path, &source_path);
    let error = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect_err("symlink sources must fail before copying");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::UnsafeFilesystemEntry
    );
}

#[test]
fn stored_truncation_digest_tamper_extra_file_and_conflict_are_refused() {
    let fixture = acquisition_fixture("stored-truncation");
    let receipt = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect("publish authority before truncation");
    fs::write(
        fixture
            .store
            .authority_path(&receipt)
            .join(fixture.manifest.artifacts()[0].relative_path()),
        b"short",
    )
    .expect("truncate stored artifact");
    let error = fixture
        .store
        .open_authority(&receipt)
        .expect_err("stored truncation must fail");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::ArtifactLengthMismatch
    );

    let fixture = acquisition_fixture("stored-digest");
    let receipt = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect("publish authority before digest tamper");
    let artifact = &fixture.manifest.artifacts()[0];
    fs::write(
        fixture
            .store
            .authority_path(&receipt)
            .join(artifact.relative_path()),
        vec![b'x'; artifact.byte_len() as usize],
    )
    .expect("same-length stored tamper");
    let error = fixture
        .store
        .open_authority(&receipt)
        .expect_err("stored digest tamper must fail");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::ArtifactDigestMismatch
    );

    let fixture = acquisition_fixture("stored-extra");
    let receipt = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect("publish authority before extra file");
    fs::write(
        fixture.store.authority_path(&receipt).join("unlisted.bin"),
        b"unlisted",
    )
    .expect("write extra stored file");
    let error = fixture
        .store
        .open_authority(&receipt)
        .expect_err("extra stored files must fail");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::ArtifactSetMismatch
    );

    let fixture = acquisition_fixture("content-address-conflict");
    let receipt = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect("obtain exact content address");
    let authority_root = fixture.store.authority_path(&receipt);
    fs::remove_dir_all(&authority_root).expect("remove exact temporary test publication");
    fs::create_dir_all(&authority_root).expect("create conflicting content-addressed directory");
    fs::write(authority_root.join("wrong.bin"), b"conflict").expect("write conflicting content");
    let error = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect_err("an occupied mismatched content address must fail");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::ManifestMissing
    );
}

#[test]
fn exact_authority_and_bft2_publish_one_immutable_evidence_only_link() {
    let fixture = acquisition_fixture("link-green");
    let authority_receipt = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect("publish authority before link");
    let (broker_receipt, exact_binding) = publish_integrity_only_bft2(fixture.store.root());

    let link_receipt = fixture
        .store
        .publish_link(&authority_receipt, &broker_receipt, &exact_binding)
        .expect("publish exact acquisition link");
    assert!(link_receipt.link_id().starts_with("bftl1-"));
    assert_eq!(
        BrokerTruthAcquisitionLinkReceiptV1::from_json_bytes(
            &link_receipt
                .canonical_json_bytes()
                .expect("canonical link receipt")
        )
        .expect("strict link receipt reopen"),
        link_receipt
    );
    let reopened = fixture
        .store
        .open_link(&link_receipt)
        .expect("strict acquisition link reopen");
    assert_eq!(reopened.receipt(), &link_receipt);
    assert_eq!(
        reopened.manifest().semantic_status(),
        BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly
    );
    assert_eq!(
        reopened.manifest().promotion_eligibility(),
        BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible
    );
    assert_eq!(reopened.manifest().authority_receipt(), &authority_receipt);
    assert_eq!(reopened.manifest().broker_truth_receipt(), &broker_receipt);
    assert_eq!(reopened.manifest().binding(), &exact_binding);
    assert_eq!(
        reopened
            .root()
            .join(BROKER_TRUTH_ACQUISITION_LINK_MANIFEST_FILE_V1),
        reopened.manifest_path()
    );
    assert!(!fixture.store.root().join("current").exists());
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("an acquisition link remains unvalidated evidence, not a permit");
}

#[test]
fn links_to_missing_authority_or_bft2_and_binding_mismatch_fail_closed() {
    let fixture = acquisition_fixture("link-refusals");
    let authority_receipt = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect("publish authority before missing-BFT2 check");
    let missing_bft2 = BrokerFinancialTruthBundleReceiptV2::from_json_bytes(
        format!(
            r#"{{"bundle_id":"bft2-{}","manifest_sha256":"{}"}}"#,
            digest(0x61),
            digest(0x61)
        )
        .as_bytes(),
    )
    .expect("syntactically exact missing BFT2 receipt");
    let error = fixture
        .store
        .publish_link(&authority_receipt, &missing_bft2, &binding())
        .expect_err("missing BFT2 target must fail");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::ReferencedBrokerBundleInvalid
    );

    let missing_authority_sha = digest(0x62);
    let missing_authority = BrokerTruthAcquisitionAuthorityReceiptV1::from_json_bytes(
        format!(
            r#"{{"authority_id":"bfta1-{missing_authority_sha}","manifest_sha256":"{missing_authority_sha}"}}"#
        )
        .as_bytes(),
    )
    .expect("syntactically exact missing authority receipt");
    let error = fixture
        .store
        .publish_link(&missing_authority, &missing_bft2, &binding())
        .expect_err("missing authority target must fail first");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::ReferencedAuthorityInvalid
    );

    let (broker_receipt, exact_binding) = publish_integrity_only_bft2(fixture.store.root());
    let different_identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        ACCOUNT_ID,
        SYMBOL_ID,
        "EURUSD",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("different exact identity");
    let mismatched_binding = BrokerFinancialTruthBindingV1::new(
        &different_identity,
        digest(0x51),
        EvidenceWindowV1::new(WINDOW_FROM_MS, WINDOW_TO_MS).expect("same window"),
        1,
        "EUR",
        2,
        "USD",
        2,
        "USD",
    )
    .expect("individually valid mismatched binding");
    let error = fixture
        .store
        .publish_link(&authority_receipt, &broker_receipt, &mismatched_binding)
        .expect_err("a link binding different from the exact BFT2 target must fail");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionStoreErrorCodeV1::ReferencedBrokerBundleInvalid
    );
    assert_eq!(exact_binding, binding());
}

#[test]
fn link_manifest_is_strict_and_cannot_claim_validated_or_promotion_status() {
    let fixture = acquisition_fixture("link-contract");
    let authority_receipt = fixture
        .store
        .publish_authority(&fixture.manifest, &fixture.sources)
        .expect("publish authority for link contract");
    let bft2_sha = digest(0x63);
    let broker_receipt = BrokerFinancialTruthBundleReceiptV2::from_json_bytes(
        format!(r#"{{"bundle_id":"bft2-{bft2_sha}","manifest_sha256":"{bft2_sha}"}}"#).as_bytes(),
    )
    .expect("syntactically exact BFT2 receipt");
    let link =
        BrokerTruthAcquisitionLinkManifestV1::new(authority_receipt, broker_receipt, binding())
            .expect("strict evidence-only link manifest");
    let canonical = link.canonical_json_bytes().expect("canonical link bytes");
    assert_eq!(
        link.identity_sha256().expect("link identity"),
        BrokerTruthAcquisitionLinkManifestV1::from_json_bytes(&canonical)
            .expect("link identity reopen")
            .identity_sha256()
            .expect("reopened link identity")
    );
    assert_eq!(
        BrokerTruthAcquisitionLinkManifestV1::from_json_bytes(&canonical)
            .expect("strict link reopen"),
        link
    );

    let mut value: serde_json::Value = serde_json::from_slice(&canonical).expect("link JSON value");
    value["semantic_status"] = serde_json::Value::String("validated".to_owned());
    BrokerTruthAcquisitionLinkManifestV1::from_json_bytes(
        &serde_json::to_vec(&value).expect("tampered link JSON"),
    )
    .expect_err("no validated acquisition-link status exists");

    let source = include_str!("../src/acquisition_store_v1.rs");
    for forbidden in [
        "BrokerFinancialTruthPermitV1",
        "BrokerFinancialTruthCapabilityV1",
        "install",
        "current.json",
        "default()",
        "std::env",
    ] {
        assert!(
            !source.contains(forbidden),
            "acquisition store contains forbidden authority/fallback token {forbidden}"
        );
    }
    assert!(source.contains(BROKER_TRUTH_ACQUISITION_AUTHORITY_MANIFEST_FILE_V1));
    assert!(source.contains(BROKER_TRUTH_ACQUISITION_LINK_MANIFEST_FILE_V1));
}
