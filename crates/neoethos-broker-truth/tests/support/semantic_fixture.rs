// Shared synthetic Vortex/review fixture for exact-ingress and acquisition-to-replay tests.
// This is test-only construction, never observed broker fills or production trust.
use neoethos_broker_truth::{
    BrokerFinancialTruthArtifactSourceV1, BrokerFinancialTruthBindingV1,
    BrokerFinancialTruthBundleManifestV2, BrokerFinancialTruthBundleStoreV1,
    BrokerFinancialTruthVortexSchemaV1, BrokerTruthAcquisitionArtifactRoleV1,
    BrokerTruthAcquisitionArtifactSourceV1, BrokerTruthAcquisitionArtifactV1,
    BrokerTruthAcquisitionAuthorityManifestV1, BrokerTruthAcquisitionLinkReceiptV1,
    BrokerTruthAcquisitionStoreV1, BrokerTruthReviewedSynchronizationBindingV1,
    CausalQuoteToAccountConversionV1, EvidenceWindowV1, ExactBrokerRequestChunkV2,
    ExactBrokerRequestPageV2, ExactCapturedEvidencePairV1, ExactConversionRouteEvidenceV2,
    ExactDealReconciliationEvidenceV2, ExactQuoteSideEvidenceV2, ExactSymbolContractEvidenceV2,
    ExecutionCommissionPolicyV1, ExecutionSymbolContractV1, ImmutableVortexArtifactV1,
    PnlConversionFeeV1, QuoteSideV1, QuoteValidatedExecutionEconomicsLedgerV1,
    ReviewedBrokerFinancialTruthEvidenceV2, ReviewedQuoteReplayRuleEvidenceV2,
    ReviewedQuoteReplayRuleIdentityV2, SealedHistoricalQuoteValidatedResearchLedgerV1,
    SignedSwapCashflowV1, SynchronizedBidAskEvidenceV2,
    build_quote_validated_execution_economics_v1,
};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use vortex_array::IntoArray;
use vortex_array::arrays::{ChunkedArray, PrimitiveArray, StructArray, VarBinArray};
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::{RuntimeSession, RuntimeSessionExt};
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

pub(crate) const ACCOUNT_ID: i64 = 7;
pub(crate) const SYMBOL_ID: i64 = 42;
pub(crate) const WINDOW_FROM: i64 = 1_700_000_000_000;
pub(crate) const WINDOW_TO: i64 = WINDOW_FROM + 60_000;
pub(crate) const TICK_TIMESTAMP: i64 = WINDOW_FROM + 30_000;
pub(crate) const CANONICAL_RUN_BYTES: &[u8] = b"exact canonical run receipt fixture v2";
pub(crate) const REVIEW_RECORD_BYTES: &[u8] = b"reviewed quote semantics fixture v2";
pub(crate) const PROTOCOL_EVIDENCE_BYTES: &[u8] = b"reviewed cTrader protocol fixture v2";
pub(crate) const TRUST_ROOT_BYTES: &[u8] = b"offline fixture trust root v2";

static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(1);
static VORTEX_RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);
static VORTEX_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let mut session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>()
        .with_handle(VORTEX_RUNTIME.handle());
    vortex_file::register_default_encodings(&mut session);
    session
});

#[derive(Clone, Copy)]
pub(crate) enum Tamper {
    None,
    CorruptVortex,
    ExtraDecodedTickField,
    WrongDeclaredRowCount,
    TickRawDecodedMismatch,
    TickFinalRowMismatch,
    TickPageOrdinalMismatch,
    InvalidRawEnvelope,
    GenericRawDecodedLinkMismatch,
    DealPageMismatch,
}

pub(crate) struct FixtureRoot(pub(crate) PathBuf);

impl Drop for FixtureRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(crate) struct AuthorityFixture {
    pub(crate) store: BrokerTruthAcquisitionStoreV1,
    pub(crate) link_receipt: BrokerTruthAcquisitionLinkReceiptV1,
    pub(crate) reviewed: ReviewedBrokerFinancialTruthEvidenceV2,
}

struct ArtifactBuilder {
    source_root: PathBuf,
    sources: Vec<BrokerFinancialTruthArtifactSourceV1>,
}

impl ArtifactBuilder {
    fn new(source_root: PathBuf) -> Self {
        Self {
            source_root,
            sources: Vec::new(),
        }
    }

    fn array(
        &mut self,
        relative_path: &str,
        schema: BrokerFinancialTruthVortexSchemaV1,
        array: vortex_array::ArrayRef,
        declared_rows: Option<u64>,
    ) -> ImmutableVortexArtifactV1 {
        let path = self.source_root.join(relative_path);
        write_vortex(&path, array.clone());
        let exact =
            ImmutableVortexArtifactV1::from_file(relative_path, schema, array.len() as u64, &path)
                .expect("inspect Vortex source");
        let artifact = if let Some(rows) = declared_rows {
            ImmutableVortexArtifactV1::new(
                relative_path,
                schema,
                exact.sha256(),
                exact.byte_len(),
                rows,
            )
            .expect("descriptor with deliberate semantic row-count mismatch")
        } else {
            exact
        };
        self.sources.push(
            BrokerFinancialTruthArtifactSourceV1::new(relative_path, path)
                .expect("artifact source"),
        );
        artifact
    }

    fn corrupt(
        &mut self,
        relative_path: &str,
        schema: BrokerFinancialTruthVortexSchemaV1,
    ) -> ImmutableVortexArtifactV1 {
        let path = self.source_root.join(relative_path);
        fs::write(&path, b"not-a-vortex-file").expect("write corrupt structural fixture");
        let artifact = ImmutableVortexArtifactV1::from_file(relative_path, schema, 1, &path)
            .expect("describe corrupt bytes without claiming Vortex semantics");
        self.sources.push(
            BrokerFinancialTruthArtifactSourceV1::new(relative_path, path)
                .expect("corrupt artifact source"),
        );
        artifact
    }
}

#[derive(Clone)]
struct EvidenceRow {
    sequence: u64,
    account_id: i64,
    symbol_id: Option<i64>,
    quote_side: Option<QuoteSideV1>,
    evidence_kind: u8,
    requested_window: Option<EvidenceWindowV1>,
    client_msg_id: String,
    payload_type: u32,
    payload_json: String,
}

pub(crate) fn fixture_execution_economics(
    quotes: &SealedHistoricalQuoteValidatedResearchLedgerV1,
) -> QuoteValidatedExecutionEconomicsLedgerV1 {
    let closed_at = quotes.positions()[0]
        .exit_reference()
        .expect("closed synthetic position")
        .timestamp_unix_ms();
    let contract = ExecutionSymbolContractV1::new("EURUSD", "EUR", "USD", 100_000.0)
        .expect("explicit fixture lot contract");
    let assumption_source = sha256(b"synthetic economics assumptions, not real broker authority");
    let conversion = CausalQuoteToAccountConversionV1::new(
        "USD",
        "USD",
        1.0,
        closed_at,
        1_000,
        &assumption_source,
    )
    .expect("explicit identity conversion fixture");
    let commission = ExecutionCommissionPolicyV1::new("USD", 7.0, &assumption_source)
        .expect("fixture commission per fill");
    let swap = SignedSwapCashflowV1::new("USD", 0.0, closed_at, &assumption_source)
        .expect("explicit no-swap fixture");
    let fee = PnlConversionFeeV1::new("USD", 0.0, &assumption_source)
        .expect("explicit no-conversion-fee fixture");
    build_quote_validated_execution_economics_v1(
        quotes,
        0,
        &contract,
        "USD",
        1.0,
        Some(&conversion),
        &commission,
        &swap,
        &fee,
    )
    .expect("production economics from sealed fills")
}

pub(crate) fn fixture(
    tamper: Tamper,
) -> (
    FixtureRoot,
    neoethos_broker_truth::VerifiedImmutableBrokerFinancialTruthBundleV2,
    AuthorityFixture,
) {
    fixture_with_ticks(
        tamper,
        &[(TICK_TIMESTAMP, 110_000)],
        &[(TICK_TIMESTAMP, 120_000)],
    )
}

pub(crate) fn fixture_with_ticks(
    tamper: Tamper,
    bid_ticks: &[(i64, i64)],
    ask_ticks: &[(i64, i64)],
) -> (
    FixtureRoot,
    neoethos_broker_truth::VerifiedImmutableBrokerFinancialTruthBundleV2,
    AuthorityFixture,
) {
    let scope_bytes = b"exact canonical holdout scope fixture v2";
    let receipt_identity = sha256(CANONICAL_RUN_BYTES);
    let scope_identity = sha256(scope_bytes);
    fixture_with_ticks_and_scope(
        tamper,
        bid_ticks,
        ask_ticks,
        FixtureScope {
            window: EvidenceWindowV1::new(WINDOW_FROM, WINDOW_TO).expect("fixture window"),
            canonical_run_identity_sha256: &receipt_identity,
            canonical_scope_identity_sha256: &scope_identity,
            canonical_run_bytes: CANONICAL_RUN_BYTES,
            canonical_scope_bytes: scope_bytes,
        },
    )
}

pub(crate) struct FixtureScope<'a> {
    pub(crate) window: EvidenceWindowV1,
    // Typed identity hashes are domain separated from raw artifact byte hashes.
    pub(crate) canonical_run_identity_sha256: &'a str,
    pub(crate) canonical_scope_identity_sha256: &'a str,
    pub(crate) canonical_run_bytes: &'a [u8],
    pub(crate) canonical_scope_bytes: &'a [u8],
}

pub(crate) fn fixture_with_ticks_and_scope(
    tamper: Tamper,
    bid_ticks: &[(i64, i64)],
    ask_ticks: &[(i64, i64)],
    scope: FixtureScope<'_>,
) -> (
    FixtureRoot,
    neoethos_broker_truth::VerifiedImmutableBrokerFinancialTruthBundleV2,
    AuthorityFixture,
) {
    let root = FixtureRoot(unique_root());
    let source_root = root.0.join("sources");
    fs::create_dir_all(&source_root).expect("create source directory");
    let mut artifacts = ArtifactBuilder::new(source_root.clone());
    let window = scope.window;
    let canonical_run_identity_sha256 = scope.canonical_run_identity_sha256.to_owned();
    let binding = binding(window, &canonical_run_identity_sha256);

    let bid = quote_side(&mut artifacts, QuoteSideV1::Bid, tamper, window, bid_ticks);
    let ask = quote_side(
        &mut artifacts,
        QuoteSideV1::Ask,
        Tamper::None,
        window,
        ask_ticks,
    );

    let sync_raw_rows = vec![
        evidence_row(
            0,
            Some(SYMBOL_ID),
            Some(QuoteSideV1::Bid),
            0,
            Some(window),
            "sync-bid",
            2146,
            json!({"reviewedRawObservation":"bid"}),
        ),
        evidence_row(
            1,
            Some(SYMBOL_ID),
            Some(QuoteSideV1::Ask),
            0,
            Some(window),
            "sync-ask",
            2146,
            json!({"reviewedRawObservation":"ask"}),
        ),
    ];
    let sync_decoded_rows = vec![evidence_row(
        0,
        Some(SYMBOL_ID),
        None,
        1,
        Some(window),
        "sync-bid",
        2146,
        json!({"reviewedReplayRule":"exact-v2"}),
    )];
    let observations_raw = artifacts.array(
        "primary-quote-session-observations-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV2,
        evidence_array(&sync_raw_rows),
        None,
    );
    let rules_decoded = artifacts.array(
        "primary-reviewed-quote-replay-rules-decoded.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderReviewedQuoteReplayRulesDecodedV2,
        evidence_array(&sync_decoded_rows),
        None,
    );
    let review_identity = ReviewedQuoteReplayRuleIdentityV2::new(
        sha256(REVIEW_RECORD_BYTES),
        sha256(PROTOCOL_EVIDENCE_BYTES),
        observations_raw.sha256(),
    )
    .expect("exact fixture review identity");
    let authority_review_identity = review_identity.clone();
    let authority_observations_sha256 = observations_raw.sha256().to_owned();
    let authority_observations_byte_len = observations_raw.byte_len();
    let authority_rules_sha256 = rules_decoded.sha256().to_owned();
    let authority_rules_byte_len = rules_decoded.byte_len();
    let replay_rule =
        ReviewedQuoteReplayRuleEvidenceV2::new(review_identity, observations_raw, rules_decoded)
            .expect("review artifact contract");
    let primary_quotes =
        SynchronizedBidAskEvidenceV2::new(bid, ask, replay_rule).expect("primary quotes");

    let light_raw_envelope = json!({
        "clientMsgId":"light",
        "payloadType":2115,
        "payload":{"ctidTraderAccountId":ACCOUNT_ID,"symbol":[{
            "symbolId":SYMBOL_ID,"symbolName":"EURUSD","baseAssetId":1,"quoteAssetId":2
        }]}
    });
    let full_symbol = json!({
        "symbolId":SYMBOL_ID,"digits":5,"pipPosition":4,"lotSize":10_000_000,
        "minVolume":1000,"maxVolume":100_000_000,"stepVolume":1000
    });
    let full_raw_envelope = json!({
        "clientMsgId":"full",
        "payloadType":2117,
        "payload":{"ctidTraderAccountId":ACCOUNT_ID,"symbol":[full_symbol.clone()]}
    });
    let raw_asset = json!({"assetId":2,"name":"USD","digits":2});
    let asset_raw_envelope = json!({
        "clientMsgId":"assets",
        "payloadType":2113,
        "payload":{"ctidTraderAccountId":ACCOUNT_ID,"asset":[raw_asset.clone()]}
    });
    let raw_trader = json!({"depositAssetId":2,"moneyDigits":2,"balance":100_000});
    let trader_raw_envelope = json!({
        "clientMsgId":"trader",
        "payloadType":2122,
        "payload":{"ctidTraderAccountId":ACCOUNT_ID,"trader":raw_trader.clone()}
    });
    let light_raw = artifacts.array(
        "light-symbol-responses-v2-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderLightSymbolResponsesRawV2,
        evidence_array(&[evidence_row(
            0,
            None,
            None,
            2,
            None,
            "light",
            2115,
            light_raw_envelope.clone(),
        )]),
        None,
    );
    let full_raw = artifacts.array(
        "full-symbol-responses-v2-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV2,
        evidence_array(&[evidence_row(
            0,
            Some(SYMBOL_ID),
            None,
            4,
            None,
            "full",
            2117,
            full_raw_envelope.clone(),
        )]),
        None,
    );
    let asset_raw = artifacts.array(
        "account-asset-responses-v2-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderAccountAssetResponsesRawV2,
        evidence_array(&[evidence_row(
            0,
            None,
            None,
            6,
            None,
            "assets",
            2113,
            asset_raw_envelope.clone(),
        )]),
        None,
    );
    let trader_raw = artifacts.array(
        "trader-account-responses-v2-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderTraderAccountResponsesRawV2,
        evidence_array(&[evidence_row(
            0,
            None,
            None,
            8,
            None,
            "trader",
            2122,
            trader_raw_envelope.clone(),
        )]),
        None,
    );
    let decoded_contract_rows = vec![
        evidence_row(
            0,
            Some(SYMBOL_ID),
            None,
            3,
            None,
            "light",
            2115,
            json!({
                "authority":"ProtoOALightSymbol",
                "exactInstrument":{
                    "symbolId":SYMBOL_ID,"symbolName":"EURUSD",
                    "baseAssetId":1,"baseAssetName":"EUR",
                    "quoteAssetId":2,"quoteAssetName":"USD"
                },
                "rawLightSymbol":light_raw_envelope["payload"]["symbol"][0].clone()
            }),
        ),
        evidence_row(
            1,
            Some(SYMBOL_ID),
            None,
            5,
            None,
            "full",
            2117,
            json!({"authority":"ProtoOASymbol","rawSymbol":full_symbol}),
        ),
        evidence_row(
            2,
            None,
            None,
            7,
            None,
            "assets",
            2113,
            json!({"accountAssetId":2,"accountAssetName":"USD","requiredRawAssets":[raw_asset]}),
        ),
        evidence_row(
            3,
            None,
            None,
            9,
            None,
            "trader",
            2122,
            json!({"accountAssetId":2,"rawTrader":raw_trader}),
        ),
    ];
    let contracts_decoded = artifacts.array(
        "symbol-money-contracts-v2-decoded.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderSymbolMoneyContractsDecodedV2,
        evidence_array(&decoded_contract_rows),
        None,
    );
    let symbol_contracts = ExactSymbolContractEvidenceV2::new(
        light_raw,
        full_raw,
        asset_raw,
        trader_raw,
        contracts_decoded,
    )
    .expect("symbol contract artifacts");

    let pnl_raw_envelope = json!({
        "clientMsgId":"pnl","payloadType":2188,
        "payload":{"ctidTraderAccountId":ACCOUNT_ID,"moneyDigits":2,
            "positionUnrealizedPnL":[{
                "positionId":9,"grossUnrealizedPnL":123,"netUnrealizedPnL":100
            }]}
    });
    let pnl_raw = artifacts.array(
        "position-unrealized-pnl-v2-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlResponsesRawV2,
        evidence_array(&[evidence_row(
            0,
            None,
            None,
            10,
            None,
            "pnl",
            2188,
            pnl_raw_envelope,
        )]),
        None,
    );
    let pnl_decoded_client = if matches!(tamper, Tamper::GenericRawDecodedLinkMismatch) {
        "missing-raw-client"
    } else {
        "pnl"
    };
    let pnl_decoded = artifacts.array(
        "position-unrealized-pnl-v2-decoded.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlDecodedV2,
        evidence_array(&[evidence_row(
            0,
            None,
            None,
            11,
            None,
            pnl_decoded_client,
            2188,
            json!({
                "accountId":ACCOUNT_ID,"moneyDigits":2,
                "positions":[{"positionId":9,"grossUnrealizedPnL":1.23,"netUnrealizedPnL":1.0}]
            }),
        )]),
        None,
    );
    let pnl_pair = ExactCapturedEvidencePairV1::new(pnl_raw, pnl_decoded);

    let reconcile_payload = json!({
        "ctidTraderAccountId":ACCOUNT_ID,"position":[],"order":[]
    });
    let reconcile_envelope = json!({
        "clientMsgId":"reconcile","payloadType":2125,"payload":reconcile_payload.clone()
    });
    let reconcile_raw = artifacts.array(
        "reconcile-responses-v2-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderReconcileResponsesRawV2,
        evidence_array(&[evidence_row(
            0,
            None,
            None,
            12,
            Some(window),
            "reconcile",
            2125,
            reconcile_envelope,
        )]),
        None,
    );
    let deal_payload = json!({"ctidTraderAccountId":ACCOUNT_ID,"hasMore":false});
    let deal_envelope = json!({
        "clientMsgId":"deal-page","payloadType":2134,"payload":deal_payload.clone()
    });
    let deal_pages_raw = artifacts.array(
        "deal-pages-v2-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderDealPagesRawV2,
        raw_deal_page_array(
            window,
            &deal_envelope.to_string(),
            matches!(tamper, Tamper::DealPageMismatch),
        ),
        None,
    );
    let reconciliation_decoded = artifacts.array(
        "close-deal-reconciliation-v2-decoded.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV2,
        evidence_array(&[evidence_row(
            0,
            None,
            None,
            13,
            Some(window),
            "deal-page",
            2134,
            json!({
                "dealPageRequest":{"fromTimestamp":window.from_unix_ms_inclusive(),"toTimestamp":window.to_unix_ms_exclusive(),"maxRows":100},
                "rawDealPayload":deal_payload,"rawReconcilePayload":reconcile_payload,
                "returnProtectionOrders":true
            }),
        )]),
        None,
    );
    let deal_page =
        ExactBrokerRequestPageV2::new(0, 0, "deal-page", window, None, None, 0, false, Some(100))
            .expect("terminal empty deal page");
    let deal_chunk =
        ExactBrokerRequestChunkV2::new(0, window, vec![deal_page]).expect("deal chunk");
    let close_deal = ExactDealReconciliationEvidenceV2::new(
        window,
        deal_chunk,
        reconcile_raw,
        deal_pages_raw,
        reconciliation_decoded,
    )
    .expect("deal evidence");

    let settlement = ExactConversionRouteEvidenceV2::new(
        "primary_pnl_settlement",
        2,
        "USD",
        2,
        "USD",
        Vec::new(),
    )
    .expect("explicit identity conversion");
    let manifest = BrokerFinancialTruthBundleManifestV2::new(
        binding.clone(),
        primary_quotes,
        vec![settlement],
        symbol_contracts,
        pnl_pair,
        close_deal,
    )
    .expect("complete V2 structural manifest");
    let store = BrokerFinancialTruthBundleStoreV1::new(root.0.join("store"));
    let receipt = store
        .publish_v2(&manifest, &artifacts.sources)
        .expect("publish exact structural fixture");
    let verified = store
        .open_exact_v2(&receipt, &binding)
        .expect("integrity reopen before semantic ingress");
    let authority_source_root = root.0.join("authority-sources");
    fs::create_dir_all(&authority_source_root).expect("create authority source directory");
    let scope_bytes = scope.canonical_scope_bytes;
    let root_verification_bytes = b"exact canonical root verification fixture v2";
    let window_binding_bytes = b"exact canonical scope-window binding fixture v2";
    let capture_plan_bytes = b"exact BFT2 capture plan fixture v2";
    let base_specs = [
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchInputReceipt,
            "canonical-search-input-receipt.json",
            scope.canonical_run_bytes,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchArtifactScope,
            "canonical-search-artifact-scope.json",
            scope_bytes,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalRootVerificationReceipt,
            "canonical-root-verification.json",
            root_verification_bytes.as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalScopeWindowBinding,
            "canonical-scope-window-binding.json",
            window_binding_bytes.as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CapturePlan,
            "broker-truth-capture-plan.json",
            capture_plan_bytes.as_slice(),
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::ReviewRecord,
            "quote-replay-review-record.json",
            REVIEW_RECORD_BYTES,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::ProtocolEvidence,
            "ctrader-protocol-evidence.json",
            PROTOCOL_EVIDENCE_BYTES,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::TrustRoot,
            "quote-review-trust-root.pub",
            TRUST_ROOT_BYTES,
        ),
    ];
    let mut authority_artifacts = Vec::new();
    let mut authority_sources = Vec::new();
    for (role, relative_path, bytes) in base_specs {
        let (artifact, source) =
            authority_artifact_from_bytes(&authority_source_root, role, relative_path, bytes);
        authority_artifacts.push(artifact);
        authority_sources.push(source);
    }
    let (observations_artifact, observations_source) = authority_artifact_from_existing(
        BrokerTruthAcquisitionArtifactRoleV1::QuoteSessionObservations { ordinal: 0 },
        "quote-session-observations-000.vortex",
        &source_root.join("primary-quote-session-observations-raw.vortex"),
        &authority_observations_sha256,
        authority_observations_byte_len,
    );
    authority_artifacts.push(observations_artifact);
    authority_sources.push(observations_source);
    let (rules_artifact, rules_source) = authority_artifact_from_existing(
        BrokerTruthAcquisitionArtifactRoleV1::ReviewedQuoteReplayRules { ordinal: 0 },
        "reviewed-quote-replay-rules-000.vortex",
        &source_root.join("primary-reviewed-quote-replay-rules-decoded.vortex"),
        &authority_rules_sha256,
        authority_rules_byte_len,
    );
    authority_artifacts.push(rules_artifact);
    authority_sources.push(rules_source);

    let synchronization = BrokerTruthReviewedSynchronizationBindingV1::new(
        0,
        ACCOUNT_ID,
        SYMBOL_ID,
        window,
        authority_review_identity,
        authority_rules_sha256,
    )
    .expect("exact reviewed synchronization fixture");
    let canonical_scope_identity_sha256 = scope.canonical_scope_identity_sha256.to_owned();
    let canonical_root_verification_sha256 = sha256(root_verification_bytes);
    let canonical_scope_window_binding_sha256 = sha256(window_binding_bytes);
    let capture_plan_sha256 = sha256(capture_plan_bytes);
    let review_record_sha256 = sha256(REVIEW_RECORD_BYTES);
    let protocol_evidence_sha256 = sha256(PROTOCOL_EVIDENCE_BYTES);
    let trust_root_sha256 = sha256(TRUST_ROOT_BYTES);
    let authority_manifest = BrokerTruthAcquisitionAuthorityManifestV1::new(
        canonical_run_identity_sha256.clone(),
        canonical_scope_identity_sha256.clone(),
        canonical_root_verification_sha256.clone(),
        canonical_scope_window_binding_sha256.clone(),
        capture_plan_sha256.clone(),
        trust_root_sha256.clone(),
        authority_artifacts,
        vec![synchronization.clone()],
    )
    .expect("complete reviewed acquisition fixture");
    let acquisition_store = BrokerTruthAcquisitionStoreV1::new(root.0.join("store"));
    let authority_receipt = acquisition_store
        .publish_authority(&authority_manifest, &authority_sources)
        .expect("publish reviewed acquisition fixture");
    let link_receipt = acquisition_store
        .publish_link(&authority_receipt, &receipt, &binding)
        .expect("publish reviewed BFT2 link fixture");
    let reviewed = ReviewedBrokerFinancialTruthEvidenceV2::checked_new(
        canonical_run_identity_sha256,
        canonical_scope_identity_sha256,
        canonical_root_verification_sha256,
        canonical_scope_window_binding_sha256,
        capture_plan_sha256,
        trust_root_sha256,
        review_record_sha256,
        protocol_evidence_sha256,
        receipt.manifest_sha256(),
        window,
        vec![synchronization],
    )
    .expect("checked independent review fixture");
    (
        root,
        verified,
        AuthorityFixture {
            store: acquisition_store,
            link_receipt,
            reviewed,
        },
    )
}

pub(crate) fn binding(
    window: EvidenceWindowV1,
    canonical_run_identity_sha256: &str,
) -> BrokerFinancialTruthBindingV1 {
    let identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        ACCOUNT_ID,
        SYMBOL_ID,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("canonical identity");
    BrokerFinancialTruthBindingV1::new(
        &identity,
        canonical_run_identity_sha256,
        window,
        1,
        "EUR",
        2,
        "USD",
        2,
        "USD",
    )
    .expect("binding")
}

fn authority_artifact_from_bytes(
    root: &Path,
    role: BrokerTruthAcquisitionArtifactRoleV1,
    relative_path: &str,
    bytes: &[u8],
) -> (
    BrokerTruthAcquisitionArtifactV1,
    BrokerTruthAcquisitionArtifactSourceV1,
) {
    let path = root.join(relative_path);
    fs::write(&path, bytes).expect("write authority input fixture");
    authority_artifact_from_existing(
        role,
        relative_path,
        &path,
        &sha256(bytes),
        bytes.len() as u64,
    )
}

fn authority_artifact_from_existing(
    role: BrokerTruthAcquisitionArtifactRoleV1,
    relative_path: &str,
    source_path: &Path,
    digest: &str,
    byte_len: u64,
) -> (
    BrokerTruthAcquisitionArtifactV1,
    BrokerTruthAcquisitionArtifactSourceV1,
) {
    let artifact = BrokerTruthAcquisitionArtifactV1::new(role, relative_path, digest, byte_len)
        .expect("authority artifact fixture");
    let source = BrokerTruthAcquisitionArtifactSourceV1::new(relative_path, source_path)
        .expect("authority source fixture");
    (artifact, source)
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn quote_side(
    artifacts: &mut ArtifactBuilder,
    side: QuoteSideV1,
    tamper: Tamper,
    window: EvidenceWindowV1,
    ticks: &[(i64, i64)],
) -> ExactQuoteSideEvidenceV2 {
    // Synthetic pagination exercises newest-first raw pages and oldest-first
    // decoded storage across the production reader's 16,384-row batch boundary.
    const PAGE_ROWS: usize = 8_192;
    let prefix = match side {
        QuoteSideV1::Bid => "bid",
        QuoteSideV1::Ask => "ask",
    };
    let mut pages = Vec::new();
    let mut raw_arrays = Vec::new();
    let mut decoded_arrays = Vec::new();
    let mut page_to = window.to_unix_ms_exclusive();
    for (sequence, page_ticks) in ticks.rchunks(PAGE_ROWS).enumerate() {
        let sequence = sequence as u64;
        let has_more = (sequence as usize + 1) * PAGE_ROWS < ticks.len();
        let client_id = if sequence == 0 {
            format!("{prefix}-page")
        } else {
            format!("{prefix}-page-{sequence}")
        };
        let page = ExactBrokerRequestPageV2::new(
            0,
            sequence,
            &client_id,
            EvidenceWindowV1::new(window.from_unix_ms_inclusive(), page_to).unwrap(),
            Some(page_ticks.first().unwrap().0),
            Some(page_ticks.last().unwrap().0),
            page_ticks.len() as u64,
            has_more,
            None,
        )
        .expect("exact synthetic page");
        let wire = if matches!(tamper, Tamper::InvalidRawEnvelope) {
            r#"{"payloadType":2146,"payload":{}}"#.to_owned()
        } else {
            tick_response(&client_id, page_ticks, has_more)
        };
        raw_arrays.push(raw_tick_page_array(window, &page, side, &wire, page_ticks));
        let mut decoded_ticks = page_ticks.to_vec();
        if sequence == 0 && matches!(tamper, Tamper::TickRawDecodedMismatch) {
            decoded_ticks[0].0 += 1;
        }
        if sequence == 0 && matches!(tamper, Tamper::TickFinalRowMismatch) {
            decoded_ticks.last_mut().unwrap().1 += 1;
        }
        let decoded_sequence = if sequence == 1 && matches!(tamper, Tamper::TickPageOrdinalMismatch)
        {
            sequence + 1
        } else {
            sequence
        };
        decoded_arrays.push(decoded_tick_array(
            side,
            &decoded_ticks,
            matches!(tamper, Tamper::ExtraDecodedTickField),
            decoded_sequence,
        ));
        page_to = page_ticks.first().unwrap().0;
        pages.push(page);
    }
    let raw_path = format!("primary-{prefix}-pages-raw.vortex");
    let raw = if matches!(tamper, Tamper::CorruptVortex) {
        artifacts.corrupt(
            &raw_path,
            BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2,
        )
    } else {
        let dtype = raw_arrays[0].dtype().clone();
        artifacts.array(
            &raw_path,
            BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2,
            ChunkedArray::try_new(raw_arrays, dtype)
                .unwrap()
                .into_array(),
            None,
        )
    };
    decoded_arrays.reverse();
    let dtype = decoded_arrays[0].dtype().clone();
    let decoded = artifacts.array(
        &format!("primary-{prefix}-ticks-decoded.vortex"),
        BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV2,
        ChunkedArray::try_new(decoded_arrays, dtype)
            .unwrap()
            .into_array(),
        matches!(tamper, Tamper::WrongDeclaredRowCount).then_some(2),
    );
    let chunk = ExactBrokerRequestChunkV2::new(0, window, pages).expect("tick chunk");
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
    .expect("quote-side evidence")
}

fn evidence_row(
    sequence: u64,
    symbol_id: Option<i64>,
    quote_side: Option<QuoteSideV1>,
    evidence_kind: u8,
    requested_window: Option<EvidenceWindowV1>,
    client_msg_id: &str,
    payload_type: u32,
    payload_json: Value,
) -> EvidenceRow {
    EvidenceRow {
        sequence,
        account_id: ACCOUNT_ID,
        symbol_id,
        quote_side,
        evidence_kind,
        requested_window,
        client_msg_id: client_msg_id.to_owned(),
        payload_type,
        payload_json: payload_json.to_string(),
    }
}

fn tick_response(client_msg_id: &str, ticks: &[(i64, i64)], has_more: bool) -> String {
    let mut previous: Option<(i64, i64)> = None;
    let tick_data = ticks
        .iter()
        .rev()
        .map(|&(timestamp, price)| {
            let (wire_time, wire_price) = match previous {
                Some((previous_time, previous_price)) => {
                    (timestamp - previous_time, price - previous_price)
                }
                None => (timestamp, price),
            };
            previous = Some((timestamp, price));
            json!({"timestamp": wire_time, "tick": wire_price})
        })
        .collect::<Vec<_>>();
    json!({
        "clientMsgId":client_msg_id,"payloadType":2146,
        "payload":{"ctidTraderAccountId":ACCOUNT_ID,"hasMore":has_more,
            "tickData": tick_data}
    })
    .to_string()
}

fn raw_tick_page_array(
    window: EvidenceWindowV1,
    page: &ExactBrokerRequestPageV2,
    side: QuoteSideV1,
    raw_response_json: &str,
    ticks: &[(i64, i64)],
) -> vortex_array::ArrayRef {
    StructArray::from_fields(&[
        (
            "chunk_sequence",
            PrimitiveArray::from_iter([0_u64]).into_array(),
        ),
        (
            "page_sequence_in_chunk",
            PrimitiveArray::from_iter([page.page_sequence_in_chunk()]).into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter([ACCOUNT_ID]).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter([SYMBOL_ID]).into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter([quote_side_code(side)]).into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(vec![page.client_msg_id()]).into_array(),
        ),
        (
            "chunk_from_unix_ms_inclusive",
            PrimitiveArray::from_iter([window.from_unix_ms_inclusive()]).into_array(),
        ),
        (
            "chunk_to_unix_ms_exclusive",
            PrimitiveArray::from_iter([window.to_unix_ms_exclusive()]).into_array(),
        ),
        (
            "page_from_unix_ms_inclusive",
            PrimitiveArray::from_iter([page.requested_window().from_unix_ms_inclusive()])
                .into_array(),
        ),
        (
            "page_to_unix_ms_exclusive",
            PrimitiveArray::from_iter([page.requested_window().to_unix_ms_exclusive()])
                .into_array(),
        ),
        (
            "first_tick_timestamp_ms",
            PrimitiveArray::from_iter([ticks.first().expect("non-empty fixture ticks").0])
                .into_array(),
        ),
        (
            "last_tick_timestamp_ms",
            PrimitiveArray::from_iter([ticks.last().expect("non-empty fixture ticks").0])
                .into_array(),
        ),
        (
            "decoded_tick_count",
            PrimitiveArray::from_iter([ticks.len() as u64]).into_array(),
        ),
        (
            "has_more",
            PrimitiveArray::from_iter([u8::from(page.response_has_more())]).into_array(),
        ),
        (
            "raw_response_json",
            VarBinArray::from(vec![raw_response_json]).into_array(),
        ),
    ])
    .expect("raw tick array")
    .into_array()
}

fn decoded_tick_array(
    side: QuoteSideV1,
    ticks: &[(i64, i64)],
    extra_field: bool,
    page_sequence: u64,
) -> vortex_array::ArrayRef {
    let mut fields = vec![
        (
            "chunk_sequence",
            PrimitiveArray::from_iter(vec![0_u64; ticks.len()]).into_array(),
        ),
        (
            "page_sequence_in_chunk",
            PrimitiveArray::from_iter(vec![page_sequence; ticks.len()]).into_array(),
        ),
        (
            "row_sequence_in_page",
            PrimitiveArray::from_iter(0..ticks.len() as u64).into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(vec![ACCOUNT_ID; ticks.len()]).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(vec![SYMBOL_ID; ticks.len()]).into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter(vec![quote_side_code(side); ticks.len()]).into_array(),
        ),
        (
            "timestamp_ms",
            PrimitiveArray::from_iter(ticks.iter().map(|tick| tick.0)).into_array(),
        ),
        (
            "price",
            PrimitiveArray::from_iter(ticks.iter().map(|tick| tick.1 as f64 / 100_000.0))
                .into_array(),
        ),
    ];
    if extra_field {
        fields.push((
            "unexpected",
            PrimitiveArray::from_iter(vec![1_u8; ticks.len()]).into_array(),
        ));
    }
    StructArray::from_fields(&fields)
        .expect("decoded tick array")
        .into_array()
}

fn evidence_array(rows: &[EvidenceRow]) -> vortex_array::ArrayRef {
    let has_symbol_id = rows.iter().map(|row| u8::from(row.symbol_id.is_some()));
    let symbol_ids = rows.iter().map(|row| row.symbol_id.unwrap_or(0));
    let has_quote_side = rows.iter().map(|row| u8::from(row.quote_side.is_some()));
    let quote_sides = rows
        .iter()
        .map(|row| row.quote_side.map_or(0, quote_side_code));
    let has_window = rows
        .iter()
        .map(|row| u8::from(row.requested_window.is_some()));
    let from = rows.iter().map(|row| {
        row.requested_window
            .map_or(0, |window| window.from_unix_ms_inclusive())
    });
    let to = rows.iter().map(|row| {
        row.requested_window
            .map_or(0, |window| window.to_unix_ms_exclusive())
    });
    StructArray::from_fields(&[
        (
            "sequence",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.sequence)).into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.account_id)).into_array(),
        ),
        (
            "evidence_kind",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.evidence_kind)).into_array(),
        ),
        (
            "has_symbol_id",
            PrimitiveArray::from_iter(has_symbol_id).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(symbol_ids).into_array(),
        ),
        (
            "has_quote_side",
            PrimitiveArray::from_iter(has_quote_side).into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter(quote_sides).into_array(),
        ),
        (
            "has_requested_window",
            PrimitiveArray::from_iter(has_window).into_array(),
        ),
        (
            "requested_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(from).into_array(),
        ),
        (
            "requested_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(to).into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(
                rows.iter()
                    .map(|row| row.client_msg_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ),
        (
            "payload_type",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.payload_type)).into_array(),
        ),
        (
            "payload_json",
            VarBinArray::from(
                rows.iter()
                    .map(|row| row.payload_json.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ),
    ])
    .expect("evidence array")
    .into_array()
}

fn raw_deal_page_array(
    window: EvidenceWindowV1,
    raw_response_json: &str,
    mismatched_has_more: bool,
) -> vortex_array::ArrayRef {
    StructArray::from_fields(&[
        (
            "chunk_sequence",
            PrimitiveArray::from_iter([0_u64]).into_array(),
        ),
        (
            "page_sequence_in_chunk",
            PrimitiveArray::from_iter([0_u64]).into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter([ACCOUNT_ID]).into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(vec!["deal-page"]).into_array(),
        ),
        (
            "chunk_from_unix_ms_inclusive",
            PrimitiveArray::from_iter([window.from_unix_ms_inclusive()]).into_array(),
        ),
        (
            "chunk_to_unix_ms_exclusive",
            PrimitiveArray::from_iter([window.to_unix_ms_exclusive()]).into_array(),
        ),
        (
            "page_from_unix_ms_inclusive",
            PrimitiveArray::from_iter([window.from_unix_ms_inclusive()]).into_array(),
        ),
        (
            "page_to_unix_ms_exclusive",
            PrimitiveArray::from_iter([window.to_unix_ms_exclusive()]).into_array(),
        ),
        (
            "max_rows",
            PrimitiveArray::from_iter([100_u32]).into_array(),
        ),
        ("has_events", PrimitiveArray::from_iter([0_u8]).into_array()),
        (
            "first_deal_execution_timestamp_ms",
            PrimitiveArray::from_iter([0_i64]).into_array(),
        ),
        (
            "last_deal_execution_timestamp_ms",
            PrimitiveArray::from_iter([0_i64]).into_array(),
        ),
        (
            "decoded_deal_count",
            PrimitiveArray::from_iter([0_u64]).into_array(),
        ),
        (
            "has_more",
            PrimitiveArray::from_iter([u8::from(mismatched_has_more)]).into_array(),
        ),
        (
            "raw_response_json",
            VarBinArray::from(vec![raw_response_json]).into_array(),
        ),
    ])
    .expect("raw deal page array")
    .into_array()
}

fn write_vortex(path: &Path, array: vortex_array::ArrayRef) {
    let mut file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)
        .expect("create Vortex fixture");
    let mut writer = VORTEX_SESSION
        .write_options()
        .blocking(&*VORTEX_RUNTIME)
        .writer(&mut file, array.dtype().clone());
    writer.push(array).expect("write Vortex row batch");
    writer.finish().expect("finish Vortex fixture");
    file.flush().expect("flush Vortex fixture");
}

fn quote_side_code(side: QuoteSideV1) -> u8 {
    match side {
        QuoteSideV1::Bid => 0,
        QuoteSideV1::Ask => 1,
    }
}

fn unique_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let sequence = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "neoethos-broker-truth-semantic-v2-{}-{nonce}-{sequence}",
        std::process::id()
    ))
}
