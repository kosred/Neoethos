use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use neoethos_broker_truth::{
    BrokerFinancialOperationV1, BrokerFinancialTruthArtifactSourceV1,
    BrokerFinancialTruthBindingV1, BrokerFinancialTruthBundleManifestV2,
    BrokerFinancialTruthBundleStoreV1, BrokerFinancialTruthSemanticIngressErrorCodeV2,
    BrokerFinancialTruthVortexSchemaV1, EvidenceWindowV1, ExactBrokerRequestChunkV2,
    ExactBrokerRequestPageV2, ExactCapturedEvidencePairV1, ExactConversionRouteEvidenceV2,
    ExactDealReconciliationEvidenceV2, ExactQuoteSideEvidenceV2, ExactSymbolContractEvidenceV2,
    ImmutableVortexArtifactV1, QuoteSideV1, ReviewedQuoteReplayRuleEvidenceV2,
    ReviewedQuoteReplayRuleIdentityV2, SynchronizedBidAskEvidenceV2,
    current_broker_financial_truth_capability_v1,
    inspect_untrusted_broker_financial_truth_bundle_v2,
};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use serde_json::{Value, json};
use vortex_array::IntoArray;
use vortex_array::arrays::{PrimitiveArray, StructArray, VarBinArray};
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::{RuntimeSession, RuntimeSessionExt};
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

const ACCOUNT_ID: i64 = 7;
const SYMBOL_ID: i64 = 42;
const WINDOW_FROM: i64 = 1_700_000_000_000;
const WINDOW_TO: i64 = WINDOW_FROM + 60_000;
const TICK_TIMESTAMP: i64 = WINDOW_FROM + 30_000;
const REVIEW_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const PROTOCOL_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const OBSERVATION_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

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
enum Tamper {
    None,
    CorruptVortex,
    ExtraDecodedTickField,
    WrongDeclaredRowCount,
    TickRawDecodedMismatch,
    InvalidRawEnvelope,
    GenericRawDecodedLinkMismatch,
    DealPageMismatch,
}

struct FixtureRoot(PathBuf);

impl Drop for FixtureRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
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

#[test]
fn structurally_consistent_synthetic_bundle_remains_explicitly_untrusted() {
    let (_root, verified) = fixture(Tamper::None);
    let ingress = inspect_untrusted_broker_financial_truth_bundle_v2(verified)
        .expect("synthetic rows may prove only sealed structural ingress");
    assert_eq!(ingress.artifact_count(), 16);
    assert_eq!(ingress.bundle_schema_version(), 2);

    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("untrusted structural ingress must never authorize finance");
}

#[test]
fn corrupt_or_schema_tampered_vortex_is_refused_before_row_semantics() {
    for (tamper, expected) in [
        (
            Tamper::CorruptVortex,
            BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexReadFailed,
        ),
        (
            Tamper::ExtraDecodedTickField,
            BrokerFinancialTruthSemanticIngressErrorCodeV2::VortexSchemaMismatch,
        ),
        (
            Tamper::WrongDeclaredRowCount,
            BrokerFinancialTruthSemanticIngressErrorCodeV2::ArtifactRowCountMismatch,
        ),
    ] {
        let (_root, verified) = fixture(tamper);
        let error = inspect_untrusted_broker_financial_truth_bundle_v2(verified)
            .expect_err("structural tampering must fail closed");
        assert_eq!(error.code(), expected, "unexpected error: {error}");
    }
}

#[test]
fn raw_decoded_identity_and_tick_mismatches_are_refused() {
    for tamper in [
        Tamper::TickRawDecodedMismatch,
        Tamper::InvalidRawEnvelope,
        Tamper::GenericRawDecodedLinkMismatch,
        Tamper::DealPageMismatch,
    ] {
        let (_root, verified) = fixture(tamper);
        let error = inspect_untrusted_broker_financial_truth_bundle_v2(verified)
            .expect_err("raw/decoded divergence must fail closed");
        assert!(
            matches!(
                error.code(),
                BrokerFinancialTruthSemanticIngressErrorCodeV2::InvalidRawEnvelope
                    | BrokerFinancialTruthSemanticIngressErrorCodeV2::RawDecodedMismatch
            ),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn source_surface_contains_no_authority_bridge_or_mutable_selector() {
    let source = include_str!("../src/semantic_v2.rs");
    for forbidden in [
        "BrokerFinancialTruthPermitV1",
        "BrokerFinancialTruthCapabilityV1",
        "current_broker_financial_truth",
        "LazyLock",
        "OnceLock",
        "static mut",
        "std::env",
        "symbol_metadata.json",
        "exact_pip_size_v1",
    ] {
        assert!(
            !source.contains(forbidden),
            "Chunk 3a structural ingress contains forbidden authority token {forbidden}"
        );
    }
    assert!(source.contains("UntrustedBrokerFinancialTruthIngressV2"));
    assert!(source.contains("VerifiedImmutableBrokerFinancialTruthBundleV2"));
}

fn fixture(
    tamper: Tamper,
) -> (
    FixtureRoot,
    neoethos_broker_truth::VerifiedImmutableBrokerFinancialTruthBundleV2,
) {
    let root = FixtureRoot(unique_root());
    let source_root = root.0.join("sources");
    fs::create_dir_all(&source_root).expect("create source directory");
    let mut artifacts = ArtifactBuilder::new(source_root);
    let window = EvidenceWindowV1::new(WINDOW_FROM, WINDOW_TO).expect("fixture window");
    let binding = binding(window);

    let bid_raw_json = tick_response("bid-page", 110_000);
    let ask_raw_json = tick_response("ask-page", 120_000);
    let bid_raw = if matches!(tamper, Tamper::CorruptVortex) {
        artifacts.corrupt(
            "primary-bid-pages-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2,
        )
    } else {
        artifacts.array(
            "primary-bid-pages-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2,
            raw_tick_page_array(
                "bid-page",
                QuoteSideV1::Bid,
                if matches!(tamper, Tamper::InvalidRawEnvelope) {
                    r#"{"payloadType":2146,"payload":{}}"#
                } else {
                    &bid_raw_json
                },
            ),
            None,
        )
    };
    let bid_decoded = artifacts.array(
        "primary-bid-ticks-decoded.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV2,
        decoded_tick_array(
            QuoteSideV1::Bid,
            if matches!(tamper, Tamper::TickRawDecodedMismatch) {
                TICK_TIMESTAMP + 1
            } else {
                TICK_TIMESTAMP
            },
            1.1,
            matches!(tamper, Tamper::ExtraDecodedTickField),
        ),
        matches!(tamper, Tamper::WrongDeclaredRowCount).then_some(2),
    );
    let ask_raw = artifacts.array(
        "primary-ask-pages-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2,
        raw_tick_page_array("ask-page", QuoteSideV1::Ask, &ask_raw_json),
        None,
    );
    let ask_decoded = artifacts.array(
        "primary-ask-ticks-decoded.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV2,
        decoded_tick_array(QuoteSideV1::Ask, TICK_TIMESTAMP, 1.2, false),
        None,
    );
    let bid = quote_side(QuoteSideV1::Bid, "bid-page", window, bid_raw, bid_decoded);
    let ask = quote_side(QuoteSideV1::Ask, "ask-page", window, ask_raw, ask_decoded);

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
    let review_identity =
        ReviewedQuoteReplayRuleIdentityV2::new(REVIEW_SHA, PROTOCOL_SHA, OBSERVATION_SHA)
            .expect("syntactic review identity only");
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
                "dealPageRequest":{"fromTimestamp":WINDOW_FROM,"toTimestamp":WINDOW_TO,"maxRows":100},
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
    (root, verified)
}

fn binding(window: EvidenceWindowV1) -> BrokerFinancialTruthBindingV1 {
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
        "11".repeat(32),
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

fn quote_side(
    side: QuoteSideV1,
    client_msg_id: &str,
    window: EvidenceWindowV1,
    raw: ImmutableVortexArtifactV1,
    decoded: ImmutableVortexArtifactV1,
) -> ExactQuoteSideEvidenceV2 {
    let page = ExactBrokerRequestPageV2::new(
        0,
        0,
        client_msg_id,
        window,
        Some(TICK_TIMESTAMP),
        Some(TICK_TIMESTAMP),
        1,
        false,
        None,
    )
    .expect("tick request page");
    let chunk = ExactBrokerRequestChunkV2::new(0, window, vec![page]).expect("tick chunk");
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

fn tick_response(client_msg_id: &str, raw_price: i64) -> String {
    json!({
        "clientMsgId":client_msg_id,"payloadType":2146,
        "payload":{"ctidTraderAccountId":ACCOUNT_ID,"hasMore":false,
            "tickData":[{"timestamp":TICK_TIMESTAMP,"tick":raw_price}]}
    })
    .to_string()
}

fn raw_tick_page_array(
    client_msg_id: &str,
    side: QuoteSideV1,
    raw_response_json: &str,
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
            "symbol_id",
            PrimitiveArray::from_iter([SYMBOL_ID]).into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter([quote_side_code(side)]).into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(vec![client_msg_id]).into_array(),
        ),
        (
            "chunk_from_unix_ms_inclusive",
            PrimitiveArray::from_iter([WINDOW_FROM]).into_array(),
        ),
        (
            "chunk_to_unix_ms_exclusive",
            PrimitiveArray::from_iter([WINDOW_TO]).into_array(),
        ),
        (
            "page_from_unix_ms_inclusive",
            PrimitiveArray::from_iter([WINDOW_FROM]).into_array(),
        ),
        (
            "page_to_unix_ms_exclusive",
            PrimitiveArray::from_iter([WINDOW_TO]).into_array(),
        ),
        (
            "first_tick_timestamp_ms",
            PrimitiveArray::from_iter([TICK_TIMESTAMP]).into_array(),
        ),
        (
            "last_tick_timestamp_ms",
            PrimitiveArray::from_iter([TICK_TIMESTAMP]).into_array(),
        ),
        (
            "decoded_tick_count",
            PrimitiveArray::from_iter([1_u64]).into_array(),
        ),
        ("has_more", PrimitiveArray::from_iter([0_u8]).into_array()),
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
    timestamp_ms: i64,
    price: f64,
    extra_field: bool,
) -> vortex_array::ArrayRef {
    let mut fields = vec![
        (
            "chunk_sequence",
            PrimitiveArray::from_iter([0_u64]).into_array(),
        ),
        (
            "page_sequence_in_chunk",
            PrimitiveArray::from_iter([0_u64]).into_array(),
        ),
        (
            "row_sequence_in_page",
            PrimitiveArray::from_iter([0_u64]).into_array(),
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
            "timestamp_ms",
            PrimitiveArray::from_iter([timestamp_ms]).into_array(),
        ),
        ("price", PrimitiveArray::from_iter([price]).into_array()),
    ];
    if extra_field {
        fields.push(("unexpected", PrimitiveArray::from_iter([1_u8]).into_array()));
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
            PrimitiveArray::from_iter([WINDOW_FROM]).into_array(),
        ),
        (
            "chunk_to_unix_ms_exclusive",
            PrimitiveArray::from_iter([WINDOW_TO]).into_array(),
        ),
        (
            "page_from_unix_ms_inclusive",
            PrimitiveArray::from_iter([WINDOW_FROM]).into_array(),
        ),
        (
            "page_to_unix_ms_exclusive",
            PrimitiveArray::from_iter([WINDOW_TO]).into_array(),
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
