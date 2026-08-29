use neoethos_broker_truth::{
    BrokerFinancialTruthContractErrorCodeV1, BrokerFinancialTruthVortexSchemaV1, EvidenceWindowV1,
    ExactBrokerRequestChunkV2, ExactBrokerRequestPageV2, ExactDealReconciliationEvidenceV2,
    ExactSymbolContractEvidenceV2, ImmutableVortexArtifactV1, MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2,
    ReviewedQuoteReplayRuleIdentityV2,
};

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

fn window(from: i64, to: i64) -> EvidenceWindowV1 {
    EvidenceWindowV1::new(from, to).expect("valid half-open fixture window")
}

fn page(
    chunk_sequence: u64,
    page_sequence_in_chunk: u64,
    request_window: EvidenceWindowV1,
    first_event_ms: Option<i64>,
    last_event_ms: Option<i64>,
    event_count: u64,
    response_has_more: bool,
    max_rows: Option<u32>,
) -> ExactBrokerRequestPageV2 {
    ExactBrokerRequestPageV2::new(
        chunk_sequence,
        page_sequence_in_chunk,
        format!("chunk-{chunk_sequence}-page-{page_sequence_in_chunk}"),
        request_window,
        first_event_ms,
        last_event_ms,
        event_count,
        response_has_more,
        max_rows,
    )
    .expect("valid exact response-page fixture")
}

fn artifact(path: &str, schema: BrokerFinancialTruthVortexSchemaV1) -> ImmutableVortexArtifactV1 {
    ImmutableVortexArtifactV1::new(path, schema, SHA_A, 16, 1).expect("valid structural artifact")
}

#[test]
fn quote_chunks_are_exact_contiguous_and_never_exceed_seven_days() {
    let full = window(0, MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2 * 2);
    let newest = window(
        MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2,
        MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2 * 2,
    );
    let oldest = window(0, MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2);
    let chunks = vec![
        ExactBrokerRequestChunkV2::new(
            0,
            newest,
            vec![page(
                0,
                0,
                newest,
                Some(newest.from_unix_ms_inclusive()),
                Some(newest.to_unix_ms_exclusive() - 1),
                2,
                false,
                None,
            )],
        )
        .expect("newest exact chunk"),
        ExactBrokerRequestChunkV2::new(
            1,
            oldest,
            vec![page(
                1,
                0,
                oldest,
                Some(oldest.from_unix_ms_inclusive()),
                Some(oldest.to_unix_ms_exclusive() - 1),
                2,
                false,
                None,
            )],
        )
        .expect("oldest exact chunk"),
    ];
    ExactBrokerRequestChunkV2::validate_quote_partition(full, &chunks)
        .expect("two exact seven-day chunks cover the full window");

    let oversized = window(0, MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2 + 1);
    let oversized = ExactBrokerRequestChunkV2::new(
        0,
        oversized,
        vec![page(
            0,
            0,
            oversized,
            Some(0),
            Some(MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2),
            2,
            false,
            None,
        )],
    )
    .expect("generic page structure is well formed");
    let error = ExactBrokerRequestChunkV2::validate_quote_partition(
        oversized.requested_window(),
        &[oversized],
    )
    .expect_err("a cTrader tick request chunk over seven days must fail closed");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthContractErrorCodeV1::InvalidWindow
    );

    let gap = window(1, MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2);
    let gapped = vec![
        chunks[0].clone(),
        ExactBrokerRequestChunkV2::new(
            1,
            gap,
            vec![page(
                1,
                0,
                gap,
                Some(1),
                Some(gap.to_unix_ms_exclusive() - 1),
                2,
                false,
                None,
            )],
        )
        .expect("structural older chunk"),
    ];
    ExactBrokerRequestChunkV2::validate_quote_partition(full, &gapped)
        .expect_err("a one-millisecond quote coverage gap must fail closed");
}

#[test]
fn page_boundaries_preserve_has_more_and_exclusive_pagination() {
    let chunk_window = window(100, 1_000);
    let first = page(0, 0, chunk_window, Some(700), Some(900), 2, true, None);
    let older_window = window(100, 700);
    let older = page(0, 1, older_window, Some(200), Some(600), 2, false, None);
    ExactBrokerRequestChunkV2::new(0, chunk_window, vec![first.clone(), older])
        .expect("exact newest-first page boundary chain");

    let truncated = page(0, 1, older_window, Some(200), Some(600), 2, true, None);
    ExactBrokerRequestChunkV2::new(0, chunk_window, vec![first.clone(), truncated])
        .expect_err("terminal hasMore=true must fail closed");

    let overlapping = page(0, 1, window(100, 701), Some(200), Some(700), 2, false, None);
    ExactBrokerRequestChunkV2::new(0, chunk_window, vec![first, overlapping])
        .expect_err("older page must use the prior oldest event as its exclusive boundary");
}

#[test]
fn exact_symbol_authority_requires_light_full_asset_trader_and_decoded_artifacts() {
    let light = artifact(
        "light-symbols-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderLightSymbolResponsesRawV2,
    );
    let full = artifact(
        "full-symbols-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV2,
    );
    let assets = artifact(
        "assets-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderAccountAssetResponsesRawV2,
    );
    let trader = artifact(
        "trader-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderTraderAccountResponsesRawV2,
    );
    let decoded = artifact(
        "contracts-decoded.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderSymbolMoneyContractsDecodedV2,
    );
    ExactSymbolContractEvidenceV2::new(light.clone(), full, assets, trader, decoded)
        .expect("all raw broker authorities are explicit");

    let wrong_light = ImmutableVortexArtifactV1::new(
        "light-symbols-wrong.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV2,
        SHA_A,
        16,
        1,
    )
    .expect("structural wrong-schema artifact");
    ExactSymbolContractEvidenceV2::new(
        wrong_light,
        artifact(
            "full-2.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV2,
        ),
        artifact(
            "assets-2.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderAccountAssetResponsesRawV2,
        ),
        artifact(
            "trader-2.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderTraderAccountResponsesRawV2,
        ),
        artifact(
            "decoded-2.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolMoneyContractsDecodedV2,
        ),
    )
    .expect_err("full-symbol bytes cannot stand in for raw light-symbol authority");
    assert_eq!(
        light.schema(),
        BrokerFinancialTruthVortexSchemaV1::CTraderLightSymbolResponsesRawV2
    );
}

#[test]
fn reviewed_replay_rule_identity_has_three_exact_immutable_inputs_and_no_default() {
    let identity = ReviewedQuoteReplayRuleIdentityV2::new(SHA_A, SHA_B, SHA_C)
        .expect("three exact review inputs form one content identity");
    assert!(identity.identity_sha256().len() == 64);
    assert_eq!(identity.review_record_sha256(), SHA_A);

    let error = ReviewedQuoteReplayRuleIdentityV2::new(SHA_A, SHA_B, "not-a-sha")
        .expect_err("a label cannot replace exact broker-observation evidence identity");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthContractErrorCodeV1::InvalidSha256
    );

    let encoded = serde_json::to_value(&identity).expect("serialize review identity");
    let mut object = encoded.as_object().expect("identity object").clone();
    object.insert("current".to_owned(), serde_json::json!(true));
    serde_json::from_value::<ReviewedQuoteReplayRuleIdentityV2>(serde_json::Value::Object(object))
        .expect_err("unknown mutable/current authority must fail closed");
}

#[test]
fn deal_reconciliation_requires_exact_terminal_paged_raw_evidence() {
    let deal_window = window(1_000, 2_000);
    let pages = ExactBrokerRequestChunkV2::new(
        0,
        deal_window,
        vec![page(0, 0, deal_window, None, None, 0, false, Some(100))],
    )
    .expect("an exact empty terminal DealList page is evidence");
    ExactDealReconciliationEvidenceV2::new(
        deal_window,
        pages,
        artifact(
            "reconcile-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderReconcileResponsesRawV2,
        ),
        artifact(
            "deal-pages-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderDealPagesRawV2,
        ),
        artifact(
            "deal-decoded.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV2,
        ),
    )
    .expect("complete terminal DealList boundary plus raw reconcile/deal artifacts");
}

#[test]
fn canonical_v2_producer_surface_is_distinct_and_has_no_adapter_or_permit() {
    let capture_source = include_str!("../src/broker_truth_capture.rs");
    let vortex_source = include_str!("../src/broker_truth_vortex.rs");
    let lib_source = include_str!("../src/lib.rs");

    for needle in [
        "ExactBrokerTruthCaptureSessionV2",
        "CapturedTickPageV2",
        "CapturedDealPageV2",
        "capture_and_publish_broker_financial_truth_v2",
        "ReviewedQuoteReplayRuleIdentityV2",
        "LightSymbolResponse",
    ] {
        assert!(
            capture_source.contains(needle),
            "missing V2 producer token {needle}"
        );
    }
    assert!(vortex_source.contains("BrokerFinancialTruthBundleManifestV2"));
    assert!(vortex_source.contains("CTraderLightSymbolResponsesRawV2"));
    assert!(vortex_source.contains("CTraderDealPagesRawV2"));
    assert!(lib_source.contains("pub mod broker_truth_capture"));
    assert!(!capture_source.contains("BrokerFinancialTruthPermitV1"));
    assert!(!capture_source.contains("ProductionCTraderOpenApiTransport::new"));
    assert!(!capture_source.contains("send_sequence"));
    assert!(!capture_source.contains("current_broker_financial_truth"));
}
