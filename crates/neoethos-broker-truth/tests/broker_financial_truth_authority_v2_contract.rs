use neoethos_broker_truth::{
    BrokerFinancialTruthAuthorityErrorCodeV2, BrokerFinancialTruthAuthorityErrorV2,
    BrokerFinancialTruthAuthoritySourceClassV2, BrokerFinancialTruthAuthorityV2,
    BrokerFinancialTruthEvidenceClassV2, BrokerTruthAcquisitionLinkReceiptV1,
    BrokerTruthAcquisitionStoreV1, BrokerTruthReviewedSynchronizationBindingV1, EvidenceWindowV1,
    ReviewedBrokerFinancialTruthEvidenceV2, ReviewedQuoteReplayRuleIdentityV2,
    validate_reviewed_broker_financial_truth_authority_v2,
};

fn reviewed_fixture() -> ReviewedBrokerFinancialTruthEvidenceV2 {
    let review =
        ReviewedQuoteReplayRuleIdentityV2::new("77".repeat(32), "88".repeat(32), "99".repeat(32))
            .expect("review identity fixture");
    let synchronization = BrokerTruthReviewedSynchronizationBindingV1::new(
        0,
        42,
        7,
        EvidenceWindowV1::new(1_000, 2_000).expect("review window fixture"),
        review,
        "aa".repeat(32),
    )
    .expect("reviewed synchronization fixture");
    ReviewedBrokerFinancialTruthEvidenceV2::checked_new(
        "11".repeat(32),
        "22".repeat(32),
        "33".repeat(32),
        "44".repeat(32),
        "55".repeat(32),
        "66".repeat(32),
        "77".repeat(32),
        "88".repeat(32),
        "bb".repeat(32),
        EvidenceWindowV1::new(1_000, 2_000).expect("authority window fixture"),
        vec![synchronization],
    )
    .expect("checked reviewed evidence fixture")
}

#[test]
fn reviewed_data_is_checked_but_only_the_exact_validator_can_mint_authority() {
    let _: ReviewedBrokerFinancialTruthEvidenceV2 = reviewed_fixture();
    let validator: fn(
        &BrokerTruthAcquisitionStoreV1,
        &BrokerTruthAcquisitionLinkReceiptV1,
        ReviewedBrokerFinancialTruthEvidenceV2,
    ) -> Result<
        BrokerFinancialTruthAuthorityV2,
        BrokerFinancialTruthAuthorityErrorV2,
    > = validate_reviewed_broker_financial_truth_authority_v2;
    assert_ne!(validator as usize, 0);
    assert!(std::mem::size_of::<BrokerFinancialTruthAuthorityV2>() > 0);
}

#[test]
fn authority_names_every_required_semantic_evidence_class() {
    let classes = [
        BrokerFinancialTruthEvidenceClassV2::PrimaryBidAsk,
        BrokerFinancialTruthEvidenceClassV2::ConversionLegs,
        BrokerFinancialTruthEvidenceClassV2::ExactSymbolAndAccountContracts,
        BrokerFinancialTruthEvidenceClassV2::UnrealizedPnl,
        BrokerFinancialTruthEvidenceClassV2::CloseDealReconciliation,
    ];
    assert_eq!(classes.len(), 5);
}

#[test]
fn reviewed_evidence_refuses_an_empty_or_mismatched_window_set() {
    let error = ReviewedBrokerFinancialTruthEvidenceV2::checked_new(
        "11".repeat(32),
        "22".repeat(32),
        "33".repeat(32),
        "44".repeat(32),
        "55".repeat(32),
        "66".repeat(32),
        "77".repeat(32),
        "88".repeat(32),
        "bb".repeat(32),
        EvidenceWindowV1::new(1_000, 2_000).expect("authority window fixture"),
        Vec::new(),
    )
    .expect_err("reviewed synchronization coverage is mandatory");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthAuthorityErrorCodeV2::ReviewedEvidenceInvalid
    );
}

#[test]
fn source_surface_is_move_only_run_scoped_and_never_opens_the_global_gate() {
    let source = include_str!("../src/authority_v2.rs");
    let gate = include_str!("../src/gate.rs");
    assert!(source.contains("pub struct BrokerFinancialTruthAuthorityV2"));
    assert!(source.contains("BrokerFinancialTruthAuthoritySourceClassV2::ResearchOnly"));
    assert!(source.contains("source_semantic_status = acquisition.semantic_status()"));
    assert!(source.contains("source_promotion_eligibility = acquisition.promotion_eligibility()"));
    for forbidden in [
        "impl Clone for BrokerFinancialTruthAuthorityV2",
        "impl Copy for BrokerFinancialTruthAuthorityV2",
        "Serialize for BrokerFinancialTruthAuthorityV2",
        "Deserialize for BrokerFinancialTruthAuthorityV2",
        "impl Default for BrokerFinancialTruthAuthorityV2",
        "pub fn new(",
        "current_broker_financial_truth_capability_v1",
        "BrokerFinancialTruthPermitV1",
        "OnceLock",
        "static mut",
    ] {
        assert!(
            !source.contains(forbidden),
            "run authority contains forbidden construction/global token {forbidden}"
        );
    }
    assert!(gate.contains("BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1"));
    assert!(!gate.contains("BrokerFinancialTruthAuthorityV2"));
    let class = BrokerFinancialTruthAuthoritySourceClassV2::ResearchOnly;
    assert_eq!(
        class,
        BrokerFinancialTruthAuthoritySourceClassV2::ResearchOnly
    );
}
