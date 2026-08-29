use neoethos_broker_truth::{
    BrokerTruthAcquisitionLinkReceiptV1, BrokerTruthAcquisitionStoreV1,
    CanonicalBarSignalResearchDecisionV1, ClosedCanonicalBarTrailingThresholdV1,
    CompleteBidAskQuoteReplayEvidenceV1, CompleteQuoteSideCoverageV1, EvidenceWindowV1,
    ExactHistoricalQuoteV1, ExactQuoteSourceOrdinalV1, LockedFinalistOosReplayScopeV1, QuoteSideV1,
    QuoteValidatedResearchAuthorityV1, QuoteValidatedResearchReplayBindingV1,
    QuoteValidatedResearchReplayErrorCodeV1, QuoteValidatedResearchReplayErrorV1,
    QuoteValidatedResearchReplayPlanV1, QuoteValidatedResearchReplayPolicyV1,
    ResearchPositionDirectionV1, ReviewedQuoteReplayRuleIdentityV2,
    SealedHistoricalBidAskQuoteReplayEvidenceV1, SealedHistoricalQuoteValidatedResearchLedgerV1,
    VersionedLatencySlippagePolicyV1, open_sealed_historical_bid_ask_quote_replay_evidence_v1,
    replay_quote_validated_research_v1, replay_sealed_quote_validated_research_v1,
};

const ACCOUNT_ID: i64 = 7;
const SYMBOL_ID: i64 = 42;
const SYMBOL_NAME: &str = "EURUSD";
const WINDOW_FROM_MS: i64 = 1_700_000_000_000;
const WINDOW_TO_MS: i64 = WINDOW_FROM_MS + 60_000;
const SIGNAL_BAR_OPEN_MS: i64 = WINDOW_FROM_MS + 5_000;
const DECISION_AT_MS: i64 = WINDOW_FROM_MS + 10_000;
const PIP_SIZE: f64 = 0.0001;

const SEARCH_RECEIPT_SHA256: &str =
    "1111111111111111111111111111111111111111111111111111111111111111";
const SIGNAL_PLAN_SHA256: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const REVIEW_SHA256: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const PROTOCOL_SHA256: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const OBSERVATION_SHA256: &str = "5555555555555555555555555555555555555555555555555555555555555555";
const MANIFEST_SHA256: &str = "6666666666666666666666666666666666666666666666666666666666666666";
const BID_RAW_SHA256: &str = "7777777777777777777777777777777777777777777777777777777777777777";
const BID_DECODED_SHA256: &str = "8888888888888888888888888888888888888888888888888888888888888888";
const ASK_RAW_SHA256: &str = "9999999999999999999999999999999999999999999999999999999999999999";
const ASK_DECODED_SHA256: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn replay_scope() -> LockedFinalistOosReplayScopeV1 {
    LockedFinalistOosReplayScopeV1::new(
        EvidenceWindowV1::new(WINDOW_FROM_MS, WINDOW_TO_MS).expect("locked OOS window"),
        5_000,
        5_000,
    )
    .expect("locked replay scope")
}

fn binding() -> QuoteValidatedResearchReplayBindingV1 {
    QuoteValidatedResearchReplayBindingV1::new(
        SEARCH_RECEIPT_SHA256,
        SIGNAL_PLAN_SHA256,
        ACCOUNT_ID,
        SYMBOL_ID,
        SYMBOL_NAME,
        replay_scope(),
        ReviewedQuoteReplayRuleIdentityV2::new(REVIEW_SHA256, PROTOCOL_SHA256, OBSERVATION_SHA256)
            .expect("reviewed replay rule identity"),
        MANIFEST_SHA256,
    )
    .expect("exact replay binding")
}

fn policy() -> QuoteValidatedResearchReplayPolicyV1 {
    QuoteValidatedResearchReplayPolicyV1::new(
        2_000,
        1_000,
        2_000,
        VersionedLatencySlippagePolicyV1::new(
            "quote-validated-latency-slippage-v1",
            1,
            0,
            0.5,
            PIP_SIZE,
        )
        .expect("versioned latency/slippage policy"),
        None,
    )
    .expect("replay policy")
}

fn decision(
    direction: ResearchPositionDirectionV1,
    stop_price: f64,
    target_price: f64,
) -> CanonicalBarSignalResearchDecisionV1 {
    CanonicalBarSignalResearchDecisionV1::new(
        SIGNAL_BAR_OPEN_MS,
        DECISION_AT_MS,
        direction,
        stop_price,
        target_price,
    )
    .expect("causal canonical-bar decision")
}

fn quote(timestamp_ms: i64, price: f64, row_index: u64) -> ExactHistoricalQuoteV1 {
    ExactHistoricalQuoteV1::new(
        timestamp_ms,
        price,
        ExactQuoteSourceOrdinalV1::new(0, 0, row_index).expect("source ordinal"),
    )
    .expect("exact quote")
}

fn side(
    quote_side: QuoteSideV1,
    quotes: Vec<ExactHistoricalQuoteV1>,
) -> CompleteQuoteSideCoverageV1 {
    let (raw_sha256, decoded_sha256) = match quote_side {
        QuoteSideV1::Bid => (BID_RAW_SHA256, BID_DECODED_SHA256),
        QuoteSideV1::Ask => (ASK_RAW_SHA256, ASK_DECODED_SHA256),
    };
    CompleteQuoteSideCoverageV1::new(
        quote_side,
        ACCOUNT_ID,
        SYMBOL_ID,
        replay_scope().required_quote_coverage_window(),
        raw_sha256,
        decoded_sha256,
        quotes,
        false,
    )
    .expect("syntactically complete caller-supplied side")
}

fn caller_supplied_evidence(
    bid: Vec<ExactHistoricalQuoteV1>,
    ask: Vec<ExactHistoricalQuoteV1>,
) -> CompleteBidAskQuoteReplayEvidenceV1 {
    CompleteBidAskQuoteReplayEvidenceV1::new(
        binding(),
        side(QuoteSideV1::Bid, bid),
        side(QuoteSideV1::Ask, ask),
    )
    .expect("syntactically complete caller-supplied book")
}

fn plan(
    decisions: Vec<CanonicalBarSignalResearchDecisionV1>,
) -> Result<QuoteValidatedResearchReplayPlanV1, QuoteValidatedResearchReplayErrorV1> {
    QuoteValidatedResearchReplayPlanV1::new(binding(), policy(), decisions, Vec::new())
}

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing production function `{signature}`"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing body for production function `{signature}`"));
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated body for production function `{signature}`")
}

#[test]
fn v1_is_exactly_one_decision_per_plan_and_ledger() {
    let first = decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0100);
    let second = CanonicalBarSignalResearchDecisionV1::new(
        SIGNAL_BAR_OPEN_MS + 1_000,
        DECISION_AT_MS + 1_000,
        ResearchPositionDirectionV1::Long,
        0.9950,
        1.0150,
    )
    .expect("a later independently valid decision");

    for decisions in [Vec::new(), vec![first.clone(), second]] {
        let error = plan(decisions).expect_err(
            "V1 has no decision identity or overlap policy, so zero/multiple decisions must refuse",
        );
        assert_eq!(
            error.code(),
            QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision
        );
    }

    plan(vec![first]).expect("exactly one decision is the complete V1 ownership boundary");
}

#[test]
fn the_single_v1_decision_owns_every_trailing_threshold_causally() {
    let research_decision = decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0100);
    let before_entry = ClosedCanonicalBarTrailingThresholdV1::new(
        SIGNAL_BAR_OPEN_MS - 1_000,
        SIGNAL_BAR_OPEN_MS,
        ResearchPositionDirectionV1::Long,
        0.9950,
    )
    .expect("individually causal but pre-entry threshold");
    let foreign_direction = ClosedCanonicalBarTrailingThresholdV1::new(
        DECISION_AT_MS,
        DECISION_AT_MS + 1_000,
        ResearchPositionDirectionV1::Short,
        1.0050,
    )
    .expect("individually causal but foreign-direction threshold");

    for threshold in [before_entry, foreign_direction] {
        let error = QuoteValidatedResearchReplayPlanV1::new(
            binding(),
            policy(),
            vec![research_decision.clone()],
            vec![threshold],
        )
        .expect_err("a threshold that the sole decision cannot own must fail closed");
        assert_eq!(
            error.code(),
            QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision
        );
    }
}

#[test]
fn crossed_synchronized_books_are_evidence_errors_for_long_and_short_entries() {
    for (direction, stop, target) in [
        (ResearchPositionDirectionV1::Long, 0.9900, 1.0100),
        (ResearchPositionDirectionV1::Short, 1.0100, 0.9900),
    ] {
        let evidence = match direction {
            ResearchPositionDirectionV1::Long => caller_supplied_evidence(
                vec![quote(DECISION_AT_MS, 1.0004, 0)],
                vec![quote(DECISION_AT_MS + 1, 1.0002, 0)],
            ),
            ResearchPositionDirectionV1::Short => caller_supplied_evidence(
                vec![quote(DECISION_AT_MS + 1, 1.0004, 0)],
                vec![quote(DECISION_AT_MS, 1.0002, 0)],
            ),
        };
        let error = replay_quote_validated_research_v1(
            &plan(vec![decision(direction, stop, target)]).expect("single decision"),
            evidence,
        )
        .expect_err("bid above ask is crossed evidence, never an executable synchronized book");
        assert_eq!(
            error.code(),
            QuoteValidatedResearchReplayErrorCodeV1::CrossedSynchronizedBook
        );
    }
}

#[test]
fn modeled_entry_must_remain_strictly_between_the_decision_stop_and_target() {
    for (direction, stop, target, bid, ask) in [
        (
            ResearchPositionDirectionV1::Long,
            0.9900,
            1.0001,
            1.0000,
            1.0001,
        ),
        (
            ResearchPositionDirectionV1::Short,
            1.0100,
            0.99996,
            1.0000,
            1.0002,
        ),
    ] {
        let evidence = match direction {
            ResearchPositionDirectionV1::Long => caller_supplied_evidence(
                vec![quote(DECISION_AT_MS, bid, 0)],
                vec![quote(DECISION_AT_MS + 1, ask, 0)],
            ),
            ResearchPositionDirectionV1::Short => caller_supplied_evidence(
                vec![quote(DECISION_AT_MS + 1, bid, 0)],
                vec![quote(DECISION_AT_MS, ask, 0)],
            ),
        };
        let error = replay_quote_validated_research_v1(
            &plan(vec![decision(direction, stop, target)]).expect("single decision"),
            evidence,
        )
        .expect_err(
            "gap plus modeled slippage outside the stop-target interval must not open a position",
        );
        assert_eq!(
            error.code(),
            QuoteValidatedResearchReplayErrorCodeV1::ModeledEntryOutsideDecisionBounds
        );
    }
}

#[test]
fn entry_ledger_retains_both_synchronized_sides_and_the_embedded_spread() {
    let evidence = caller_supplied_evidence(
        vec![quote(DECISION_AT_MS, 1.0000, 0)],
        vec![quote(DECISION_AT_MS + 1, 1.0002, 0)],
    );
    let ledger = replay_quote_validated_research_v1(
        &plan(vec![decision(
            ResearchPositionDirectionV1::Long,
            0.9900,
            1.0100,
        )])
        .expect("single decision"),
        evidence,
    )
    .expect("non-crossed synchronized entry book");
    let entry_book = ledger.positions()[0].entry_book();

    assert_eq!(
        entry_book.bid_reference().timestamp_unix_ms(),
        DECISION_AT_MS
    );
    assert_eq!(entry_book.bid_reference().price(), 1.0000);
    assert_eq!(
        entry_book.ask_reference().timestamp_unix_ms(),
        DECISION_AT_MS + 1
    );
    assert_eq!(entry_book.ask_reference().price(), 1.0002);
    assert!((entry_book.quoted_spread_price() - 0.0002).abs() < 1.0e-12);
}

#[test]
fn unreviewed_same_timestamp_opposite_update_is_ambiguous_even_with_a_fresh_prior_quote() {
    let evidence = caller_supplied_evidence(
        vec![
            quote(DECISION_AT_MS - 1, 1.0000, 0),
            quote(DECISION_AT_MS + 1, 1.0001, 1),
        ],
        vec![quote(DECISION_AT_MS + 1, 1.0002, 0)],
    );
    let error = replay_quote_validated_research_v1(
        &plan(vec![decision(
            ResearchPositionDirectionV1::Long,
            0.9900,
            1.0100,
        )])
        .expect("single decision"),
        evidence,
    )
    .expect_err("an unreviewed same-timestamp Bid update makes the actual entry spread ambiguous");
    assert_eq!(
        error.code(),
        QuoteValidatedResearchReplayErrorCodeV1::AmbiguousSameTimestampCrossSideOutcome
    );
}

#[test]
fn caller_supplied_quote_vectors_cannot_mint_historical_broker_authority() {
    let evidence = caller_supplied_evidence(
        vec![quote(DECISION_AT_MS, 1.0000, 0)],
        vec![quote(DECISION_AT_MS + 1, 1.0002, 0)],
    );
    let ledger = replay_quote_validated_research_v1(
        &plan(vec![decision(
            ResearchPositionDirectionV1::Long,
            0.9900,
            1.0100,
        )])
        .expect("single decision"),
        evidence,
    )
    .expect("caller-supplied semantic replay remains available as explicitly unverified research");

    assert_eq!(
        ledger.authority(),
        QuoteValidatedResearchAuthorityV1::UnverifiedCallerSuppliedQuotes
    );
    assert_eq!(
        ledger.receipt().authority(),
        QuoteValidatedResearchAuthorityV1::UnverifiedCallerSuppliedQuotes
    );
}

#[test]
fn historical_authority_requires_the_reopened_link_bft2_and_semantic_ingress_seal() {
    type OpenSealedEvidence = fn(
        &BrokerTruthAcquisitionStoreV1,
        &BrokerTruthAcquisitionLinkReceiptV1,
        &QuoteValidatedResearchReplayBindingV1,
    ) -> Result<
        SealedHistoricalBidAskQuoteReplayEvidenceV1,
        QuoteValidatedResearchReplayErrorV1,
    >;
    type ReplaySealedEvidence = fn(
        &QuoteValidatedResearchReplayPlanV1,
        SealedHistoricalBidAskQuoteReplayEvidenceV1,
    ) -> Result<
        SealedHistoricalQuoteValidatedResearchLedgerV1,
        QuoteValidatedResearchReplayErrorV1,
    >;

    let _: OpenSealedEvidence = open_sealed_historical_bid_ask_quote_replay_evidence_v1;
    let _: ReplaySealedEvidence = replay_sealed_quote_validated_research_v1;
}

#[test]
fn source_keeps_historical_authority_behind_exact_reopen_and_never_mints_a_permit() {
    let source = include_str!("../src/execution_replay_v1.rs");
    let lib = include_str!("../src/lib.rs");

    for required in [
        "pub struct SealedHistoricalBidAskQuoteReplayEvidenceV1",
        "pub struct SealedHistoricalQuoteValidatedResearchLedgerV1",
        "pub fn open_sealed_historical_bid_ask_quote_replay_evidence_v1(",
        ".open_link(",
        ".broker_truth_receipt()",
        ".open_exact_v2(",
        "inspect_untrusted_broker_financial_truth_bundle_v2(",
        "pub fn replay_sealed_quote_validated_research_v1(",
        "QuoteValidatedResearchAuthorityV1::UnverifiedCallerSuppliedQuotes",
        "QuoteValidatedResearchAuthorityV1::HistoricalBidAskQuotesOnly",
    ] {
        assert!(
            source.contains(required),
            "missing sealed-ingress invariant: {required}"
        );
    }
    for required_export in [
        "SealedHistoricalBidAskQuoteReplayEvidenceV1",
        "SealedHistoricalQuoteValidatedResearchLedgerV1",
        "open_sealed_historical_bid_ask_quote_replay_evidence_v1",
        "replay_sealed_quote_validated_research_v1",
    ] {
        assert!(
            lib.contains(required_export),
            "missing sealed replay export: {required_export}"
        );
    }
    let caller_supplied = function_body(source, "pub fn replay_quote_validated_research_v1(");
    assert!(
        caller_supplied
            .contains("QuoteValidatedResearchAuthorityV1::UnverifiedCallerSuppliedQuotes"),
        "caller-supplied quote vectors must be explicitly unverified"
    );
    assert!(
        !caller_supplied.contains("QuoteValidatedResearchAuthorityV1::HistoricalBidAskQuotesOnly"),
        "caller-supplied quote vectors must not mint historical broker authority"
    );
    let sealed = function_body(source, "pub fn replay_sealed_quote_validated_research_v1(");
    assert!(
        sealed.contains("QuoteValidatedResearchAuthorityV1::HistoricalBidAskQuotesOnly"),
        "only sealed replay may carry historical Bid/Ask authority"
    );
    for forbidden in [
        "BrokerFinancialTruthPermitV1",
        "BrokerFinancialTruthCapabilityV1",
        "install_broker_financial_truth",
        "current_broker_financial_truth",
        "OnceLock",
        "LazyLock",
        "std::env",
    ] {
        assert!(
            !source.contains(forbidden),
            "replay must not contain authority escape hatch {forbidden}"
        );
    }
}
