use std::path::PathBuf;

use neoethos_broker_truth::{
    BrokerFinancialOperationV1, BrokerFinancialTruthBundleReceiptV2,
    CanonicalBarSignalResearchDecisionV1, ClosedCanonicalBarTrailingThresholdV1,
    CompleteBidAskQuoteReplayEvidenceV1, CompleteQuoteSideCoverageV1, EvidenceWindowV1,
    ExactHistoricalQuoteV1, ExactQuoteSourceOrdinalV1, ExactZeroRowQuoteWindowProofV1,
    LockedFinalistOosReplayScopeV1, QuoteSideV1, QuoteValidatedResearchAuthorityV1,
    QuoteValidatedResearchExitReasonV1, QuoteValidatedResearchNonEntryReasonV1,
    QuoteValidatedResearchPromotionEligibilityV1, QuoteValidatedResearchReplayBindingV1,
    QuoteValidatedResearchReplayErrorCodeV1, QuoteValidatedResearchReplayPlanV1,
    QuoteValidatedResearchReplayPolicyV1, ResearchPositionDirectionV1,
    ReviewedQuoteReplayRuleIdentityV2, ReviewedSameTimestampMergeRuleV1,
    SameTimestampCrossSideOrderV1, VersionedLatencySlippagePolicyV1,
    current_broker_financial_truth_capability_v1, replay_quote_validated_research_v1,
};

const ACCOUNT_ID: i64 = 7;
const SYMBOL_ID: i64 = 42;
const SYMBOL_NAME: &str = "EURUSD";
const EVALUATED_WINDOW_FROM_MS: i64 = 1_700_000_000_000;
const EVALUATED_WINDOW_TO_MS: i64 = EVALUATED_WINDOW_FROM_MS + 60_000;
const SEED_PADDING_MS: i64 = 5_000;
const EXIT_PADDING_MS: i64 = 5_000;
const SIGNAL_BAR_OPEN_MS: i64 = EVALUATED_WINDOW_FROM_MS + 5_000;
const DECISION_AT_MS: i64 = EVALUATED_WINDOW_FROM_MS + 10_000;
const PIP_SIZE: f64 = 0.0001;
const SLIPPAGE_PIPS_PER_FILL: f64 = 0.5;
const LATENCY_SLIPPAGE_POLICY_VERSION: &str = "quote-validated-latency-slippage-v1";

const SEARCH_RECEIPT_SHA256: &str =
    "1111111111111111111111111111111111111111111111111111111111111111";
const CANONICAL_SIGNAL_PLAN_SHA256: &str =
    "2222222222222222222222222222222222222222222222222222222222222222";
const REVIEW_SHA256: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const PROTOCOL_SHA256: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const OBSERVATION_SHA256: &str = "5555555555555555555555555555555555555555555555555555555555555555";
const EVIDENCE_MANIFEST_SHA256: &str =
    "6666666666666666666666666666666666666666666666666666666666666666";
const BID_RAW_SHA256: &str = "7777777777777777777777777777777777777777777777777777777777777777";
const BID_DECODED_SHA256: &str = "8888888888888888888888888888888888888888888888888888888888888888";
const ASK_RAW_SHA256: &str = "9999999999999999999999999999999999999999999999999999999999999999";
const ASK_DECODED_SHA256: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const CHANGED_SHA256: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn evaluated_window() -> EvidenceWindowV1 {
    EvidenceWindowV1::new(EVALUATED_WINDOW_FROM_MS, EVALUATED_WINDOW_TO_MS)
        .expect("valid locked finalist/OOS window")
}

fn replay_scope() -> LockedFinalistOosReplayScopeV1 {
    LockedFinalistOosReplayScopeV1::new(evaluated_window(), SEED_PADDING_MS, EXIT_PADDING_MS)
        .expect("locked OOS window with seed and exit coverage padding")
}

fn required_quote_window() -> EvidenceWindowV1 {
    replay_scope().required_quote_coverage_window()
}

fn replay_rule() -> ReviewedQuoteReplayRuleIdentityV2 {
    ReviewedQuoteReplayRuleIdentityV2::new(REVIEW_SHA256, PROTOCOL_SHA256, OBSERVATION_SHA256)
        .expect("reviewed quote replay-rule identity")
}

fn binding() -> QuoteValidatedResearchReplayBindingV1 {
    QuoteValidatedResearchReplayBindingV1::new(
        SEARCH_RECEIPT_SHA256,
        CANONICAL_SIGNAL_PLAN_SHA256,
        ACCOUNT_ID,
        SYMBOL_ID,
        SYMBOL_NAME,
        replay_scope(),
        replay_rule(),
        EVIDENCE_MANIFEST_SHA256,
    )
    .expect("exact quote-validated research binding")
}

fn latency_slippage_policy() -> VersionedLatencySlippagePolicyV1 {
    VersionedLatencySlippagePolicyV1::new(
        LATENCY_SLIPPAGE_POLICY_VERSION,
        1, // first eligible entry-side quote is at decision_at + 1 ms
        0, // stop/target reference uses the causal triggering quote
        SLIPPAGE_PIPS_PER_FILL,
        PIP_SIZE,
    )
    .expect("explicit versioned latency/slippage assumption")
}

fn policy() -> QuoteValidatedResearchReplayPolicyV1 {
    policy_with_merge_rule(None)
}

fn policy_with_merge_rule(
    merge_rule: Option<ReviewedSameTimestampMergeRuleV1>,
) -> QuoteValidatedResearchReplayPolicyV1 {
    QuoteValidatedResearchReplayPolicyV1::new(
        2_000, // maximum wait for a market entry
        1_000, // maximum age of the opposite side used to form the book
        2_000, // maximum wait for a required market exit
        latency_slippage_policy(),
        merge_rule,
    )
    .expect("explicit replay timing policy")
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
    .expect("decision derived from adjacent canonical bar opens")
}

fn quote(timestamp_ms: i64, price: f64, row: u64) -> ExactHistoricalQuoteV1 {
    ExactHistoricalQuoteV1::new(
        timestamp_ms,
        price,
        ExactQuoteSourceOrdinalV1::new(0, 0, row).expect("exact quote source ordinal"),
    )
    .expect("valid exact historical quote")
}

fn side_coverage(
    side: QuoteSideV1,
    quotes: Vec<ExactHistoricalQuoteV1>,
) -> CompleteQuoteSideCoverageV1 {
    side_coverage_for(side, ACCOUNT_ID, SYMBOL_ID, required_quote_window(), quotes)
}

fn side_coverage_for(
    side: QuoteSideV1,
    account_id: i64,
    symbol_id: i64,
    evidence_window: EvidenceWindowV1,
    quotes: Vec<ExactHistoricalQuoteV1>,
) -> CompleteQuoteSideCoverageV1 {
    let (raw_sha256, decoded_sha256) = match side {
        QuoteSideV1::Bid => (BID_RAW_SHA256, BID_DECODED_SHA256),
        QuoteSideV1::Ask => (ASK_RAW_SHA256, ASK_DECODED_SHA256),
    };
    CompleteQuoteSideCoverageV1::new(
        side,
        account_id,
        symbol_id,
        evidence_window,
        raw_sha256,
        decoded_sha256,
        quotes,
        false,
    )
    .expect("complete exact quote-side coverage")
}

fn evidence(
    bid_quotes: Vec<ExactHistoricalQuoteV1>,
    ask_quotes: Vec<ExactHistoricalQuoteV1>,
) -> CompleteBidAskQuoteReplayEvidenceV1 {
    CompleteBidAskQuoteReplayEvidenceV1::new(
        binding(),
        side_coverage(QuoteSideV1::Bid, bid_quotes),
        side_coverage(QuoteSideV1::Ask, ask_quotes),
    )
    .expect("complete synchronized historical quote evidence")
}

fn plan(decision: CanonicalBarSignalResearchDecisionV1) -> QuoteValidatedResearchReplayPlanV1 {
    plan_with_policy(decision, policy())
}

fn plan_with_policy(
    decision: CanonicalBarSignalResearchDecisionV1,
    replay_policy: QuoteValidatedResearchReplayPolicyV1,
) -> QuoteValidatedResearchReplayPlanV1 {
    QuoteValidatedResearchReplayPlanV1::new(binding(), replay_policy, vec![decision], Vec::new())
        .expect("quote-validated research replay plan")
}

#[test]
fn decision_time_is_the_next_canonical_bar_open_and_predecision_quotes_never_form_an_entry() {
    let research_decision = decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0100);
    assert_eq!(
        research_decision.signal_bar_open_unix_ms(),
        SIGNAL_BAR_OPEN_MS
    );
    assert_eq!(
        research_decision.next_canonical_bar_open_unix_ms(),
        DECISION_AT_MS
    );
    assert_eq!(research_decision.decision_at_unix_ms(), DECISION_AT_MS);

    let quotes = evidence(
        vec![
            quote(DECISION_AT_MS - 100, 0.9998, 0),
            quote(DECISION_AT_MS + 3, 1.0100, 1),
        ],
        vec![
            quote(DECISION_AT_MS - 1, 0.9999, 0),
            quote(DECISION_AT_MS, 1.0001, 1),
            quote(DECISION_AT_MS + 1, 1.0002, 2),
        ],
    );
    let ledger = replay_quote_validated_research_v1(&plan(research_decision), quotes)
        .expect("quote-validated research replay");
    let position = ledger.positions().first().expect("one modeled position");

    assert_eq!(
        position.entry_reference().timestamp_unix_ms(),
        DECISION_AT_MS + 1
    );
    assert_eq!(position.entry_reference().price(), 1.0002);
}

#[test]
fn long_uses_an_ask_entry_reference_and_a_bid_exit_reference() {
    let quotes = evidence(
        vec![
            quote(DECISION_AT_MS - 100, 0.9998, 0),
            quote(DECISION_AT_MS + 4, 1.0010, 1),
        ],
        vec![quote(DECISION_AT_MS + 1, 1.0000, 0)],
    );
    let ledger = replay_quote_validated_research_v1(
        &plan(decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0010)),
        quotes,
    )
    .expect("long quote-validated replay");
    let position = ledger.positions().first().expect("one long position");

    assert_eq!(position.entry_reference().side(), QuoteSideV1::Ask);
    assert_eq!(position.entry_reference().price(), 1.0000);
    assert_eq!(
        position.exit_reference().expect("closed long").side(),
        QuoteSideV1::Bid
    );
    assert_eq!(
        position.exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::Target)
    );
}

#[test]
fn short_uses_a_bid_entry_reference_and_an_ask_exit_reference() {
    let quotes = evidence(
        vec![quote(DECISION_AT_MS + 1, 1.0000, 0)],
        vec![
            quote(DECISION_AT_MS - 100, 1.0002, 0),
            quote(DECISION_AT_MS + 4, 0.9990, 1),
        ],
    );
    let ledger = replay_quote_validated_research_v1(
        &plan(decision(ResearchPositionDirectionV1::Short, 1.0100, 0.9990)),
        quotes,
    )
    .expect("short quote-validated replay");
    let position = ledger.positions().first().expect("one short position");

    assert_eq!(position.entry_reference().side(), QuoteSideV1::Bid);
    assert_eq!(position.entry_reference().price(), 1.0000);
    assert_eq!(
        position.exit_reference().expect("closed short").side(),
        QuoteSideV1::Ask
    );
    assert_eq!(
        position.exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::Target)
    );
}

#[test]
fn entry_wait_and_book_staleness_are_explicit_typed_research_non_entries() {
    let late_executable_side = evidence(
        vec![quote(DECISION_AT_MS - 100, 0.9998, 0)],
        vec![quote(DECISION_AT_MS + 2_001, 1.0000, 0)],
    );
    let late_ledger = replay_quote_validated_research_v1(
        &plan(decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0100)),
        late_executable_side,
    )
    .expect("complete coverage proves a legitimate unavailable research entry");
    assert!(late_ledger.positions().is_empty());
    assert_eq!(
        late_ledger.entry_unavailable()[0].reason(),
        QuoteValidatedResearchNonEntryReasonV1::NoEligibleQuoteWithinEntryWait
    );
    assert_eq!(
        late_ledger.entry_unavailable()[0].deadline_unix_ms(),
        DECISION_AT_MS + 2_000
    );

    let stale_book = evidence(
        vec![quote(DECISION_AT_MS - 1_001, 0.9998, 0)],
        vec![quote(DECISION_AT_MS + 10, 1.0000, 0)],
    );
    let stale_ledger = replay_quote_validated_research_v1(
        &plan(decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0100)),
        stale_book,
    )
    .expect("stale but complete book is typed research unavailability, not missing evidence");
    assert_eq!(
        stale_ledger.entry_unavailable()[0].reason(),
        QuoteValidatedResearchNonEntryReasonV1::StaleSynchronizedBook
    );
}

#[test]
fn historical_quotes_are_research_references_not_observed_broker_fills() {
    let assumptions = latency_slippage_policy();
    assert_eq!(
        assumptions.policy_version(),
        LATENCY_SLIPPAGE_POLICY_VERSION
    );
    assert_eq!(assumptions.entry_latency_ms(), 1);
    assert_eq!(assumptions.exit_latency_ms(), 0);
    assert_eq!(assumptions.slippage_pips_per_fill(), SLIPPAGE_PIPS_PER_FILL);

    let ledger = replay_quote_validated_research_v1(
        &plan(decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0010)),
        evidence(
            vec![
                quote(DECISION_AT_MS - 100, 0.9998, 0),
                quote(DECISION_AT_MS + 2, 1.0010, 1),
            ],
            vec![quote(DECISION_AT_MS + 1, 1.0002, 0)],
        ),
    )
    .expect("historical quotes validate only causal reference prices");
    let position = &ledger.positions()[0];

    assert_eq!(
        ledger.authority(),
        QuoteValidatedResearchAuthorityV1::UnverifiedCallerSuppliedQuotes
    );
    assert_eq!(position.entry_reference().price(), 1.0002);
    assert_eq!(
        position.exit_reference().expect("target reference").price(),
        1.0010
    );
    assert!((position.modeled_entry_price() - 1.00025).abs() < 1e-12);
    assert!((position.modeled_exit_price().expect("modeled exit") - 1.00095).abs() < 1e-12);
    assert!((position.slippage_pips_charged() - 1.0).abs() < f64::EPSILON);
    assert_eq!(position.additional_spread_pips_charged(), 0.0);
    assert_eq!(
        ledger.promotion_eligibility(),
        QuoteValidatedResearchPromotionEligibilityV1::NotPromotionEligible
    );
}

#[test]
fn complete_locked_oos_side_coverage_includes_seed_and_exit_padding() {
    assert_eq!(
        replay_scope().locked_evaluation_window(),
        evaluated_window()
    );
    assert_eq!(
        required_quote_window(),
        EvidenceWindowV1::new(
            EVALUATED_WINDOW_FROM_MS - SEED_PADDING_MS,
            EVALUATED_WINDOW_TO_MS + EXIT_PADDING_MS,
        )
        .expect("padded quote window")
    );

    let sparse_ask = side_coverage_for(
        QuoteSideV1::Ask,
        ACCOUNT_ID,
        SYMBOL_ID,
        evaluated_window(),
        vec![quote(DECISION_AT_MS + 1, 1.0000, 0)],
    );
    let error = CompleteBidAskQuoteReplayEvidenceV1::new(
        binding(),
        side_coverage(
            QuoteSideV1::Bid,
            vec![quote(DECISION_AT_MS - 100, 0.9998, 0)],
        ),
        sparse_ask,
    )
    .expect_err("precomputed sparse trade windows cannot replace full locked OOS coverage");
    assert_eq!(
        error.code(),
        QuoteValidatedResearchReplayErrorCodeV1::RequiredCoverageWindowMismatch
    );
}

#[test]
fn same_timestamp_cross_side_outcome_ambiguity_requires_a_reviewed_merge_rule() {
    let ambiguous_quotes = || {
        evidence(
            vec![
                quote(DECISION_AT_MS - 100, 0.9998, 0),
                quote(DECISION_AT_MS + 1, 0.9900, 1),
            ],
            vec![quote(DECISION_AT_MS + 1, 1.0000, 0)],
        )
    };
    let decision = || decision(ResearchPositionDirectionV1::Long, 0.9900, 1.1000);

    let error = replay_quote_validated_research_v1(&plan(decision()), ambiguous_quotes())
        .expect_err("an outcome-changing cross-side tie has no implicit ordering");
    assert_eq!(
        error.code(),
        QuoteValidatedResearchReplayErrorCodeV1::AmbiguousSameTimestampCrossSideOutcome
    );

    let reviewed_merge_rule = ReviewedSameTimestampMergeRuleV1::new(
        replay_rule(),
        SameTimestampCrossSideOrderV1::AskBeforeBid,
    )
    .expect("review-bound Ask-before-Bid rule");
    let resolved_policy = policy_with_merge_rule(Some(reviewed_merge_rule));
    let ledger = replay_quote_validated_research_v1(
        &plan_with_policy(decision(), resolved_policy),
        ambiguous_quotes(),
    )
    .expect("the bound reviewed rule resolves the same-timestamp outcome");
    assert_eq!(
        ledger.positions()[0].exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::Stop)
    );
}

#[test]
fn quote_order_resolves_a_same_ohlc_stop_target_ambiguity() {
    let bar_low = 0.9800;
    let bar_high = 1.0200;
    assert!(bar_low <= 0.9900 && bar_high >= 1.0100);

    let target_first = evidence(
        vec![
            quote(DECISION_AT_MS - 100, 0.9998, 0),
            quote(DECISION_AT_MS + 2, 1.0100, 1),
            quote(DECISION_AT_MS + 3, 0.9900, 2),
        ],
        vec![quote(DECISION_AT_MS + 1, 1.0000, 0)],
    );
    let target_ledger = replay_quote_validated_research_v1(
        &plan(decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0100)),
        target_first,
    )
    .expect("quote order, not OHLC precedence, resolves target first");
    assert_eq!(
        target_ledger.positions()[0].exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::Target)
    );

    let stop_first = evidence(
        vec![
            quote(DECISION_AT_MS - 100, 0.9998, 0),
            quote(DECISION_AT_MS + 2, 0.9900, 1),
            quote(DECISION_AT_MS + 3, 1.0100, 2),
        ],
        vec![quote(DECISION_AT_MS + 1, 1.0000, 0)],
    );
    let stop_ledger = replay_quote_validated_research_v1(
        &plan(decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0100)),
        stop_first,
    )
    .expect("quote order, not OHLC precedence, resolves stop first");
    assert_eq!(
        stop_ledger.positions()[0].exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::Stop)
    );
}

#[test]
fn trailing_threshold_becomes_effective_only_after_its_source_bar_closed() {
    let next_bar_open_ms = DECISION_AT_MS + 10_000;
    let trailing = ClosedCanonicalBarTrailingThresholdV1::new(
        DECISION_AT_MS,
        next_bar_open_ms,
        ResearchPositionDirectionV1::Long,
        1.0002,
    )
    .expect("closed-bar-only trailing threshold");
    let replay_plan = QuoteValidatedResearchReplayPlanV1::new(
        binding(),
        policy(),
        vec![decision(ResearchPositionDirectionV1::Long, 0.9500, 1.1000)],
        vec![trailing],
    )
    .expect("plan with causal trailing schedule");
    let quotes = evidence(
        vec![
            quote(DECISION_AT_MS - 100, 0.9998, 0),
            quote(DECISION_AT_MS + 5_000, 0.9999, 1),
            quote(next_bar_open_ms + 1, 1.0001, 2),
        ],
        vec![quote(DECISION_AT_MS + 1, 1.0000, 0)],
    );

    let ledger = replay_quote_validated_research_v1(&replay_plan, quotes)
        .expect("quotes cannot ratchet a bar-owned trailing threshold early");
    let position = &ledger.positions()[0];
    assert_eq!(
        position
            .exit_reference()
            .expect("trailing exit")
            .timestamp_unix_ms(),
        next_bar_open_ms + 1
    );
    assert_eq!(
        position.exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::TrailingStop)
    );
}

#[test]
fn missing_incomplete_or_tampered_coverage_is_an_evidence_error_not_a_non_entry() {
    let missing = CompleteQuoteSideCoverageV1::new(
        QuoteSideV1::Ask,
        ACCOUNT_ID,
        SYMBOL_ID,
        required_quote_window(),
        ASK_RAW_SHA256,
        ASK_DECODED_SHA256,
        Vec::new(),
        false,
    )
    .expect_err("an empty side needs a specific zero-row proof");
    assert_eq!(
        missing.code(),
        QuoteValidatedResearchReplayErrorCodeV1::MissingZeroRowWindowProof
    );

    let incomplete = CompleteQuoteSideCoverageV1::new(
        QuoteSideV1::Ask,
        ACCOUNT_ID,
        SYMBOL_ID,
        required_quote_window(),
        ASK_RAW_SHA256,
        ASK_DECODED_SHA256,
        vec![quote(DECISION_AT_MS + 1, 1.0000, 0)],
        true,
    )
    .expect_err("terminal hasMore=true is incomplete evidence");
    assert_eq!(
        incomplete.code(),
        QuoteValidatedResearchReplayErrorCodeV1::IncompleteCoverage
    );

    let exact = evidence(
        vec![quote(DECISION_AT_MS, 0.9998, 0)],
        vec![quote(DECISION_AT_MS, 1.0000, 0)],
    );
    let encoded = String::from_utf8(
        exact
            .canonical_json_bytes()
            .expect("canonical evidence JSON"),
    )
    .expect("UTF-8 evidence JSON");
    let changed = encoded.replacen(BID_RAW_SHA256, CHANGED_SHA256, 1);
    let tampered = CompleteBidAskQuoteReplayEvidenceV1::from_json_bytes(changed.as_bytes())
        .expect_err("a changed artifact digest cannot inherit the sealed manifest binding");
    assert_eq!(
        tampered.code(),
        QuoteValidatedResearchReplayErrorCodeV1::ArtifactDigestMismatch
    );
}

#[test]
fn replay_receipt_binds_search_account_symbol_window_rule_policy_and_ledger() {
    for mismatched_ask in [
        side_coverage_for(
            QuoteSideV1::Ask,
            ACCOUNT_ID + 1,
            SYMBOL_ID,
            required_quote_window(),
            vec![quote(DECISION_AT_MS, 1.0000, 0)],
        ),
        side_coverage_for(
            QuoteSideV1::Ask,
            ACCOUNT_ID,
            SYMBOL_ID + 1,
            required_quote_window(),
            vec![quote(DECISION_AT_MS, 1.0000, 0)],
        ),
        side_coverage_for(
            QuoteSideV1::Ask,
            ACCOUNT_ID,
            SYMBOL_ID,
            EvidenceWindowV1::new(
                EVALUATED_WINDOW_FROM_MS - SEED_PADDING_MS,
                EVALUATED_WINDOW_TO_MS + EXIT_PADDING_MS + 1,
            )
            .expect("different valid evidence window"),
            vec![quote(DECISION_AT_MS, 1.0000, 0)],
        ),
    ] {
        let error = CompleteBidAskQuoteReplayEvidenceV1::new(
            binding(),
            side_coverage(QuoteSideV1::Bid, vec![quote(DECISION_AT_MS, 0.9998, 0)]),
            mismatched_ask,
        )
        .expect_err("account, symbol, and window are exact evidence bindings");
        assert_eq!(
            error.code(),
            QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch
        );
    }

    let quotes = evidence(
        vec![
            quote(DECISION_AT_MS - 100, 0.9998, 0),
            quote(DECISION_AT_MS + 2, 1.0100, 1),
        ],
        vec![quote(DECISION_AT_MS + 1, 1.0000, 0)],
    );
    let ledger = replay_quote_validated_research_v1(
        &plan(decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0100)),
        quotes,
    )
    .expect("receipt-bound exact replay");
    let receipt = ledger.receipt();

    assert_eq!(
        receipt.canonical_search_input_receipt_sha256(),
        SEARCH_RECEIPT_SHA256
    );
    assert_eq!(
        receipt.canonical_signal_plan_sha256(),
        CANONICAL_SIGNAL_PLAN_SHA256
    );
    assert_eq!(receipt.account_id(), ACCOUNT_ID);
    assert_eq!(receipt.symbol_id(), SYMBOL_ID);
    assert_eq!(receipt.symbol_name(), SYMBOL_NAME);
    assert_eq!(receipt.locked_evaluation_window(), evaluated_window());
    assert_eq!(
        receipt.required_quote_coverage_window(),
        required_quote_window()
    );
    assert_eq!(receipt.seed_padding_ms(), SEED_PADDING_MS);
    assert_eq!(receipt.exit_padding_ms(), EXIT_PADDING_MS);
    assert_eq!(
        receipt.reviewed_replay_rule_identity_sha256(),
        replay_rule().identity_sha256()
    );
    assert_eq!(
        receipt.quote_evidence_manifest_sha256(),
        EVIDENCE_MANIFEST_SHA256
    );
    assert_eq!(
        receipt.latency_slippage_policy_sha256(),
        latency_slippage_policy().identity_sha256()
    );
    assert_eq!(receipt.replay_policy_sha256(), policy().identity_sha256());
    assert_eq!(receipt.ledger_sha256(), ledger.ledger_sha256());
    assert_eq!(
        ledger.promotion_eligibility(),
        QuoteValidatedResearchPromotionEligibilityV1::NotPromotionEligible
    );
    assert_eq!(
        ledger.authority(),
        QuoteValidatedResearchAuthorityV1::UnverifiedCallerSuppliedQuotes
    );

    let different_rule =
        ReviewedQuoteReplayRuleIdentityV2::new(CHANGED_SHA256, PROTOCOL_SHA256, OBSERVATION_SHA256)
            .expect("different reviewed replay-rule identity");
    let different_binding = QuoteValidatedResearchReplayBindingV1::new(
        SEARCH_RECEIPT_SHA256,
        CANONICAL_SIGNAL_PLAN_SHA256,
        ACCOUNT_ID,
        SYMBOL_ID,
        SYMBOL_NAME,
        replay_scope(),
        different_rule,
        EVIDENCE_MANIFEST_SHA256,
    )
    .expect("different exact quote-validated research binding");
    let different_plan = QuoteValidatedResearchReplayPlanV1::new(
        different_binding,
        policy(),
        vec![decision(ResearchPositionDirectionV1::Long, 0.9900, 1.0100)],
        Vec::new(),
    )
    .expect("well-formed but differently bound replay plan");
    let error = replay_quote_validated_research_v1(
        &different_plan,
        evidence(
            vec![quote(DECISION_AT_MS, 0.9998, 0)],
            vec![quote(DECISION_AT_MS, 1.0000, 0)],
        ),
    )
    .expect_err("the reviewed replay rule is part of the exact replay binding");
    assert_eq!(
        error.code(),
        QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch
    );
}

#[test]
fn a_zero_row_quote_window_requires_an_explicit_terminal_broker_proof() {
    let proof = ExactZeroRowQuoteWindowProofV1::new(
        QuoteSideV1::Ask,
        ACCOUNT_ID,
        SYMBOL_ID,
        required_quote_window(),
        ASK_RAW_SHA256,
        ASK_DECODED_SHA256,
        0,
        false,
    )
    .expect("explicit empty terminal response proof");
    let empty = CompleteQuoteSideCoverageV1::empty(proof).expect("proven zero-row side coverage");
    assert_eq!(empty.event_count(), 0);
    assert!(empty.is_explicit_zero_row_window());

    let incomplete = ExactZeroRowQuoteWindowProofV1::new(
        QuoteSideV1::Ask,
        ACCOUNT_ID,
        SYMBOL_ID,
        required_quote_window(),
        ASK_RAW_SHA256,
        ASK_DECODED_SHA256,
        0,
        true,
    )
    .expect_err("an empty response with hasMore=true proves no terminal coverage");
    assert_eq!(
        incomplete.code(),
        QuoteValidatedResearchReplayErrorCodeV1::IncompleteCoverage
    );
}

#[test]
fn quote_validated_research_surface_has_no_ohlc_scalar_raw_iterator_fill_or_permit_escape_hatch() {
    let module_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("execution_replay_v1.rs");
    let source = std::fs::read_to_string(&module_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", module_path.display()));

    for forbidden in [
        "BacktestSettings",
        "configured_spread_pips",
        "session_spread",
        "spread_cost",
        "apply_spread",
        "charge_spread",
        "simulate_trades_core",
        "fast_evaluate_strategy_core",
        "resample",
        "Resample",
        "Ohlc",
        "OHLC",
        "bar_high",
        "bar_low",
        "tick_signal",
        "BrokerFinancialTruthCapabilityV1",
        "BrokerFinancialTruthPermitV1",
        "BrokerFinancialTruthBundleReceiptV2",
        "ObservedBrokerFill",
        "ActualBrokerFill",
        "observed_broker_fill",
        "actual_broker_fill",
        "deal_id",
        "pub fn fills(",
        "pub fn raw_quotes(",
        "pub fn quotes(",
        "pub fn bid_quotes(",
        "pub fn ask_quotes(",
        "pub fn quote_events(",
        "pub fn decoded_quotes(",
        "pub fn events(",
        "pub fn rows(",
        "pub fn permit(",
        "pub fn capability(",
        "pub fn iter(",
        "impl IntoIterator",
        "impl std::ops::Deref",
    ] {
        assert!(
            !source.contains(forbidden),
            "quote-validated research replay exposes forbidden fallback/raw/fill/capability surface `{forbidden}`"
        );
    }
}

#[test]
fn a_legacy_v2_bundle_receipt_alone_has_no_promotion_authority() {
    let manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    let legacy_json = format!(
        "{{\"bundle_id\":\"bft2-{manifest_sha256}\",\"manifest_sha256\":\"{manifest_sha256}\"}}"
    );
    let legacy = BrokerFinancialTruthBundleReceiptV2::from_json_bytes(legacy_json.as_bytes())
        .expect("well-formed legacy V2 receipt");
    assert_eq!(legacy.manifest_sha256(), manifest_sha256);

    let error = current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::Promotion)
        .expect_err("legacy V2 storage integrity alone must not promote");
    assert_eq!(error.operation(), BrokerFinancialOperationV1::Promotion);
}
