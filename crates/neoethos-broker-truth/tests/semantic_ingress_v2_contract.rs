use neoethos_broker_truth::{
    BrokerFinancialOperationV1, BrokerFinancialTruthAuthoritySourceClassV2,
    BrokerFinancialTruthEvidenceClassV2, BrokerFinancialTruthSemanticIngressErrorCodeV2,
    BrokerTruthAcquisitionPromotionEligibilityV1, BrokerTruthAcquisitionSemanticStatusV1,
    CanonicalBarSignalResearchDecisionV1, ClosedCanonicalBarTimeExitV1,
    ClosedCanonicalBarTrailingThresholdV1, EvidenceWindowV1, ExecutionSymbolContractV1,
    LockedFinalistOosReplayScopeV1, QuoteValidatedExecutionEconomicsLedgerV1,
    QuoteValidatedResearchExitReasonV1, QuoteValidatedResearchReplayBindingV1,
    QuoteValidatedResearchReplayErrorCodeV1, QuoteValidatedResearchReplayPlanV1,
    QuoteValidatedResearchReplayPolicyV1, ResearchPositionDirectionV1,
    ReviewedQuoteReplayRuleIdentityV2, SealedHistoricalBidAskQuoteReplayEvidenceV1,
    SealedHistoricalQuoteValidatedResearchLedgerV1, VersionedLatencySlippagePolicyV1,
    current_broker_financial_truth_capability_v1,
    inspect_untrusted_broker_financial_truth_bundle_v2,
    into_sealed_historical_bid_ask_quote_replay_evidence_v2,
    open_sealed_historical_bid_ask_quote_replay_evidence_v1,
    preview_sealed_quote_validated_research_entry_v1,
    replay_sealed_quote_validated_decision_sequence_v1, replay_sealed_quote_validated_research_v1,
    validate_reviewed_broker_financial_truth_authority_v2,
};
use serde_json::{Value, json};
use std::fs;

#[path = "support/semantic_fixture.rs"]
mod semantic_fixture;
use semantic_fixture::*;

#[test]
fn structurally_consistent_synthetic_bundle_remains_explicitly_untrusted() {
    let (_root, verified, _authority) = fixture(Tamper::None);
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
        let (_root, verified, _authority) = fixture(tamper);
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
        let (_root, verified, _authority) = fixture(tamper);
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
fn exact_reviewed_fixture_mints_only_a_move_only_run_authority() {
    let (_root, _verified, fixture) = fixture(Tamper::None);
    let authority = validate_reviewed_broker_financial_truth_authority_v2(
        &fixture.store,
        &fixture.link_receipt,
        fixture.reviewed,
    )
    .expect("exact reviewed semantic fixture mints run-scoped authority");
    assert_eq!(authority.reviewed_synchronization_count(), 1);
    assert_eq!(
        authority.source_artifact_class(),
        BrokerFinancialTruthAuthoritySourceClassV2::ResearchOnly
    );
    assert_eq!(
        authority.source_semantic_status(),
        BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly
    );
    assert_eq!(
        authority.source_promotion_eligibility(),
        BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible
    );
    for class in [
        BrokerFinancialTruthEvidenceClassV2::PrimaryBidAsk,
        BrokerFinancialTruthEvidenceClassV2::ConversionLegs,
        BrokerFinancialTruthEvidenceClassV2::ExactSymbolAndAccountContracts,
        BrokerFinancialTruthEvidenceClassV2::UnrealizedPnl,
        BrokerFinancialTruthEvidenceClassV2::CloseDealReconciliation,
    ] {
        let digest = authority.evidence_class_binding_sha256(class);
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("run authority must not mutate the global V1 gate");
}

#[test]
fn reviewed_identity_never_overrides_raw_decoded_semantic_failure() {
    for tamper in [
        Tamper::TickRawDecodedMismatch,
        Tamper::GenericRawDecodedLinkMismatch,
        Tamper::DealPageMismatch,
    ] {
        let (_root, _verified, fixture) = fixture(tamper);
        validate_reviewed_broker_financial_truth_authority_v2(
            &fixture.store,
            &fixture.link_receipt,
            fixture.reviewed,
        )
        .expect_err("review metadata cannot override a semantic class mismatch");
    }
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("failed reviewed fixtures must leave the global V1 gate closed");
}

#[test]
fn reviewed_quote_transfer_rejects_every_changed_replay_binding() {
    for changed_field in [
        "receipt",
        "account",
        "symbol_id",
        "symbol_name",
        "window",
        "review",
        "manifest",
    ] {
        let (_root, verified, fixture) = fixture(Tamper::None);
        let authority = validate_reviewed_broker_financial_truth_authority_v2(
            &fixture.store,
            &fixture.link_receipt,
            fixture.reviewed,
        )
        .expect("independently checked fixture authority");
        let review = if changed_field == "review" {
            ReviewedQuoteReplayRuleIdentityV2::new(
                sha256(b"different independent review"),
                sha256(PROTOCOL_EVIDENCE_BYTES),
                verified
                    .manifest()
                    .primary_quotes()
                    .replay_rule()
                    .identity()
                    .broker_observation_sha256(),
            )
            .expect("well-formed but unrelated replay rule")
        } else {
            verified
                .manifest()
                .primary_quotes()
                .replay_rule()
                .identity()
                .clone()
        };
        let scope = LockedFinalistOosReplayScopeV1::new(
            EvidenceWindowV1::new(
                WINDOW_FROM + 1_000 + i64::from(changed_field == "window"),
                WINDOW_TO - 1_000,
            )
            .expect("well-formed locked window"),
            1_000,
            1_000,
        )
        .expect("well-formed padded window");
        let binding = QuoteValidatedResearchReplayBindingV1::new(
            if changed_field == "receipt" {
                sha256(b"another canonical receipt")
            } else {
                sha256(CANONICAL_RUN_BYTES)
            },
            sha256(b"fixed closed-quote signal fixture"),
            ACCOUNT_ID + i64::from(changed_field == "account"),
            SYMBOL_ID + i64::from(changed_field == "symbol_id"),
            if changed_field == "symbol_name" {
                "GBPUSD"
            } else {
                "EURUSD"
            },
            scope,
            review,
            if changed_field == "manifest" {
                sha256(b"another BFT2 manifest")
            } else {
                verified.receipt().manifest_sha256().to_owned()
            },
        )
        .expect("self-consistent alternate binding, not merely a broken checksum");
        let error = into_sealed_historical_bid_ask_quote_replay_evidence_v2(authority, &binding)
            .expect_err("a reviewed authority must not transfer to another replay binding");
        assert_eq!(
            error.code(),
            neoethos_broker_truth::QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch,
            "changed {changed_field}",
        );
    }
}

#[test]
fn reviewed_quote_transfer_uses_the_verified_snapshot_not_later_file_bytes() {
    let (root, verified, fixture) = fixture_with_ticks(
        Tamper::None,
        &[
            (TICK_TIMESTAMP - 1, 124_000),
            (TICK_TIMESTAMP + 1_000, 112_500),
        ],
        &[(TICK_TIMESTAMP, 125_000)],
    );
    let binding = closed_quote_replay_binding(&verified);
    let authority = validate_reviewed_broker_financial_truth_authority_v2(
        &fixture.store,
        &fixture.link_receipt,
        fixture.reviewed,
    )
    .expect("read and verify the complete original Vortex snapshot");
    let quote_path =
        verified.artifact_path(verified.manifest().primary_quotes().ask().decoded_ticks());
    assert!(
        quote_path.starts_with(&root.0),
        "only this test's generated fixture may change"
    );
    fs::write(
        &quote_path,
        b"changed after the reviewed snapshot was fully decoded",
    )
    .expect("corrupt only a generated temporary fixture");
    open_sealed_historical_bid_ask_quote_replay_evidence_v1(
        &fixture.store,
        &fixture.link_receipt,
        &binding,
    )
    .expect_err("the old reopen route must refuse the changed on-disk object");
    let evidence = into_sealed_historical_bid_ask_quote_replay_evidence_v2(authority, &binding)
        .expect("move the previously verified owned records without reopening the changed file");
    let plan = closed_quote_replay_plan(binding);
    let ledger = replay_sealed_quote_validated_research_v1(&plan, &evidence)
        .expect("the consumer replays the original verified prices");
    assert_eq!(
        ledger,
        replay_sealed_quote_validated_research_v1(&plan, &evidence)
            .expect("the same immutable snapshot remains reusable after on-disk tampering")
    );
    assert_eq!(ledger.positions()[0].modeled_entry_price(), 1.25);
    assert_eq!(ledger.positions()[0].modeled_exit_price(), Some(1.125));
    assert_eq!(ledger.positions()[0].additional_spread_pips_charged(), 0.0);
    assert_eq!(
        ledger.promotion_eligibility(),
        neoethos_broker_truth::QuoteValidatedResearchPromotionEligibilityV1::NotPromotionEligible,
    );
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("snapshot transfer does not open the global V1 gate");
}

#[test]
fn sealed_quote_execution_economics_replays_actual_vortex_fills() {
    let (_root, quotes, economics) = closed_quote_economics_fixture();
    let position = &quotes.positions()[0];
    assert_eq!(position.modeled_entry_price(), 1.25);
    assert_eq!(position.modeled_exit_price(), Some(1.125));
    assert_eq!(economics.base_units(), 100_000.0);
    // Independent exact-binary oracle: -(1/8) * 100,000 - 7 - 7.
    assert_eq!(economics.gross_pnl_account_currency().amount(), -12_500.0);
    assert_eq!(economics.net_pnl_account_currency().amount(), -12_514.0);
    economics
        .validate_against_quote_ledger(&quotes)
        .expect("economics built from these exact sealed fills must validate");
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("synthetic execution evidence must not open the global financial gate");
}

#[test]
fn sealed_quote_execution_economics_rejects_rehashed_detached_fills() {
    let (_root, quotes, economics) = closed_quote_economics_fixture();
    let original = serde_json::to_value(&economics).expect("execution fixture JSON");
    let mut accepted = Vec::new();
    for changed in [
        "position_index",
        "symbol",
        "direction",
        "entry_price",
        "exit_price",
        "entry_timestamp",
        "exit_timestamp",
    ] {
        let mut wire = original.clone();
        match changed {
            "position_index" => wire["quote_position_index"] = json!(1),
            "symbol" => {
                wire["symbol_contract"] = serde_json::to_value(
                    ExecutionSymbolContractV1::new("GBPUSD", "GBP", "USD", 100_000.0)
                        .expect("different internally valid symbol contract"),
                )
                .expect("different contract JSON");
            }
            "direction" => wire["direction"] = json!("short"),
            "entry_price" => {
                wire["modeled_entry_price"] = json!(1.0);
                wire["entry_notional_quote_currency"] = json!(100_000.0);
            }
            "exit_price" => wire["modeled_exit_price"] = json!(1.375),
            "entry_timestamp" => wire["entry_fill_timestamp_unix_ms"] = json!(TICK_TIMESTAMP - 1),
            "exit_timestamp" => wire["exit_fill_timestamp_unix_ms"] = json!(TICK_TIMESTAMP + 1_001),
            _ => unreachable!("finite fixture mutation set"),
        }
        if matches!(changed, "direction" | "entry_price" | "exit_price") {
            wire["gross_pnl_quote_currency"] = json!(12_500.0);
            wire["gross_pnl_account_currency"]["amount"] = json!(12_500.0);
            wire["net_pnl_account_currency"]["amount"] = json!(12_486.0);
        }
        rehash_untrusted_execution_fixture(&mut wire);
        let detached = QuoteValidatedExecutionEconomicsLedgerV1::from_json_bytes(
            &serde_json::to_vec(&wire).expect("encode detached fixture"),
        )
        .expect("forgery is deliberately internally consistent, not a stale-hash failure");
        assert_eq!(detached.quote_ledger_sha256(), quotes.ledger_sha256());
        if matches!(changed, "direction" | "entry_price" | "exit_price") {
            assert_eq!(detached.net_pnl_account_currency().amount(), 12_486.0);
        }
        if detached.validate_against_quote_ledger(&quotes).is_ok() {
            accepted.push(changed);
        }
    }
    assert!(
        accepted.is_empty(),
        "sealed loss -12514 was accepted with detached fields: {accepted:?}"
    );
}

#[test]
fn one_reviewed_snapshot_supports_concurrent_replays_of_a_long_short_sequence() {
    let (_root, binding, evidence) = sealed_quote_sequence_fixture(
        &[
            (TICK_TIMESTAMP - 1, 124_000),
            (TICK_TIMESTAMP + 1_000, 112_500),
            (TICK_TIMESTAMP + 2_000, 125_000),
        ],
        &[
            (TICK_TIMESTAMP, 125_000),
            (TICK_TIMESTAMP + 1_999, 126_000),
            (TICK_TIMESTAMP + 3_000, 112_500),
        ],
    );
    let plans = vec![
        closed_quote_replay_plan(binding.clone()),
        quote_replay_plan_at(
            binding,
            TICK_TIMESTAMP + 2_000,
            ResearchPositionDirectionV1::Short,
            1.5,
            1.2,
            Vec::new(),
        ),
    ];
    // Concurrent replays of these same bound inputs share one snapshot.
    // Time remains ordered inside each replay; this is not a Search scheduler.
    let (first_run, second_run) = std::thread::scope(|scope| {
        let other = scope.spawn(|| {
            replay_sealed_quote_validated_decision_sequence_v1(&plans, &evidence)
                .expect("second lane uses the same immutable quote snapshot")
        });
        let first = replay_sealed_quote_validated_decision_sequence_v1(&plans, &evidence)
            .expect("long loss then short gain, each closing before the next decision");
        (first, other.join().expect("bounded replay lane thread"))
    });
    assert_eq!(
        first_run, second_run,
        "all outcomes and receipt hashes agree"
    );
    assert_eq!(first_run.len(), 2);
    assert_ne!(first_run[0].ledger_sha256(), first_run[1].ledger_sha256());
    for ledger in &first_run {
        assert_eq!(ledger.positions().len(), 1);
        assert!(ledger.entry_unavailable().is_empty());
        assert_eq!(ledger.positions()[0].modeled_entry_price(), 1.25);
        assert_eq!(ledger.positions()[0].modeled_exit_price(), Some(1.125));
        assert_eq!(ledger.positions()[0].additional_spread_pips_charged(), 0.0);
        assert_eq!(
            ledger.promotion_eligibility(),
            neoethos_broker_truth::QuoteValidatedResearchPromotionEligibilityV1::NotPromotionEligible
        );
    }
    let loss = fixture_execution_economics(&first_run[0]);
    let gain = fixture_execution_economics(&first_run[1]);
    // Independent exact-binary oracle: +/- (1/8 * 100,000), USD 7 per fill.
    assert_eq!(loss.net_pnl_account_currency().amount(), -12_514.0);
    assert_eq!(gain.net_pnl_account_currency().amount(), 12_486.0);
    assert_eq!(
        loss.net_pnl_account_currency().amount() + gain.net_pnl_account_currency().amount(),
        -28.0
    );
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("multi-decision replay never opens the global V1 gate");
}

#[test]
fn replay_sequence_rejects_overlapping_or_same_time_position_transitions() {
    let (_root, binding, evidence) = sealed_quote_sequence_fixture(
        &[
            (TICK_TIMESTAMP - 1, 124_000),
            (TICK_TIMESTAMP + 1_000, 112_500),
        ],
        &[(TICK_TIMESTAMP, 125_000)],
    );
    for offset in [500, 1_000] {
        let plans = vec![
            closed_quote_replay_plan(binding.clone()),
            quote_replay_plan_at(
                binding.clone(),
                TICK_TIMESTAMP + offset,
                ResearchPositionDirectionV1::Long,
                1.2,
                1.5,
                Vec::new(),
            ),
        ];
        let error = replay_sealed_quote_validated_decision_sequence_v1(&plans, &evidence)
            .expect_err("a prior close must strictly precede the next decision");
        assert_eq!(
            error.code(),
            QuoteValidatedResearchReplayErrorCodeV1::OverlappingDecisionWindow
        );
    }
}

#[test]
fn replay_sequence_retains_open_terminal_position_and_refuses_another_entry() {
    let (_root, binding, evidence) = sealed_quote_sequence_fixture(
        &[(TICK_TIMESTAMP - 1, 124_000)],
        &[(TICK_TIMESTAMP, 125_000)],
    );
    let first = closed_quote_replay_plan(binding.clone());
    let single =
        replay_sealed_quote_validated_decision_sequence_v1(std::slice::from_ref(&first), &evidence)
            .expect("a final open position is explicit, never a fabricated close");
    assert!(single[0].positions()[0].exit_reference().is_none());
    let next = quote_replay_plan_at(
        binding,
        TICK_TIMESTAMP + 2_000,
        ResearchPositionDirectionV1::Long,
        1.2,
        1.5,
        Vec::new(),
    );
    let error = replay_sealed_quote_validated_decision_sequence_v1(&[first, next], &evidence)
        .expect_err("an open prior position cannot be forgotten by a later decision");
    assert_eq!(
        error.code(),
        QuoteValidatedResearchReplayErrorCodeV1::OpenPositionBeforeNextDecision
    );
}

#[test]
fn replay_sequence_waits_for_non_entry_deadline_before_next_decision() {
    let (_root, binding, evidence) = sealed_quote_sequence_fixture(
        &[
            (TICK_TIMESTAMP + 1_999, 124_000),
            (TICK_TIMESTAMP + 3_000, 150_000),
        ],
        &[(TICK_TIMESTAMP + 2_000, 125_000)],
    );
    let first = closed_quote_replay_plan(binding.clone());
    let next = |offset| {
        quote_replay_plan_at(
            binding.clone(),
            TICK_TIMESTAMP + offset,
            ResearchPositionDirectionV1::Long,
            1.2,
            1.5,
            Vec::new(),
        )
    };
    for offset in [499, 500] {
        let error = replay_sealed_quote_validated_decision_sequence_v1(
            &[first.clone(), next(offset)],
            &evidence,
        )
        .expect_err("a pending entry still owns its inclusive wait deadline");
        assert_eq!(
            error.code(),
            QuoteValidatedResearchReplayErrorCodeV1::OverlappingDecisionWindow
        );
    }
    let ledgers =
        replay_sealed_quote_validated_decision_sequence_v1(&[first, next(2_000)], &evidence)
            .expect("expired entry followed by a later executable decision");
    assert!(ledgers[0].positions().is_empty());
    assert_eq!(
        ledgers[0].entry_unavailable()[0].deadline_unix_ms(),
        TICK_TIMESTAMP + 500
    );
    assert_eq!(ledgers[1].positions()[0].modeled_exit_price(), Some(1.5));
}

#[test]
fn each_sequence_decision_owns_its_profit_protecting_trailing_schedule() {
    let (_root, binding, evidence) = sealed_quote_sequence_fixture(
        &[
            (TICK_TIMESTAMP - 1, 124_000),
            (TICK_TIMESTAMP + 1_000, 137_500),
            (TICK_TIMESTAMP + 1_999, 124_000),
            (TICK_TIMESTAMP + 2_001, 124_000),
            (TICK_TIMESTAMP + 3_000, 150_000),
        ],
        &[(TICK_TIMESTAMP, 125_000), (TICK_TIMESTAMP + 2_000, 125_000)],
    );
    let trail = ClosedCanonicalBarTrailingThresholdV1::new(
        TICK_TIMESTAMP,
        TICK_TIMESTAMP + 500,
        ResearchPositionDirectionV1::Long,
        1.375,
    )
    .expect("closed-bar trailing fixture");
    let plans = [
        quote_replay_plan_at(
            binding.clone(),
            TICK_TIMESTAMP,
            ResearchPositionDirectionV1::Long,
            1.2,
            1.5,
            vec![trail],
        ),
        quote_replay_plan_at(
            binding,
            TICK_TIMESTAMP + 2_000,
            ResearchPositionDirectionV1::Long,
            1.2,
            1.5,
            Vec::new(),
        ),
    ];
    let ledgers = replay_sealed_quote_validated_decision_sequence_v1(&plans, &evidence)
        .expect("trailing protection belongs only to the first position");
    assert_eq!(ledgers[0].positions()[0].modeled_exit_price(), Some(1.375));
    assert_eq!(
        ledgers[0].positions()[0].exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::TrailingStop)
    );
    assert_eq!(ledgers[1].positions()[0].modeled_exit_price(), Some(1.5));
    assert_eq!(
        ledgers[1].positions()[0].exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::Target)
    );
}

#[test]
fn reviewed_snapshot_entry_preview_and_time_exit_feed_existing_execution_economics() {
    let (_root, binding, evidence) = sealed_quote_sequence_fixture(
        &[
            (TICK_TIMESTAMP - 2, 148_000),
            (TICK_TIMESTAMP - 1, 124_000),
            (TICK_TIMESTAMP + 250, 126_000),
            (TICK_TIMESTAMP + 500, 131_250),
            (TICK_TIMESTAMP + 1_000, 112_500),
        ],
        &[(TICK_TIMESTAMP, 125_000)],
    );
    let plan = closed_quote_replay_plan(binding);
    let entry = preview_sealed_quote_validated_research_entry_v1(&plan, &evidence)
        .expect("borrow the already reviewed quote snapshot")
        .expect("fixture entry is available");
    assert_eq!(entry.timestamp_unix_ms(), TICK_TIMESTAMP);
    assert_eq!(entry.modeled_entry_price(), 1.25);
    assert_eq!(
        entry.bid_extrema_before(TICK_TIMESTAMP + 500).unwrap(),
        (1.26, 1.24),
        "neither the pre-entry high nor the close-boundary tick belongs in the entry-bar trail"
    );
    let timed = plan
        .with_time_exit(
            ClosedCanonicalBarTimeExitV1::new(TICK_TIMESTAMP, TICK_TIMESTAMP + 500).unwrap(),
        )
        .unwrap();
    let quotes = replay_sealed_quote_validated_research_v1(&timed, &evidence).unwrap();
    assert_eq!(
        quotes.positions()[0].exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::MaxHold)
    );
    assert_eq!(quotes.positions()[0].modeled_exit_price(), Some(1.3125));
    let economics = fixture_execution_economics(&quotes);
    // Exact binary oracle: (1/16 * 100,000) - USD 7 per fill. The
    // commission/conversion assumptions are synthetic, not certified costs.
    assert_eq!(economics.gross_pnl_account_currency().amount(), 6_250.0);
    assert_eq!(economics.net_pnl_account_currency().amount(), 6_236.0);
    economics.validate_against_quote_ledger(&quotes).unwrap();
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("this synthetic Vortex proof does not open the global finance gate");
}

#[test]
fn sealed_snapshot_checks_exact_context_even_before_any_directional_decision() {
    let (_root, binding, evidence) = sealed_quote_sequence_fixture(
        &[(TICK_TIMESTAMP - 1, 124_000)],
        &[(TICK_TIMESTAMP, 125_000)],
    );
    let plan = closed_quote_replay_plan(binding.clone());
    let policy =
        serde_json::from_value(serde_json::to_value(&plan).unwrap()["policy"].clone()).unwrap();
    evidence
        .validate_replay_context_v1(&binding, &policy)
        .unwrap();
    let wire = serde_json::to_value(&binding).unwrap();
    let changed = QuoteValidatedResearchReplayBindingV1::new(
        sha256(CANONICAL_RUN_BYTES),
        sha256(b"different locked signal plan"),
        ACCOUNT_ID,
        SYMBOL_ID,
        "EURUSD",
        binding.replay_scope(),
        serde_json::from_value(wire["reviewed_replay_rule"].clone()).unwrap(),
        wire["quote_evidence_manifest_sha256"].as_str().unwrap(),
    )
    .unwrap();
    let error = evidence
        .validate_replay_context_v1(&changed, &policy)
        .expect_err("even an empty/no-direction lane cannot use another signal plan's snapshot");
    assert_eq!(
        error.code(),
        QuoteValidatedResearchReplayErrorCodeV1::BindingMismatch
    );
}

#[test]
fn sealed_kernel_keeps_actual_decision_identity_even_when_different_plans_fill_identically() {
    let (_root, binding, evidence) = sealed_quote_sequence_fixture(
        &[
            (TICK_TIMESTAMP - 1, 124_000),
            (TICK_TIMESTAMP + 500, 131_250),
        ],
        &[(TICK_TIMESTAMP, 125_000)],
    );
    let timed = closed_quote_replay_plan(binding)
        .with_time_exit(
            ClosedCanonicalBarTimeExitV1::new(TICK_TIMESTAMP, TICK_TIMESTAMP + 500).unwrap(),
        )
        .unwrap();
    let mut changed_wire = serde_json::to_value(&timed).unwrap();
    changed_wire["decisions"][0]["stop_price"] = serde_json::json!(0.75);
    let changed: QuoteValidatedResearchReplayPlanV1 = serde_json::from_value(changed_wire).unwrap();
    let first = replay_sealed_quote_validated_research_v1(&timed, &evidence).unwrap();
    let second = replay_sealed_quote_validated_research_v1(&changed, &evidence).unwrap();
    assert_eq!(first.positions(), second.positions());
    assert_eq!(
        first.ledger_sha256(),
        second.ledger_sha256(),
        "legacy V1 fill identity stays compatible"
    );
    assert_ne!(first.executed_plan_sha256(), second.executed_plan_sha256());
    assert_ne!(
        first.executed_decision().stop_price(),
        second.executed_decision().stop_price()
    );
    assert_eq!(second.executed_decision().stop_price(), 0.75);
    assert_eq!(first.executed_time_exit(), second.executed_time_exit());
    assert!(first.executed_time_exit().is_some());
    let wire = serde_json::to_value(&first).unwrap();
    assert_eq!(wire.as_object().unwrap().len(), 1);
    assert!(
        wire.get("ledger").is_some(),
        "constant-space process witnesses do not change V1 wire"
    );
}

fn sealed_quote_sequence_fixture(
    bid: &[(i64, i64)],
    ask: &[(i64, i64)],
) -> (
    FixtureRoot,
    QuoteValidatedResearchReplayBindingV1,
    SealedHistoricalBidAskQuoteReplayEvidenceV1,
) {
    let (root, verified, fixture) = fixture_with_ticks(Tamper::None, bid, ask);
    let binding = closed_quote_replay_binding(&verified);
    let authority = validate_reviewed_broker_financial_truth_authority_v2(
        &fixture.store,
        &fixture.link_receipt,
        fixture.reviewed,
    )
    .expect("independent synthetic V2 review over actual temporary Vortex files");
    let evidence = into_sealed_historical_bid_ask_quote_replay_evidence_v2(authority, &binding)
        .expect("one transfer of the reviewed quote snapshot");
    (root, binding, evidence)
}

fn closed_quote_economics_fixture() -> (
    FixtureRoot,
    SealedHistoricalQuoteValidatedResearchLedgerV1,
    QuoteValidatedExecutionEconomicsLedgerV1,
) {
    // Tiny synthetic ticks in genuine Vortex files, not claimed broker observations.
    let (root, verified, authority_fixture) = fixture_with_ticks(
        Tamper::None,
        &[
            (TICK_TIMESTAMP - 1, 124_000),
            (TICK_TIMESTAMP + 1_000, 112_500),
        ],
        &[(TICK_TIMESTAMP, 125_000)],
    );
    let run_authority = validate_reviewed_broker_financial_truth_authority_v2(
        &authority_fixture.store,
        &authority_fixture.link_receipt,
        authority_fixture.reviewed,
    )
    .expect("synthetic reviewed authority remains run-scoped");
    let binding = closed_quote_replay_binding(&verified);
    let reopened_evidence = open_sealed_historical_bid_ask_quote_replay_evidence_v1(
        &authority_fixture.store,
        &authority_fixture.link_receipt,
        &binding,
    )
    .expect("reopen and decode real Vortex files through production sealed ingress");
    let evidence = into_sealed_historical_bid_ask_quote_replay_evidence_v2(run_authority, &binding)
        .expect("consume reviewed V2 records through the production replay bridge");
    let plan = closed_quote_replay_plan(binding);
    let quotes = replay_sealed_quote_validated_research_v1(&plan, &evidence)
        .expect("production sealed replay closes the losing long on Bid");
    let reopened_quotes = replay_sealed_quote_validated_research_v1(&plan, &reopened_evidence)
        .expect("old sealed ingress remains the exact comparison baseline");
    assert_eq!(
        quotes, reopened_quotes,
        "V2 transfer preserves every fill and receipt digest"
    );
    let economics = fixture_execution_economics(&quotes);
    (root, quotes, economics)
}

fn closed_quote_replay_binding(
    verified: &neoethos_broker_truth::VerifiedImmutableBrokerFinancialTruthBundleV2,
) -> QuoteValidatedResearchReplayBindingV1 {
    let replay_scope = LockedFinalistOosReplayScopeV1::new(
        EvidenceWindowV1::new(WINDOW_FROM + 1_000, WINDOW_TO - 1_000).expect("locked window"),
        1_000,
        1_000,
    )
    .expect("exact padded quote window");
    QuoteValidatedResearchReplayBindingV1::new(
        sha256(CANONICAL_RUN_BYTES),
        sha256(b"fixed closed-quote signal fixture"),
        ACCOUNT_ID,
        SYMBOL_ID,
        "EURUSD",
        replay_scope,
        verified
            .manifest()
            .primary_quotes()
            .replay_rule()
            .identity()
            .clone(),
        verified.receipt().manifest_sha256(),
    )
    .expect("exact replay fixture binding")
}

fn closed_quote_replay_plan(
    binding: QuoteValidatedResearchReplayBindingV1,
) -> QuoteValidatedResearchReplayPlanV1 {
    quote_replay_plan_at(
        binding,
        TICK_TIMESTAMP,
        ResearchPositionDirectionV1::Long,
        1.2,
        1.5,
        Vec::new(),
    )
}

fn quote_replay_plan_at(
    binding: QuoteValidatedResearchReplayBindingV1,
    decision_at: i64,
    direction: ResearchPositionDirectionV1,
    stop: f64,
    target: f64,
    thresholds: Vec<ClosedCanonicalBarTrailingThresholdV1>,
) -> QuoteValidatedResearchReplayPlanV1 {
    let policy = QuoteValidatedResearchReplayPolicyV1::new(
        500,
        1_000,
        500,
        VersionedLatencySlippagePolicyV1::new("synthetic-fill-binding-v1", 0, 0, 0.0, 0.0001)
            .expect("explicit fixture slippage assumptions"),
        None,
    )
    .expect("causal fixture quote policy");
    QuoteValidatedResearchReplayPlanV1::new(
        binding,
        policy,
        vec![
            CanonicalBarSignalResearchDecisionV1::new(
                decision_at - 10_000,
                decision_at,
                direction,
                stop,
                target,
            )
            .expect("fixture decision"),
        ],
        thresholds,
    )
    .expect("one-decision fixture replay")
}

// Deliberately recreate the public wire digest: hashes prove byte identity, not
// agreement with the independently held opaque quote ledger. This makes every
// negative fixture self-consistent rather than merely corrupting its checksum.
fn rehash_untrusted_execution_fixture(wire: &mut Value) {
    for role in ["entry", "exit"] {
        let timestamp_key = format!("{role}_fill_timestamp_unix_ms");
        let price_key = format!("modeled_{role}_price");
        let identity = ordered_fixture_hash(
            "quote-validated-derived-fill",
            &[
                ("quote_ledger_sha256", &wire["quote_ledger_sha256"]),
                ("quote_position_index", &wire["quote_position_index"]),
                ("fill_role", &json!(role)),
                ("fill_timestamp_unix_ms", &wire[&timestamp_key]),
                ("modeled_fill_price", &wire[&price_key]),
                ("direction", &wire["direction"]),
            ],
        );
        wire[format!("{role}_fill_identity_sha256")] = json!(identity);
    }
    let identity = ordered_fixture_hash(
        "quote-validated-execution-economics-ledger",
        &[
            ("schema_version", &wire["schema_version"]),
            ("quote_ledger_sha256", &wire["quote_ledger_sha256"]),
            ("quote_position_index", &wire["quote_position_index"]),
            (
                "symbol_contract_identity_sha256",
                &wire["symbol_contract"]["symbol_contract_identity_sha256"],
            ),
            ("account_currency", &wire["account_currency"]),
            (
                "conversion_evidence_identity_sha256",
                &wire["conversion"]["conversion_evidence_identity_sha256"],
            ),
            (
                "entry_fill_identity_sha256",
                &wire["entry_fill_identity_sha256"],
            ),
            (
                "exit_fill_identity_sha256",
                &wire["exit_fill_identity_sha256"],
            ),
            (
                "commission_policy_identity_sha256",
                &wire["commission_policy"]["commission_policy_identity_sha256"],
            ),
            (
                "swap_evidence_identity_sha256",
                &wire["swap"]["swap_evidence_identity_sha256"],
            ),
            (
                "pnl_conversion_fee_evidence_identity_sha256",
                &wire["pnl_conversion_fee"]["pnl_conversion_fee_evidence_identity_sha256"],
            ),
            ("filled_lots", &wire["filled_lots"]),
            ("base_units", &wire["base_units"]),
            ("direction", &wire["direction"]),
            (
                "entry_fill_timestamp_unix_ms",
                &wire["entry_fill_timestamp_unix_ms"],
            ),
            (
                "exit_fill_timestamp_unix_ms",
                &wire["exit_fill_timestamp_unix_ms"],
            ),
            ("modeled_entry_price", &wire["modeled_entry_price"]),
            ("modeled_exit_price", &wire["modeled_exit_price"]),
            (
                "entry_notional_quote_currency",
                &wire["entry_notional_quote_currency"],
            ),
            (
                "gross_pnl_quote_currency",
                &wire["gross_pnl_quote_currency"],
            ),
            (
                "conversion_rate_account_per_quote",
                &wire["conversion"]["conversion_rate_account_per_quote"],
            ),
            (
                "conversion_observed_at_unix_ms",
                &wire["conversion"]["conversion_observed_at_unix_ms"],
            ),
            (
                "gross_pnl_account_currency",
                &wire["gross_pnl_account_currency"],
            ),
            (
                "entry_commission_account_currency",
                &wire["entry_commission_account_currency"],
            ),
            (
                "exit_commission_account_currency",
                &wire["exit_commission_account_currency"],
            ),
            (
                "swap_account_currency_signed",
                &wire["swap_account_currency_signed"],
            ),
            (
                "pnl_conversion_fee_account_currency",
                &wire["pnl_conversion_fee_account_currency"],
            ),
            (
                "additional_spread_account_currency",
                &wire["additional_spread_account_currency"],
            ),
            (
                "net_pnl_account_currency",
                &wire["net_pnl_account_currency"],
            ),
            ("artifact_class", &wire["artifact_class"]),
            ("promotion_eligibility", &wire["promotion_eligibility"]),
        ],
    );
    wire["ledger_sha256"] = json!(identity);
}

fn ordered_fixture_hash(domain: &str, fields: &[(&str, &Value)]) -> String {
    let mut json = String::from("{");
    for (index, (key, value)) in fields.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        json.push_str(&serde_json::to_string(key).expect("fixture key encoding"));
        json.push(':');
        if value.get("currency").is_some() && value.get("amount").is_some() {
            // AccountMoneyV1 has this struct-field order, unlike a JSON map.
            json.push_str(&format!(
                "{{\"currency\":{},\"amount\":{}}}",
                value["currency"], value["amount"]
            ));
        } else {
            json.push_str(&serde_json::to_string(value).expect("fixture value encoding"));
        }
    }
    json.push('}');
    sha256(format!("neoethos-{domain}-v1\n{json}").as_bytes())
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

fn batched_ticks() -> (Vec<(i64, i64)>, Vec<(i64, i64)>) {
    let bid = (0..20_000)
        .map(|index| {
            (
                WINDOW_FROM + index * 2,
                if index == 19_999 { 112_500 } else { 124_000 },
            )
        })
        .collect();
    let ask = (0..20_000)
        .map(|index| (WINDOW_FROM + index * 2 + 1, 125_000))
        .collect();
    (bid, ask)
}

#[test]
fn batched_vortex_ingress_preserves_all_pages_and_a_fill_beyond_the_first_decode_batch() {
    let (bid, ask) = batched_ticks();
    let (_root, verified, fixture) = fixture_with_ticks(Tamper::None, &bid, &ask);
    assert_eq!(
        verified
            .manifest()
            .primary_quotes()
            .bid()
            .raw_pages()
            .row_count(),
        3
    );
    let binding = closed_quote_replay_binding(&verified);
    let authority = validate_reviewed_broker_financial_truth_authority_v2(
        &fixture.store,
        &fixture.link_receipt,
        fixture.reviewed,
    )
    .expect("every Vortex batch and exact newest-first page must validate");
    let evidence =
        into_sealed_historical_bid_ask_quote_replay_evidence_v2(authority, &binding).unwrap();
    let ledger =
        replay_sealed_quote_validated_research_v1(&closed_quote_replay_plan(binding), &evidence)
            .unwrap();
    assert_eq!(ledger.positions().len(), 1);
    assert_eq!(ledger.positions()[0].modeled_entry_price(), 1.25);
    assert_eq!(ledger.positions()[0].modeled_exit_price(), Some(1.125));
    assert_eq!(
        ledger.positions()[0]
            .exit_reference()
            .unwrap()
            .timestamp_unix_ms(),
        WINDOW_FROM + 39_998
    );
    assert_eq!(
        fixture_execution_economics(&ledger)
            .net_pnl_account_currency()
            .amount(),
        -12_514.0
    );
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("large synthetic replay is still not global financial authority");
}

#[test]
fn batched_vortex_ingress_refuses_changed_page_ordinals_and_corruption_after_row_16384() {
    let (bid, ask) = batched_ticks();
    for tamper in [
        Tamper::TickPageOrdinalMismatch,
        Tamper::TickFinalRowMismatch,
    ] {
        let (_root, verified, _fixture) = fixture_with_ticks(tamper, &bid, &ask);
        let error = inspect_untrusted_broker_financial_truth_bundle_v2(verified)
            .expect_err("hash-valid files with altered ticks or page identity must be refused");
        assert_eq!(
            error.code(),
            BrokerFinancialTruthSemanticIngressErrorCodeV2::RawDecodedMismatch
        );
    }
}
