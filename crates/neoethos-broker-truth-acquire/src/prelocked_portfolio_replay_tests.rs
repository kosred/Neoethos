//! Actual stored Vortex -> independent review -> leased signal replay -> money
//! consumer. Every market observation and cost here is an explicit synthetic
//! test fixture, not observed broker performance or promotion authority.

use super::*;
use neoethos_broker_truth::{
    AccountMoneyV1, BrokerFinancialOperationV1, QuoteValidatedResearchExitReasonV1,
    ResearchPositionDirectionV1, SealedHistoricalBidAskQuoteReplayEvidenceV1,
    current_broker_financial_truth_capability_v1, replay_sealed_quote_validated_research_v1,
};
use neoethos_core::execution::{BudgetedCpuExecutor, BudgetedCpuScope};
use neoethos_core::execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalTimeframe,
    FeatureCellValidity, FeatureColumnF64, FeatureFrame, Ohlcv,
};
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOutputV1, FeaturePlanV1,
    SourceArtifactBindingV1, SourceSegmentV1,
};
use neoethos_search::{
    CanonicalSearchInputReceiptV2, CanonicalSignalAccountRiskPolicyV3, CanonicalSignalExitPolicyV2,
    Gene, LockedCanonicalSignalPlanV3, LockedPortfolioOuterHoldoutReplaySetV3,
    QuoteEntryFinancialInputsV3, QuoteEntrySizingEvidenceV3, QuoteReplayLotConstraintsV3,
    QuoteValidatedOuterHoldoutArtifactClassV1, QuoteValidatedOuterHoldoutErrorCodeV1,
    QuoteValidatedOuterHoldoutPromotionEligibilityV1, evaluate_locked_portfolio_outer_holdout_v3,
};
use neoethos_trader::data_replay::replay_locked_canonical_signal_portfolio_v3;

// Share the same real Vortex writer/review fixture as semantic-ingress tests.
// Its older corruption cases/default window are used by that suite only.
#[allow(dead_code)]
mod semantic_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../neoethos-broker-truth/tests/support/semantic_fixture.rs"
    ));
}

// Use actual in-range Unix milliseconds on a canonical minute boundary.
// Small relative offsets keep the entry/exit ordering readable below.
const fn at(offset_ms: i64) -> i64 {
    1_700_000_040_000 + offset_ms
}

struct SignalFixture {
    bars: Ohlcv,
    scope: CanonicalSearchArtifactScopeV2,
    genes: Vec<Gene>,
    signals: Vec<Vec<i8>>,
    confidences: Vec<Vec<f64>>,
    size_multipliers: Vec<Vec<f64>>,
    config_hash: String,
}

impl SignalFixture {
    fn new() -> Self {
        // Binary-exact prices make the expected cash result independent of
        // decimal-rounding tolerances. The pip quantum is test-only.
        let bars = Ohlcv {
            timestamp: Some([60_000, 120_000, 180_000, 240_000].map(at).to_vec()),
            open: vec![1.0, 2.0, 1.375, 1.375],
            high: vec![1.0, 2.0, 1.375, 1.375],
            low: vec![1.0, 1.125, 1.375, 1.125],
            close: vec![1.0, 1.375, 1.375, 1.125],
            volume: None,
        };
        let identity = CanonicalDatasetIdentity::ctrader(
            CTraderEnvironment::Demo,
            "demo.ctraderapi.com",
            semantic_fixture::ACCOUNT_ID,
            semantic_fixture::SYMBOL_ID,
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .unwrap();
        let plan = FeaturePlanV1::new(
            vec![
                FeatureNodeV1::source(
                    "test:source",
                    identity.clone(),
                    "test.acquired-prelocked-replay.v2",
                    1,
                    vec![FeatureOutputV1::f64("test_signal_input", 1).unwrap()],
                    [1; 32],
                )
                .unwrap(),
            ],
            vec!["test_signal_input".to_owned()],
        )
        .unwrap();
        let provenance = DatasetFeatureArtifactProvenanceV1::new(
            &plan,
            vec![
                SourceArtifactBindingV1::new(
                    "test:source",
                    identity.clone(),
                    "test.acquired-prelocked-replay.v2",
                    [1; 32],
                    "test-only-generation",
                    [2; 32],
                    BarTimestampConvention::BarOpen,
                    vec![SourceSegmentV1::new(0, 4, at(60_000), at(240_000)).unwrap()],
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let features = FeatureFrame::from_columns(
            bars.timestamp.clone().unwrap(),
            vec![
                FeatureColumnF64::new(
                    "test_signal_input",
                    vec![0.0; 4],
                    vec![FeatureCellValidity::Valid; 4],
                )
                .unwrap(),
            ],
            plan,
            provenance,
        )
        .unwrap();
        let receipt =
            CanonicalSearchInputReceiptV2::from_feature_frame(&identity, &features).unwrap();
        let scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
            CanonicalSearchWindowRoleV1::Holdout,
            receipt,
        )
        .unwrap();
        let gene = |name: &str, target| Gene {
            strategy_id: name.to_owned(),
            sl_pips: 2.0,
            tp_pips: target,
            stop_vol_mult: 0.0,
            ..Gene::default()
        };
        Self {
            bars,
            scope,
            genes: vec![
                gene("fixture-long", 8.0),
                gene("fixture-short", 1.0),
                gene("fixture-flat", 4.0),
            ],
            signals: vec![vec![1, 0, 0, 0], vec![0, -1, 0, 0], vec![0; 4]],
            confidences: vec![vec![0.5, 0.0, 0.0, 0.0], vec![0.0; 4], vec![0.0; 4]],
            size_multipliers: vec![vec![1.0; 4], vec![0.5; 4], vec![1.0; 4]],
            // Match the canonical tagged identity produced by Search rather
            // than hiding the cross-crate format boundary behind a SHA fixture.
            config_hash: "fnv64:0123456789abcdef".to_owned(),
        }
    }

    fn lock(&self) -> LockedCanonicalSignalPlanV3<'_> {
        LockedCanonicalSignalPlanV3::new(
            &self.genes,
            &neoethos_search::canonical_locked_portfolio_identity_sha256_v1(&self.genes).unwrap(),
            &self.signals,
            &self.confidences,
            &self.size_multipliers,
            &self.bars,
            &self.scope,
            &self.config_hash,
            CanonicalSignalExitPolicyV2 {
                pip_size: 0.125,
                max_hold_bars: 3,
                trailing_enabled: true,
                trailing_stop_multiplier: 1.0,
                trailing_be_trigger_r: 1.0,
                trailing_min_lock_pips: 0.0,
            },
            CanonicalSignalAccountRiskPolicyV3::new(
                AccountMoneyV1::new("USD", 100_000.0).unwrap(),
                0.125,
                0.375,
                1.0,
            )
            .unwrap(),
            &neoethos_search::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(
                neoethos_search::stop_target::StopTargetSettings {
                    vol_estimator: "parkinson".into(),
                    ..Default::default()
                },
                true,
                2.0,
            )
            .unwrap(),
        )
        .unwrap()
    }
}

struct StoredFixture {
    authority: semantic_fixture::AuthorityFixture,
    capture: FinalistQuoteReplayAcquisitionOutcomeV1,
    // Drop the temporary files after the store and all replay work end.
    root: semantic_fixture::FixtureRoot,
}

impl StoredFixture {
    fn new(locked: &LockedCanonicalSignalPlanV3<'_>) -> Self {
        let scope =
            LockedFinalistOosReplayScopeV1::new(locked.locked_evaluation_window(), 1_000, 1_000)
                .unwrap();
        let receipt_bytes = locked.holdout_scope().receipt().to_json_bytes().unwrap();
        let scope_bytes = locked.holdout_scope().to_json_bytes().unwrap();
        // Use actual typed JSON and its domain-separated identity. The raw
        // file SHA stored in the artifact descriptor is deliberately distinct.
        assert_ne!(
            semantic_fixture::sha256(&receipt_bytes),
            locked.holdout_scope().receipt_sha256()
        );
        let (root, verified, authority) = semantic_fixture::fixture_with_ticks_and_scope(
            semantic_fixture::Tamper::None,
            &[
                (59_000, 100_000),
                (120_000, 200_000), // Pre-entry high must NOT arm the long trail.
                (120_001, 112_500),
                (140_001, 162_500),
                (179_998, 137_500),
                (180_000, 137_500),
                (260_000, 112_500),
                (300_000, 112_500),
            ]
            .map(|(timestamp, price)| (at(timestamp), price)),
            &[
                (59_001, 212_500),
                (119_999, 212_500),
                (120_002, 125_000),
                (140_000, 175_000),
                (179_999, 150_000),
                (180_001, 150_000),
                (260_001, 125_000),
                (300_001, 125_000),
            ]
            .map(|(timestamp, price)| (at(timestamp), price)),
            semantic_fixture::FixtureScope {
                window: scope.required_quote_coverage_window(),
                canonical_run_identity_sha256: locked.holdout_scope().receipt_sha256(),
                canonical_scope_identity_sha256: locked.scope_identity_sha256(),
                canonical_run_bytes: &receipt_bytes,
                canonical_scope_bytes: &scope_bytes,
            },
        );
        let link = authority.store.open_link(&authority.link_receipt).unwrap();
        let replay_binding = QuoteValidatedResearchReplayBindingV1::new(
            locked.holdout_scope().receipt_sha256(),
            locked.identity_sha256(),
            semantic_fixture::ACCOUNT_ID,
            semantic_fixture::SYMBOL_ID,
            locked.symbol_name(),
            scope,
            verified
                .manifest()
                .primary_quotes()
                .replay_rule()
                .identity()
                .clone(),
            verified.receipt().manifest_sha256(),
        )
        .unwrap();
        let replay_policy = QuoteValidatedResearchReplayPolicyV1::new(
            1_000,
            1_000,
            1_000,
            VersionedLatencySlippagePolicyV1::new(
                "test-only-binary-exact-quotes",
                0,
                0,
                0.0,
                0.125,
            )
            .unwrap(),
            None,
        )
        .unwrap();
        locked
            .validate_replay_binding(&replay_binding, &replay_policy)
            .unwrap();
        let capture = FinalistQuoteReplayAcquisitionOutcomeV1 {
            authority_receipt: link.manifest().authority_receipt().clone(),
            broker_truth_receipt: verified.receipt().clone(),
            acquisition_link_receipt: authority.link_receipt.clone(),
            replay_binding,
            replay_policy,
            artifact_class: FinalistQuoteReplayArtifactClassV1::ResearchOnly,
            semantic_status: BrokerTruthSemanticStatusV1::UnvalidatedEvidenceOnly,
            promotion_eligibility: BrokerTruthPromotionEligibilityV1::NotPromotionEligible,
            portfolio_identity_sha256: locked.portfolio_identity_sha256().to_owned(),
            search_config_hash: locked.search_config_hash().to_owned(),
            holdout_scope_identity_sha256: locked.scope_identity_sha256().to_owned(),
        };
        Self {
            root,
            authority,
            capture,
        }
    }
}

fn in_leased_pool<R: Send>(work: impl FnOnce(&BudgetedCpuScope<'_>) -> R + Send) -> R {
    let width = WorkerLimit::new(3).unwrap();
    let broker = CpuPermitBroker::new(width);
    let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), width);
    let lease = broker.acquire(CpuPermitRequest::local(width)).unwrap();
    let result = executor
        .execute_with_scope(lease.into_transfer(), work)
        .unwrap();
    assert_eq!(broker.snapshot().live_reserved_sum, 0);
    result
}

fn entry_financial_inputs(
    ledger: &SealedHistoricalQuoteValidatedResearchLedgerV1,
    _: &neoethos_search::QuoteValidatedDecisionProvenanceV3,
) -> anyhow::Result<QuoteEntryFinancialInputsV3> {
    use neoethos_broker_truth::{CausalQuoteToAccountConversionV1, ExecutionSymbolContractV1};
    let source =
        semantic_fixture::sha256(b"explicit synthetic entry economics, not broker authority");
    Ok(QuoteEntryFinancialInputsV3 {
        symbol_contract: ExecutionSymbolContractV1::new("EURUSD", "EUR", "USD", 100_000.0)?,
        entry_conversion: CausalQuoteToAccountConversionV1::new(
            "USD",
            "USD",
            1.0,
            ledger.positions()[0].entry_reference().timestamp_unix_ms(),
            1_000,
            &source,
        )?,
        lot_constraints: QuoteReplayLotConstraintsV3::new(0.125, 100.0, 0.125, source)?,
    })
}

fn sized_economics(
    ledger: &SealedHistoricalQuoteValidatedResearchLedgerV1,
    sizing: &QuoteEntrySizingEvidenceV3,
) -> anyhow::Result<neoethos_broker_truth::QuoteValidatedExecutionEconomicsLedgerV1> {
    use neoethos_broker_truth::{
        CausalQuoteToAccountConversionV1, ExecutionCommissionPolicyV1, ExecutionSymbolContractV1,
        PnlConversionFeeV1, SignedSwapCashflowV1, build_quote_validated_execution_economics_v1,
    };
    let source =
        semantic_fixture::sha256(b"synthetic economics assumptions, not real broker authority");
    let closed_at = ledger.positions()[0]
        .exit_reference()
        .unwrap()
        .timestamp_unix_ms();
    Ok(build_quote_validated_execution_economics_v1(
        ledger,
        0,
        &ExecutionSymbolContractV1::new("EURUSD", "EUR", "USD", 100_000.0)?,
        "USD",
        sizing.filled_lots,
        Some(&CausalQuoteToAccountConversionV1::new(
            "USD", "USD", 1.0, closed_at, 1_000, &source,
        )?),
        &ExecutionCommissionPolicyV1::new("USD", 7.0, &source)?,
        &SignedSwapCashflowV1::new("USD", 0.0, closed_at, &source)?,
        &PnlConversionFeeV1::new("USD", 0.0, &source)?,
    )?)
}

#[test]
fn stored_vortex_replays_three_prelocked_lanes_into_exact_closed_cash_results() {
    let signal_fixture = SignalFixture::new();
    let locked = signal_fixture.lock();
    let stored = StoredFixture::new(&locked);
    let identity_before = locked.identity_sha256().to_owned();
    let mut fills = Vec::new();
    let replay = in_leased_pool(|cpu| {
        stored
            .capture
            .replay_reviewed_locked_portfolio_v3(
                cpu,
                &stored.authority.store,
                stored.authority.reviewed,
                &locked,
                entry_financial_inputs,
                |ledger, sizing| {
                    cpu.require_current_pool().unwrap();
                    let position = &ledger.positions()[0];
                    fills.push((
                        position.direction(),
                        position.entry_reference().timestamp_unix_ms(),
                        position.modeled_entry_price(),
                        position.exit_reference().unwrap().timestamp_unix_ms(),
                        position.modeled_exit_price().unwrap(),
                        position.exit_reason().unwrap(),
                    ));
                    sized_economics(ledger, sizing)
                },
            )
            .unwrap()
    });
    assert_eq!(
        fills,
        vec![
            (
                ResearchPositionDirectionV1::Long,
                at(120_002),
                1.25,
                at(180_000),
                1.375,
                QuoteValidatedResearchExitReasonV1::TrailingStop
            ),
            (
                ResearchPositionDirectionV1::Short,
                at(180_000),
                1.375,
                at(260_001),
                1.25,
                QuoteValidatedResearchExitReasonV1::Target
            ),
        ]
    );
    let evidence = evaluate_locked_portfolio_outer_holdout_v3(&locked, replay).unwrap();
    // Long confidence .5 -> risk .25 -> one lot: 12500 - 14 = 12486.
    // Short entry is at the SAME timestamp as the long exit: no future net
    // settlement is used. Sealed Bid marks the long +12500, entry fee is -7,
    // so equity = 112493. Confidence ZERO uses minimum risk .125, then ML .5:
    // requested 112493*.125*.5/(2*12500) = .2812325, rounded DOWN to .25 lots.
    // Short: 3125 - 3.5 = 3121.5. Exact total = 15607.5, not fixed-1-lot cash.
    // This is an arithmetic oracle, NOT a profitability or live-market claim.
    assert_eq!(evidence.metrics().trade_count(), 2);
    assert_eq!(evidence.metrics().net_profit(), 15_607.5);
    assert_eq!(evidence.metrics().ending_balance(), 115_607.5);
    let second_sizing = evidence.decision_provenance()[1]
        .entry_sizing
        .as_ref()
        .unwrap();
    assert_eq!(second_sizing.balance_before_entry.amount(), 99_993.0);
    assert_eq!(second_sizing.unrealized_before_entry.amount(), 12_500.0);
    assert_eq!(second_sizing.equity_before_entry.amount(), 112_493.0);
    assert_eq!(second_sizing.requested_lots, 0.2812325);
    assert_eq!(second_sizing.filled_lots, 0.25);
    assert_eq!(
        evidence
            .decision_provenance()
            .iter()
            .map(|origin| {
                (
                    origin.portfolio_index,
                    origin.decision_bar_index,
                    origin.risk_pips,
                )
            })
            .collect::<Vec<_>>(),
        vec![(0, 0, 2.0), (1, 1, 2.0)]
    );
    assert_eq!(locked.identity_sha256(), identity_before);
    assert_eq!(
        stored.capture.artifact_class(),
        FinalistQuoteReplayArtifactClassV1::ResearchOnly
    );
    assert_eq!(
        stored.capture.promotion_eligibility(),
        BrokerTruthPromotionEligibilityV1::NotPromotionEligible
    );
    assert_eq!(
        evidence.execution().receipt().artifact_class(),
        QuoteValidatedOuterHoldoutArtifactClassV1::ResearchOnly
    );
    assert_eq!(
        evidence.execution().receipt().promotion_eligibility(),
        QuoteValidatedOuterHoldoutPromotionEligibilityV1::NotPromotionEligible
    );
    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("synthetic replay must not open the global financial gate");
}

#[test]
fn changed_full_trailing_plans_are_refused_even_when_sealed_fills_are_identical() {
    let signal_fixture = SignalFixture::new();
    let locked = signal_fixture.lock();
    let stored = StoredFixture::new(&locked);
    let snapshot: SealedHistoricalBidAskQuoteReplayEvidenceV1 = stored
        .capture
        .open_reviewed_quote_snapshot_v1(&stored.authority.store, stored.authority.reviewed)
        .unwrap();
    in_leased_pool(|cpu| {
        let threshold = |open, effective, price| {
            ClosedCanonicalBarTrailingThresholdV1::new(
                at(open),
                at(effective),
                ResearchPositionDirectionV1::Long,
                price,
            )
            .unwrap()
        };
        for (label, thresholds, same_fills) in [
            (
                "pre-entry high substituted",
                vec![threshold(120_000, 180_000, 1.75)],
                true,
            ),
            ("trailing omitted", Vec::new(), false),
            (
                "trailing delayed",
                vec![threshold(180_000, 240_000, 1.375)],
                false,
            ),
            (
                "unused future trailing injected",
                vec![
                    threshold(120_000, 180_000, 1.375),
                    threshold(180_000, 240_000, 1.5),
                ],
                true,
            ),
        ] {
            let mut lanes = replay_locked_canonical_signal_portfolio_v3(
                cpu,
                &locked,
                stored.capture.replay_binding(),
                stored.capture.replay_policy(),
                &snapshot,
            )
            .unwrap();
            assert_eq!(
                lanes.iter().map(Vec::len).collect::<Vec<_>>(),
                vec![1, 1, 0]
            );
            let actual = lanes[0].pop().unwrap();
            let mut changed = QuoteValidatedResearchReplayPlanV1::new(
                stored.capture.replay_binding().clone(),
                stored.capture.replay_policy().clone(),
                vec![actual.executed_decision().clone()],
                thresholds,
            )
            .unwrap();
            if let Some(timer) = actual.executed_time_exit() {
                changed = changed.with_time_exit(timer.clone()).unwrap();
            }
            let substituted =
                replay_sealed_quote_validated_research_v1(&changed, &snapshot).unwrap();
            assert_ne!(
                substituted.executed_plan_sha256(),
                actual.executed_plan_sha256(),
                "{label}"
            );
            if same_fills {
                assert_eq!(substituted.positions(), actual.positions(), "{label}");
                assert_eq!(
                    substituted.ledger_sha256(),
                    actual.ledger_sha256(),
                    "legacy fill hash is insufficient: {label}"
                );
            }
            lanes[0].push(substituted);
            let error = LockedPortfolioOuterHoldoutReplaySetV3::new(
                &locked,
                stored.capture.replay_binding(),
                stored.capture.replay_policy(),
                &snapshot,
                lanes,
                entry_financial_inputs,
                sized_economics,
            )
            .expect_err(label);
            assert_eq!(
                error.code(),
                QuoteValidatedOuterHoldoutErrorCodeV1::BindingMismatch,
                "{label}: {error}"
            );
            assert!(
                error.to_string().contains("pinned policy"),
                "{label}: {error}"
            );
        }
    });
}

#[test]
fn acquisition_wrapper_refuses_an_unleased_thread_before_any_store_io() {
    let signal_fixture = SignalFixture::new();
    let locked = signal_fixture.lock();
    let stored = StoredFixture::new(&locked);
    let absent_root = stored.root.0.join("must-stay-absent");
    let absent_store = BrokerTruthAcquisitionStoreV1::new(&absent_root);
    in_leased_pool(|cpu| {
        std::thread::scope(|threads| {
            let task = threads.spawn(|| {
                stored.capture.replay_reviewed_locked_portfolio_v3(
                    cpu,
                    &absent_store,
                    stored.authority.reviewed,
                    &locked,
                    |_, _| panic!("unleased execution must never reach entry sizing"),
                    |_, _| panic!("unleased execution must never reach money calculation"),
                )
            });
            let error = task
                .join()
                .unwrap()
                .expect_err("a borrowed reservation alone does not put another thread in its pool");
            assert_eq!(
                error.code(),
                FinalistQuoteReplayAcquisitionErrorCodeV1::InvalidRequest
            );
            assert!(error.detail().contains("exact active leased CPU pool"));
        });
        cpu.require_current_pool().unwrap();
    });
    assert!(
        !absent_root.exists(),
        "reject the wrong pool before evidence-store IO"
    );
}
