use super::*;

struct Fixture {
    bars: Ohlcv,
    scope: CanonicalSearchArtifactScopeV2,
    genes: Vec<Gene>,
    signals: Vec<Vec<i8>>,
    confidences: Vec<Vec<f64>>,
    size_multipliers: Vec<Vec<f64>>,
}

impl Fixture {
    fn new() -> Self {
        let bars = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
        // Explicit synthetic broker identity for INPUT-contract tests only.
        // The shared sample feature helper deliberately has External scope;
        // do not relax production validation or relabel a real stored receipt.
        use neoethos_feature_contracts::{
            DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOutputV1, FeaturePlanV1,
            SourceArtifactBindingV1, SourceSegmentV1,
        };
        let identity = neoethos_data::CanonicalDatasetIdentity::ctrader(
            neoethos_data::CTraderEnvironment::Demo,
            "prelocked-unit-test",
            101,
            7,
            "EURUSD",
            CanonicalTimeframe::M1,
            neoethos_data::BarTimestampConvention::BarOpen,
        )
        .unwrap();
        let timestamps = bars.timestamp.as_ref().unwrap();
        let plan = FeaturePlanV1::new(
            vec![
                FeatureNodeV1::source(
                    "test:source",
                    identity.clone(),
                    "test.prelocked-input-only.v1",
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
                    "test.prelocked-input-only.v1",
                    [1; 32],
                    "test-only-generation",
                    [2; 32],
                    neoethos_data::BarTimestampConvention::BarOpen,
                    vec![
                        SourceSegmentV1::new(
                            0,
                            timestamps.len() as u64,
                            timestamps[0],
                            *timestamps.last().unwrap(),
                        )
                        .unwrap(),
                    ],
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let features = neoethos_data::FeatureFrame::from_columns(
            timestamps.clone(),
            vec![
                neoethos_data::FeatureColumnF64::new(
                    "test_signal_input",
                    vec![0.0; timestamps.len()],
                    vec![neoethos_data::FeatureCellValidity::Valid; timestamps.len()],
                )
                .unwrap(),
            ],
            plan,
            provenance,
        )
        .unwrap();
        let receipt =
            crate::CanonicalSearchInputReceiptV2::from_feature_frame(&identity, &features).unwrap();
        let scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
            CanonicalSearchWindowRoleV1::Holdout,
            receipt,
        )
        .unwrap();
        let genes = vec![Gene {
            strategy_id: "locked-a".into(),
            sl_pips: 20.0,
            tp_pips: 80.0,
            stop_vol_mult: 0.0,
            ..Gene::default()
        }];
        let signals = vec![vec![0; bars.close.len()]];
        let confidences = vec![vec![0.25; bars.close.len()]];
        let size_multipliers = vec![vec![1.0; bars.close.len()]];
        Self {
            bars,
            scope,
            genes,
            signals,
            confidences,
            size_multipliers,
        }
    }

    fn lock(&self, policy: CanonicalSignalExitPolicyV2) -> LockedCanonicalSignalPlanV3<'_> {
        self.lock_with_adaptive(policy, &adaptive_policy())
    }

    fn lock_with_adaptive(
        &self,
        policy: CanonicalSignalExitPolicyV2,
        adaptive: &crate::stop_target::ResolvedAdaptiveStopsPolicyV1,
    ) -> LockedCanonicalSignalPlanV3<'_> {
        LockedCanonicalSignalPlanV3::new(
            &self.genes,
            &canonical_locked_portfolio_identity_sha256_v1(&self.genes).unwrap(),
            &self.signals,
            &self.confidences,
            &self.size_multipliers,
            &self.bars,
            &self.scope,
            "pinned-config",
            policy,
            risk_policy(),
            adaptive,
        )
        .unwrap()
    }
}

fn risk_policy() -> CanonicalSignalAccountRiskPolicyV3 {
    CanonicalSignalAccountRiskPolicyV3::new(
        AccountMoneyV1::new("USD", 100_000.0).unwrap(),
        0.005,
        0.03,
        0.65,
    )
    .unwrap()
}

fn adaptive_policy() -> crate::stop_target::ResolvedAdaptiveStopsPolicyV1 {
    crate::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(
        crate::stop_target::StopTargetSettings {
            vol_estimator: "parkinson".into(),
            ..Default::default()
        },
        true,
        2.0,
    )
    .unwrap()
}

fn policy() -> CanonicalSignalExitPolicyV2 {
    CanonicalSignalExitPolicyV2 {
        pip_size: 0.0001,
        max_hold_bars: 2,
        trailing_enabled: false,
        trailing_stop_multiplier: 1.0,
        trailing_be_trigger_r: 1.0,
        trailing_min_lock_pips: 0.0,
    }
}

struct Witness {
    decision: CanonicalBarSignalResearchDecisionV1,
    entry: Option<(i64, f64)>,
    completed_at: Option<i64>,
    time_exit: Option<ClosedCanonicalBarTimeExitV1>,
}

impl Witness {
    fn closed(locked: &LockedCanonicalSignalPlanV3<'_>, row: usize, exit_row: usize) -> Self {
        let at = locked.timestamps[row] + locked.duration_ms;
        let entry = locked.bars.close[row];
        let sign = f64::from(locked.ordered_signals[0][row]);
        let (stop, target) = locked.entry_stop_target_pips(0, row).unwrap();
        let timer_row = row + locked.exit_policy.max_hold_bars;
        Self {
            decision: CanonicalBarSignalResearchDecisionV1::new(
                locked.timestamps[row],
                at,
                if sign > 0.0 {
                    ResearchPositionDirectionV1::Long
                } else {
                    ResearchPositionDirectionV1::Short
                },
                entry - sign * stop * locked.exit_policy.pip_size,
                entry + sign * target * locked.exit_policy.pip_size,
            )
            .unwrap(),
            entry: Some((at, entry)),
            completed_at: Some(locked.timestamps[exit_row]),
            time_exit: (locked.exit_policy.max_hold_bars > 0)
                .then(|| locked.timestamps.get(timer_row))
                .flatten()
                .map(|timestamp| {
                    ClosedCanonicalBarTimeExitV1::new(*timestamp, *timestamp + locked.duration_ms)
                        .unwrap()
                }),
        }
    }

    fn view(&self) -> OutcomeView<'_> {
        OutcomeView {
            decision: &self.decision,
            entry: self.entry,
            completed_at: self.completed_at,
            time_exit: self.time_exit.as_ref(),
            plan_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ledger_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        }
    }
}

#[test]
fn pre_acquisition_identity_has_no_dependency_on_future_outcome_count() {
    let mut fixture = Fixture::new();
    fixture.signals[0][0] = 1;
    fixture.signals[0][1] = -1;
    fixture.signals[0][2] = 1;
    let locked = fixture.lock(policy());
    let before = locked.identity_sha256().to_owned();
    // A long-lived position consumes all three signals as one decision.
    let longer = Witness::closed(&locked, 0, 3);
    assert_eq!(
        validate_lane(&locked, 0, [longer.view()].into_iter())
            .unwrap()
            .len(),
        1
    );
    // A different quote path closes early and admits the later signal.
    let first = Witness::closed(&locked, 0, 2);
    let second = Witness::closed(&locked, 2, 4);
    let origins = validate_lane(&locked, 0, [first.view(), second.view()].into_iter()).unwrap();
    assert_eq!(
        origins
            .iter()
            .map(|origin| origin.decision_bar_index)
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    assert_eq!(locked.identity_sha256(), before);
    // This was the old circularity. It remains V1 behavior, not silently V2.
    let old = |risks: &[f64]| {
        canonical_signal_plan_sha256_v1(
            locked.holdout_scope.receipt_sha256(),
            locked.portfolio_identity_sha256(),
            locked.search_config_hash(),
            locked.scope_identity_sha256(),
            locked.ordered_signals(),
            risks,
        )
        .unwrap()
    };
    assert_ne!(old(&[20.0]), old(&[20.0, 20.0]));
    assert_ne!(old(&[20.0]), before);
}

#[test]
fn hash_stream_matches_the_existing_domain_separated_encoding_without_buffering() {
    let value = serde_json::json!({"signals": [[1, 0, -1]], "risk_policy": [20.0, 80.0]});
    assert_eq!(
        streamed_sha256("test.v2", &value).unwrap(),
        stable_sha256("test.v2", &value).unwrap()
    );
}

#[test]
fn every_locked_exit_input_changes_the_pre_acquisition_identity() {
    let fixture = Fixture::new();
    let original = fixture.lock(policy()).identity_sha256().to_owned();
    for changed in [
        CanonicalSignalExitPolicyV2 {
            max_hold_bars: 3,
            ..policy()
        },
        CanonicalSignalExitPolicyV2 {
            pip_size: 0.00001,
            ..policy()
        },
        CanonicalSignalExitPolicyV2 {
            trailing_enabled: true,
            ..policy()
        },
        CanonicalSignalExitPolicyV2 {
            trailing_stop_multiplier: 1.5,
            ..policy()
        },
        CanonicalSignalExitPolicyV2 {
            trailing_be_trigger_r: 0.5,
            ..policy()
        },
        CanonicalSignalExitPolicyV2 {
            trailing_min_lock_pips: 1.0,
            ..policy()
        },
    ] {
        assert_ne!(fixture.lock(changed).identity_sha256(), original);
    }
}

#[test]
fn signals_gene_risk_bar_values_and_config_are_all_pinned() {
    let mut fixture = Fixture::new();
    let original = fixture.lock(policy()).identity_sha256().to_owned();
    fixture.signals[0][0] = 1;
    assert_ne!(fixture.lock(policy()).identity_sha256(), original);
    fixture.signals[0][0] = 0;
    fixture.genes[0].sl_pips += 1.0;
    assert_ne!(fixture.lock(policy()).identity_sha256(), original);
    fixture.genes[0].sl_pips -= 1.0;
    fixture.bars.high[0] += 0.01;
    assert_ne!(fixture.lock(policy()).identity_sha256(), original);
    let fixture = Fixture::new();
    let other = LockedCanonicalSignalPlanV3::new(
        &fixture.genes,
        &canonical_locked_portfolio_identity_sha256_v1(&fixture.genes).unwrap(),
        &fixture.signals,
        &fixture.confidences,
        &fixture.size_multipliers,
        &fixture.bars,
        &fixture.scope,
        "other-config",
        policy(),
        risk_policy(),
        &adaptive_policy(),
    )
    .unwrap();
    assert_ne!(other.identity_sha256(), original);
}

#[test]
fn malformed_signals_bars_adaptive_risk_and_invalid_policy_are_refused_before_quotes() {
    let rejects = |fixture: &Fixture, exit| {
        LockedCanonicalSignalPlanV3::new(
            &fixture.genes,
            &canonical_locked_portfolio_identity_sha256_v1(&fixture.genes).unwrap(),
            &fixture.signals,
            &fixture.confidences,
            &fixture.size_multipliers,
            &fixture.bars,
            &fixture.scope,
            "config",
            exit,
            risk_policy(),
            &adaptive_policy(),
        )
        .is_err()
    };
    let mut fixture = Fixture::new();
    fixture.signals[0][0] = 2;
    assert!(rejects(&fixture, policy()));
    fixture.signals[0][0] = 0;
    fixture.signals[0].pop();
    assert!(rejects(&fixture, policy()));
    let mut fixture = Fixture::new();
    fixture.bars.timestamp.as_mut().unwrap()[1] += 1;
    assert!(rejects(&fixture, policy()));
    let mut fixture = Fixture::new();
    fixture.bars.high[0] = f64::NAN;
    assert!(rejects(&fixture, policy()));
    let mut fixture = Fixture::new();
    fixture.genes[0].stop_vol_mult = f64::NAN;
    assert!(rejects(&fixture, policy()));
    let fixture = Fixture::new();
    assert!(!rejects(
        &fixture,
        CanonicalSignalExitPolicyV2 {
            max_hold_bars: 0,
            ..policy()
        }
    ));
    assert!(rejects(
        &fixture,
        CanonicalSignalExitPolicyV2 {
            trailing_enabled: true,
            trailing_stop_multiplier: 0.0,
            ..policy()
        }
    ));
}

#[test]
fn missing_extra_and_reassigned_outcomes_do_not_form_complete_signal_evidence() {
    let mut fixture = Fixture::new();
    fixture.signals[0][0] = 1;
    fixture.signals[0][4] = -1;
    let locked = fixture.lock(policy());
    let first = Witness::closed(&locked, 0, 2);
    let second = Witness::closed(&locked, 4, 6);
    assert!(validate_lane(&locked, 0, [first.view()].into_iter()).is_err());
    assert!(validate_lane(&locked, 0, [second.view(), first.view()].into_iter()).is_err());
    assert!(
        validate_lane(
            &locked,
            0,
            [first.view(), second.view(), second.view()].into_iter()
        )
        .is_err()
    );
    let origins = validate_lane(&locked, 0, [first.view(), second.view()].into_iter()).unwrap();
    assert_eq!(
        origins[1].canonical_source_row,
        locked.holdout_scope.evaluated_window().row_start() + 4
    );
    assert_eq!(origins[1].risk_pips, 20.0);
}

#[test]
fn same_time_occupied_signals_are_not_fabricated_as_new_entries() {
    let mut fixture = Fixture::new();
    fixture.signals[0][0..3].copy_from_slice(&[1, -1, 1]);
    let locked = fixture.lock(policy());
    let first = Witness::closed(&locked, 0, 3);
    assert_eq!(
        validate_lane(&locked, 0, [first.view()].into_iter())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn actual_brackets_and_holding_timer_must_match_the_prelocked_gene_policy() {
    for signal in [-1, 1] {
        let mut fixture = Fixture::new();
        fixture.signals[0][0] = signal;
        let locked = fixture.lock(policy());
        let mut witness = Witness::closed(&locked, 0, 2);
        witness.decision = CanonicalBarSignalResearchDecisionV1::new(
            witness.decision.signal_bar_open_unix_ms(),
            witness.decision.decision_at_unix_ms(),
            witness.decision.direction(),
            witness.decision.stop_price() + 0.0001,
            witness.decision.target_price(),
        )
        .unwrap();
        assert!(validate_lane(&locked, 0, [witness.view()].into_iter()).is_err());
        let mut witness = Witness::closed(&locked, 0, 2);
        witness.time_exit = None;
        assert!(validate_lane(&locked, 0, [witness.view()].into_iter()).is_err());
    }
}

#[test]
fn explicit_non_entry_deadline_controls_next_eligible_signal() {
    let mut fixture = Fixture::new();
    fixture.signals[0][0..4].copy_from_slice(&[1, -1, 1, -1]);
    let locked = fixture.lock(policy());
    let mut pending = Witness::closed(&locked, 0, 3);
    pending.entry = None;
    pending.time_exit = None;
    let next = Witness::closed(&locked, 3, 5);
    let origins = validate_lane(&locked, 0, [pending.view(), next.view()].into_iter()).unwrap();
    assert_eq!(
        origins
            .iter()
            .map(|origin| origin.decision_bar_index)
            .collect::<Vec<_>>(),
        vec![0, 3]
    );
}

#[test]
fn terminal_open_position_is_retained_and_no_later_trade_is_invented() {
    let mut fixture = Fixture::new();
    fixture.signals[0].fill(1);
    let locked = fixture.lock(CanonicalSignalExitPolicyV2 {
        max_hold_bars: fixture.bars.close.len() + 1,
        ..policy()
    });
    let mut open = Witness::closed(&locked, 0, 3);
    open.completed_at = None;
    assert_eq!(
        validate_lane(&locked, 0, [open.view()].into_iter())
            .unwrap()
            .len(),
        1
    );
    assert!(validate_lane(&locked, 0, [open.view(), open.view()].into_iter()).is_err());
}

#[test]
fn flat_and_last_bar_signals_produce_no_invented_decision_receipts() {
    let mut fixture = Fixture::new();
    *fixture.signals[0].last_mut().unwrap() = 1;
    let locked = fixture.lock(policy());
    assert!(
        validate_lane(&locked, 0, std::iter::empty())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn adaptive_bracket_risk_and_confidence_sizing_use_the_same_independent_distance() {
    let mut fixture = Fixture::new();
    fixture.genes[0].stop_vol_mult = 1.5;
    fixture.signals[0][4] = 1;
    let settings = crate::stop_target::StopTargetSettings {
        vol_estimator: "parkinson".into(),
        vol_window: 2,
        tail_window: 3,
        stop_k_vol: 0.0,
        stop_k_tail: 0.0,
        meta_label_min_dist: 0.002,
        ..Default::default()
    };
    let adaptive =
        crate::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(settings.clone(), true, 9.0)
            .unwrap();
    let locked = fixture.lock_with_adaptive(policy(), &adaptive);
    // Explicit floor / pip = 20; gene multiplier = 1.5; gene RR = 80/20 = 4.
    // The deliberately different fallback RR=9 must not erase the evolved 4R.
    assert_eq!(locked.entry_stop_target_pips(0, 4), Some((30.0, 120.0)));
    let witness = Witness::closed(&locked, 4, 6);
    let origins = validate_lane(&locked, 0, [witness.view()].into_iter()).unwrap();
    assert_eq!(origins[0].risk_pips, 30.0);
    assert_eq!(origins[0].confidence, 0.25);
    let equity = AccountMoneyV1::new("USD", 100_000.0).unwrap();
    let lots = locked
        .account_risk_policy()
        .entry_lots(0.25, &equity, origins[0].risk_pips, 10.0)
        .unwrap();
    let expected = 100_000.0 * (0.005 + 0.025 * (0.25 / 0.65)) / 300.0;
    assert!((lots - expected).abs() < 1e-12);
    let changed = crate::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(
        crate::stop_target::StopTargetSettings {
            meta_label_min_dist: 0.004,
            ..settings
        },
        true,
        9.0,
    )
    .unwrap();
    let changed_lock = fixture.lock_with_adaptive(policy(), &changed);
    assert_eq!(
        changed_lock.entry_stop_target_pips(0, 4),
        Some((60.0, 240.0))
    );
    assert_ne!(locked.identity_sha256(), changed_lock.identity_sha256());
}

#[test]
fn adaptive_warmup_and_zero_account_risk_are_nonentries_before_quote_occupancy() {
    let mut fixture = Fixture::new();
    fixture.genes[0].stop_vol_mult = 1.0;
    fixture.signals[0][0] = 1;
    let adaptive = crate::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(
        crate::stop_target::StopTargetSettings {
            vol_estimator: "parkinson".into(),
            vol_window: 2,
            tail_window: 3,
            stop_k_tail: 0.0,
            meta_label_min_dist: 0.001,
            ..Default::default()
        },
        true,
        2.0,
    )
    .unwrap();
    let locked = fixture.lock_with_adaptive(policy(), &adaptive);
    assert_eq!(locked.entry_stop_target_pips(0, 0), None);
    assert!(!locked.entry_eligible(0, 0));
    assert!(locked.entry_stop_target_pips(0, 1).is_some());
    assert!(
        validate_lane(&locked, 0, std::iter::empty())
            .unwrap()
            .is_empty()
    );
    fixture.genes[0].stop_vol_mult = 0.0;
    fixture.confidences[0][0] = 0.0;
    let locked = LockedCanonicalSignalPlanV3::new(
        &fixture.genes,
        &"c".repeat(64),
        &fixture.signals,
        &fixture.confidences,
        &fixture.size_multipliers,
        &fixture.bars,
        &fixture.scope,
        "config",
        policy(),
        CanonicalSignalAccountRiskPolicyV3::new(
            AccountMoneyV1::new("USD", 100_000.0).unwrap(),
            0.0,
            0.03,
            0.65,
        )
        .unwrap(),
        &adaptive,
    )
    .unwrap();
    assert!(!locked.entry_eligible(0, 0));
    assert!(
        validate_lane(&locked, 0, std::iter::empty())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn zero_holding_limit_has_no_timer_and_full_artifact_identity_is_bound() {
    let mut fixture = Fixture::new();
    fixture.signals[0][0] = 1;
    let exit = CanonicalSignalExitPolicyV2 {
        max_hold_bars: 0,
        ..policy()
    };
    let locked = fixture.lock(exit);
    let witness = Witness::closed(&locked, 0, 3);
    assert!(witness.time_exit.is_none());
    assert_eq!(
        validate_lane(&locked, 0, [witness.view()].into_iter())
            .unwrap()
            .len(),
        1
    );
    let other = LockedCanonicalSignalPlanV3::new(
        &fixture.genes,
        &"d".repeat(64),
        &fixture.signals,
        &fixture.confidences,
        &fixture.size_multipliers,
        &fixture.bars,
        &fixture.scope,
        "pinned-config",
        exit,
        risk_policy(),
        &adaptive_policy(),
    )
    .unwrap();
    assert_ne!(locked.identity_sha256(), other.identity_sha256());
    assert_eq!(other.portfolio_identity_sha256(), "d".repeat(64));
}
