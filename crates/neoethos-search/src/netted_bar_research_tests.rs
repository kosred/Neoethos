use super::*;

// Deliberately hand-calculated account-money fixtures. This private constructor
// tests arithmetic, not canonical data authority or broker execution evidence.
fn settings() -> BacktestSettings {
    BacktestSettings {
        initial_equity_override: Some(1_000.0),
        pip_value: 1.0,
        pip_value_per_lot: 10.0,
        sl_pips: 2.0,
        tp_pips: 4.0,
        spread_pips: 0.0,
        commission_per_trade: 2.0,
        risk_based_sizing: true,
        risk_per_trade_min: 0.01,
        risk_per_trade_max: 0.03,
        high_quality_confidence: 1.0,
        max_hold_bars: 0,
        gap_threshold_ms: 0,
        kill_zones_enabled: false,
        trailing_enabled: false,
        session_spread_profile: None,
        swap_long_pips_per_day: 0.0,
        swap_short_pips_per_day: 0.0,
        pnl_conversion_fee_rate: 0.0,
        ..Default::default()
    }
}

fn prepared(close: &[f64], high: &[f64], low: &[f64]) -> PreparedNettedCanonicalBarResearchV1 {
    PreparedNettedCanonicalBarResearchV1::from_checked_parts(
        close.to_vec(),
        high.to_vec(),
        low.to_vec(),
        (0..close.len())
            .map(|row| 1_704_067_200_000 + row as i64 * 60_000)
            .collect(),
        settings(),
        "USD".into(),
        "private-numerical-fixture-not-authority".into(),
        "private-explicit-screening-cost-fixture".into(),
    )
    .unwrap()
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "actual={actual:.15} expected={expected:.15}"
    );
}

fn joint_policy() -> JointBarAccountPolicyV1 {
    JointBarAccountPolicyV1 {
        max_open_positions: 4,
        max_entry_stop_risk_fraction: 0.02,
        max_gross_notional_multiple: 30.0,
        max_target_pips: 20.0,
    }
}

fn joint_single(
    run: &PreparedNettedCanonicalBarResearchV1,
    tape: NettedBarDecisionTapeV1<'_>,
) -> JointBarAccountResultV1 {
    run.evaluate_joint(
        &[JointBarLaneV1 {
            expert_id: "a",
            tape,
        }],
        joint_policy(),
        None,
        true,
    )
    .unwrap()
}

fn goal_fixture(high: &[f64], low: &[f64]) -> PreparedNettedCanonicalBarResearchV1 {
    let mut run = prepared(&vec![100.; high.len()], high, low);
    // Real 180-day deadlines, independently chosen sparse numerical fixtures.
    let start = run.timestamps[0];
    run.timestamps = [0, 60, 120, 180, 181]
        .iter()
        .take(high.len())
        .map(|d| start + d * 86_400_000)
        .collect();
    run
}

fn goal_single(run: &PreparedNettedCanonicalBarResearchV1) -> JointBarAccountResultV1 {
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1; 5],
        confidences: &[1.; 5],
        sl_pips: &[2.; 5],
        tp_pips: &[4.; 5],
        ml_multipliers: &[1.; 5],
    };
    run.evaluate_joint_risky(
        &[JointBarLaneV1 {
            expert_id: "a",
            tape,
        }],
        joint_policy(),
        None,
        true,
        crate::scoring::RiskyGrowthGoal {
            start_balance: 1000.,
            target_balance: 1030.,
            horizon_days: 180.,
        },
    )
    .unwrap()
}

#[test]
fn risky_joint_banks_only_realized_target_and_ignores_later_losses() {
    let run = goal_fixture(
        &[100., 100., 104., 100., 100.],
        &[100., 100., 100., 98., 98.],
    );
    let out = goal_single(&run);
    // Admission budget20 =>1lot; TP40 less commission2 =>38 settled profit.
    near(out.valuation.ending_realized_balance, 1038.);
    assert_eq!(out.closed_trades.len(), 1);
    assert!(out.terminal_open.is_empty());
    let goal = out.risky_goal.unwrap();
    assert_eq!(goal.status, "target_banked");
    assert_eq!(goal.banked_target_timestamp_ms, Some(run.timestamps[2]));
    assert!(goal.full_horizon_available);
}

#[test]
fn risky_joint_deadline_does_not_credit_a_later_target_or_force_close() {
    let run = goal_fixture(&[100., 100., 100., 100., 104.], &[100.; 5]);
    let out = goal_single(&run);
    near(out.valuation.ending_realized_balance, 1000.);
    near(out.valuation.ending_modeled_net_equity, 998.);
    assert_eq!(out.terminal_open.len(), 1);
    assert!(out.closed_trades.is_empty());
    assert_eq!(out.valuation.last_processed_timestamp_ms, run.timestamps[3]);
    assert_eq!(
        out.risky_goal.unwrap().status,
        "deadline_elapsed_without_banked_target"
    );
}

#[test]
fn risky_joint_cash_crossing_with_an_open_losing_sleeve_is_not_success() {
    let run = goal_fixture(
        &[100., 100., 104., 100., 100.],
        &[100., 100., 100., 98., 100.],
    );
    let tape = |tp| NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0, 0, 0],
        confidences: &[1.; 5],
        sl_pips: &[2.; 5],
        tp_pips: tp,
        ml_multipliers: &[0.5; 5],
    };
    let out = run
        .evaluate_joint_risky(
            &[
                JointBarLaneV1 {
                    expert_id: "a",
                    tape: tape(&[4.; 5]),
                },
                JointBarLaneV1 {
                    expert_id: "b",
                    tape: tape(&[8.; 5]),
                },
            ],
            joint_policy(),
            None,
            true,
            crate::scoring::RiskyGrowthGoal {
                start_balance: 1000.,
                target_balance: 1010.,
                horizon_days: 180.,
            },
        )
        .unwrap();
    // Two .5lot sleeves: A settles +19, then B settles -11 =>1008.
    near(out.valuation.ending_realized_balance, 1008.);
    let goal = out.risky_goal.unwrap();
    assert_eq!(
        goal.first_realized_crossing_timestamp_ms,
        Some(run.timestamps[2])
    );
    assert_eq!(goal.banked_target_timestamp_ms, None);
    assert_eq!(goal.status, "deadline_elapsed_without_banked_target");
}

#[test]
fn risky_joint_requires_actual_start_capital_and_complete_horizon_to_grade_failure() {
    let run = goal_fixture(&[100.; 3], &[100.; 3]);
    let mut goal = crate::scoring::RiskyGrowthGoal {
        start_balance: 1000.,
        target_balance: 50000.,
        horizon_days: 180.,
    };
    let out = run
        .evaluate_joint_risky(&[], joint_policy(), None, true, goal)
        .unwrap();
    assert_eq!(out.risky_goal.unwrap().status, "incomplete_history");
    goal.start_balance = 100.;
    assert!(
        run.evaluate_joint_risky(&[], joint_policy(), None, true, goal)
            .is_err()
    );
}

#[test]
fn allocation_update_keeps_open_protection_and_sizes_new_expert_from_the_same_account() {
    let run = prepared(&[100.; 5], &[100., 100., 104., 100., 108.], &[100.; 5]);
    let older = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0, 0, 0],
        confidences: &[1.; 5],
        sl_pips: &[2.; 5],
        tp_pips: &[8., 1., 1., 1., 1.],
        ml_multipliers: &[0.5, 0., 0., 0., 0.],
    };
    let newer = NettedBarDecisionTapeV1 {
        signals: &[0, 0, 1, 0, 0],
        confidences: &[1.; 5],
        sl_pips: &[2.; 5],
        tp_pips: &[4.; 5],
        ml_multipliers: &[0., 0., 1., 1., 1.],
    };
    let out = run
        .evaluate_joint(
            &[
                JointBarLaneV1 {
                    expert_id: "older",
                    tape: older,
                },
                JointBarLaneV1 {
                    expert_id: "newer",
                    tape: newer,
                },
            ],
            joint_policy(),
            None,
            true,
        )
        .unwrap();
    // Old sleeve: .75 lots, 15 initial stop-risk. At the new entry its pending
    // commission leaves998.5 equity; remaining budget998.5*.02-15=4.97.
    // New lots=.2485. Old target remains8 even when its allocation becomes0.
    let old = out
        .closed_trades
        .iter()
        .find(|t| t.expert_id == "older")
        .unwrap();
    let new = out
        .closed_trades
        .iter()
        .find(|t| t.expert_id == "newer")
        .unwrap();
    near(old.terms.target_pips, 8.0);
    near(old.trade.pnl, 58.5);
    near(new.terms.lots, 0.2485);
    near(new.trade.pnl, 9.443);
    near(out.valuation.ending_realized_balance, 1067.943);
    assert_eq!(out.admission.max_simultaneous_positions, 2);
    assert_eq!(old.trade.exit_time, Some(run.timestamps[4]));
    assert_eq!(new.trade.entry_time, run.timestamps[3]);
}

#[test]
fn simultaneous_experts_share_cash_budget_and_keep_independent_brackets() {
    let run = prepared(&[100.; 4], &[100., 100., 104., 108.], &[100.; 4]);
    let first = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0, 0],
        confidences: &[1.; 4],
        sl_pips: &[2.; 4],
        tp_pips: &[4.; 4],
        ml_multipliers: &[1.; 4],
    };
    let second = NettedBarDecisionTapeV1 {
        sl_pips: &[4.; 4],
        tp_pips: &[8.; 4],
        ..first
    };
    let lanes = [
        JointBarLaneV1 {
            expert_id: "a",
            tape: first,
        },
        JointBarLaneV1 {
            expert_id: "b",
            tape: second,
        },
    ];
    let out = run
        .evaluate_joint(&lanes, joint_policy(), None, true)
        .unwrap();
    // Each requests 3% of the SAME 1000. Combined 60 risk is scaled to 20:
    // .5 lots at SL2 and .25 at SL4. They close on different bars.
    assert_eq!(out.admission.max_simultaneous_positions, 2);
    assert_eq!(out.closed_trades.len(), 2);
    near(out.closed_trades[0].terms.lots, 0.5);
    near(out.closed_trades[1].terms.lots, 0.25);
    near(out.closed_trades[0].trade.pnl, 19.0);
    near(out.closed_trades[1].trade.pnl, 19.5);
    near(out.valuation.ending_realized_balance, 1038.5);
    near(out.admission.max_admitted_stop_risk_fraction, 0.02);
    assert_eq!(
        out.closed_trades[0].trade.exit_time,
        Some(run.timestamps[2])
    );
    assert_eq!(
        out.closed_trades[1].trade.exit_time,
        Some(run.timestamps[3])
    );
    let reverse = [
        JointBarLaneV1 {
            expert_id: "b",
            tape: second,
        },
        JointBarLaneV1 {
            expert_id: "a",
            tape: first,
        },
    ];
    let reversed = run
        .evaluate_joint(&reverse, joint_policy(), None, true)
        .unwrap();
    assert_eq!(
        serde_json::to_value(out).unwrap(),
        serde_json::to_value(reversed).unwrap()
    );
}

#[test]
fn joint_target_cap_rejects_actual_entry_distance_without_clamping() {
    let run = prepared(&[100.; 4], &[100.; 4], &[100.; 4]);
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 1, 1, 0],
        confidences: &[1.; 4],
        sl_pips: &[2.; 4],
        tp_pips: &[70., 20.0001, 20., 0.],
        ml_multipliers: &[1.; 4],
    };
    let out = run
        .evaluate_joint(
            &[JointBarLaneV1 {
                expert_id: "normal",
                tape,
            }],
            joint_policy(),
            None,
            false,
        )
        .unwrap();
    assert_eq!(out.admission.entries("oversized_target"), 2);
    assert_eq!(out.admission.entries("accepted"), 1);
    near(out.terminal_open[0].terms.target_pips, 20.0);
    assert_eq!(out.terminal_open[0].entry_timestamp_ms, run.timestamps[3]);
    assert!(
        JointBarAccountPolicyV1 {
            max_target_pips: 25.,
            ..joint_policy()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn joint_opposite_positions_keep_marks_separate_without_crediting_unrealized_cash() {
    let run = prepared(
        &[100., 100., 101.],
        &[100., 100., 101.],
        &[100., 100., 101.],
    );
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0],
        confidences: &[1.; 3],
        sl_pips: &[10.; 3],
        tp_pips: &[20.; 3],
        ml_multipliers: &[1.; 3],
    };
    let opposite = NettedBarDecisionTapeV1 {
        signals: &[-1, 0, 0],
        ..tape
    };
    let out = run
        .evaluate_joint(
            &[
                JointBarLaneV1 {
                    expert_id: "long",
                    tape,
                },
                JointBarLaneV1 {
                    expert_id: "short",
                    tape: opposite,
                },
            ],
            joint_policy(),
            None,
            false,
        )
        .unwrap();
    // .1 lots each. Opposite price marks cancel; .2+.2 pending commission.
    assert_eq!(out.terminal_open.len(), 2);
    assert!(out.closed_trades.is_empty());
    near(out.valuation.ending_realized_balance, 1000.0);
    near(out.valuation.ending_modeled_net_equity, 999.6);
    near(out.terminal_open[0].gross_unrealized_account, 1.0);
    near(out.terminal_open[1].gross_unrealized_account, -1.0);
}

#[test]
fn joint_lot_grid_rounds_down_and_rejected_entries_do_not_occupy_slots() {
    let mut run = prepared(&[100.; 4], &[100., 100., 100., 104.], &[100.; 4]);
    run.settings.risk_per_trade_min = 0.02;
    run.settings.risk_per_trade_max = 0.02;
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 1, 0, 0],
        confidences: &[1.; 4],
        sl_pips: &[2.; 4],
        tp_pips: &[4.; 4],
        ml_multipliers: &[0.25, 0.99, 0., 0.],
    };
    let out = run
        .evaluate_joint(
            &[JointBarLaneV1 {
                expert_id: "a",
                tape,
            }],
            joint_policy(),
            Some(BarReplayLotGridV1::new(0.5, 100., 0.5).unwrap()),
            false,
        )
        .unwrap();
    assert_eq!(out.admission.entries("below_min_lots"), 1);
    assert_eq!(out.admission.entries("accepted"), 1);
    near(out.closed_trades[0].terms.lots, 0.5);
    near(out.closed_trades[0].trade.pnl, 19.0);
    assert_eq!(out.closed_trades[0].trade.entry_time, run.timestamps[2]);
}

#[test]
fn explicit_rounded_joint_replay_can_use_the_frozen_fractional_recipe() {
    let mut run = prepared(&[100.; 3], &[100., 100., 104.], &[100.; 3]);
    run.settings.risk_per_trade_min = 0.30;
    run.settings.risk_per_trade_max = 0.30;
    run.settings.risky_account_experiment = Some(crate::eval::RiskyAccountExperimentV1 {
        target_balance: 50_000.0,
        continuous_exact_stop_risk: true,
        lot_constraints: None,
    });
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0],
        confidences: &[1.; 3],
        sl_pips: &[9.5; 3],
        tp_pips: &[4.; 3],
        ml_multipliers: &[1.; 3],
    };
    let lanes = [JointBarLaneV1 {
        expert_id: "archived",
        tape,
    }];
    let exact_sizing = JointBarRiskyOrderSizingV1 {
        leverage: Some(JointBarFixedNotionalLeverageV1 {
            maximum: std::num::NonZeroU32::new(400).unwrap(),
            account_notional_per_lot: 100_000.0,
        }),
        confidence_risk_floor: None,
        lot_grid: None,
        fit_to_buying_power: false,
        max_open_positions: 3,
        price_stops: None,
    };
    let grid = BarReplayLotGridV1::new(1., 10., 1.).unwrap();
    let rounded_sizing = JointBarRiskyOrderSizingV1 {
        lot_grid: Some(grid),
        ..exact_sizing
    };

    let exact = run
        .evaluate_joint_risky_full_size(&lanes, 20., false, None, exact_sizing)
        .unwrap();
    let rounded = run
        .evaluate_joint_risky_full_size(&lanes, 20., false, None, rounded_sizing)
        .unwrap();
    near(exact.closed_trades[0].terms.lots, 300.0 / 95.0);
    near(rounded.closed_trades[0].terms.lots, 3.);
    assert_eq!(exact.lot_grid, None);
    assert_eq!(rounded.lot_grid, Some(grid));
    assert!(run.evaluate(tape, Some(grid)).is_err());

    let policy = run.settings.risky_account_experiment.as_mut().unwrap();
    policy.continuous_exact_stop_risk = false;
    policy.lot_constraints =
        Some(crate::QuoteReplayLotConstraintsV3::new(0.5, 10., 0.5, "a".repeat(64)).unwrap());
    assert!(
        run.evaluate_joint_risky_full_size(&lanes, 20., false, None, rounded_sizing)
            .is_err()
    );
}

#[test]
fn joint_single_lane_reuses_legacy_price_cost_and_compounding_arithmetic() {
    let mut run = prepared(&[100.; 5], &[100., 100., 104., 100., 104.], &[100.; 5]);
    run.settings.spread_pips = 0.2;
    run.settings.swap_long_pips_per_day = -0.3;
    run.settings.pnl_conversion_fee_rate = 0.005;
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 1, 0, 0],
        confidences: &[0.5; 5],
        sl_pips: &[2.; 5],
        tp_pips: &[3.; 5],
        ml_multipliers: &[0.5; 5],
    };
    let legacy = run.evaluate(tape, None).unwrap();
    let joint = run
        .evaluate_joint(
            &[JointBarLaneV1 {
                expert_id: "a",
                tape,
            }],
            joint_policy(),
            None,
            false,
        )
        .unwrap();
    near(
        joint.valuation.ending_realized_balance,
        legacy.ending_realized_balance,
    );
    assert_eq!(joint.closed_trades.len(), 2);
    for (a, b) in joint.closed_trades.iter().zip(&legacy.closed_trades) {
        near(a.trade.pnl, b.pnl);
        near(a.trade.r_multiple, b.r_multiple);
    }
}

#[test]
fn joint_protection_starts_after_entry_and_prefixes_are_causal() {
    let run = prepared(
        &[100.; 4],
        &[100., 110., 104., 100.],
        &[100., 90., 100., 100.],
    );
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0, 0],
        confidences: &[1.; 4],
        sl_pips: &[2.; 4],
        tp_pips: &[4.; 4],
        ml_multipliers: &[1.; 4],
    };
    let out = joint_single(&run, tape);
    assert_eq!(
        out.closed_trades[0].trade.exit_time,
        Some(run.timestamps[2])
    );
    near(out.closed_trades[0].trade.pnl, 38.0);
    let changed = NettedBarDecisionTapeV1 {
        signals: &[1, 0, -1, -1],
        ..tape
    };
    let later = run
        .evaluate_joint(
            &[JointBarLaneV1 {
                expert_id: "a",
                tape: changed,
            }],
            joint_policy(),
            None,
            true,
        )
        .unwrap();
    assert_eq!(
        serde_json::to_value(&out.equity_trace[..2]).unwrap(),
        serde_json::to_value(&later.equity_trace[..2]).unwrap()
    );
    let bad = NettedBarDecisionTapeV1 {
        ml_multipliers: &[f64::NAN; 4],
        ..tape
    };
    assert!(
        run.evaluate_joint(
            &[JointBarLaneV1 {
                expert_id: "a",
                tape: bad
            }],
            joint_policy(),
            None,
            false
        )
        .is_err()
    );
}

#[test]
fn joint_entry_cost_exhaustion_stops_before_later_profit_can_hide_it() {
    let mut run = prepared(&[100.; 4], &[100., 100., 104., 104.], &[100.; 4]);
    run.settings.commission_per_trade = 5000.0;
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0, 0],
        confidences: &[1.; 4],
        sl_pips: &[2.; 4],
        tp_pips: &[4.; 4],
        ml_multipliers: &[1.; 4],
    };
    let out = joint_single(&run, tape);
    assert!(out.valuation.drawdown.capital_bound_breached);
    assert_eq!(out.valuation.last_processed_timestamp_ms, run.timestamps[1]);
    assert!(out.closed_trades.is_empty());
    assert_eq!(out.terminal_open.len(), 1);
    near(out.valuation.ending_realized_balance, 1000.0);
    near(out.valuation.ending_modeled_net_equity, -4000.0);
}

#[test]
fn joint_trailing_cannot_install_a_stop_already_crossed_at_close() {
    let mut run = prepared(
        &[100.; 4],
        &[100., 100., 104., 101.],
        &[100., 100., 100., 99.],
    );
    run.settings.trailing_enabled = true;
    run.settings.trailing_be_trigger_r = 0.1;
    run.settings.trailing_atr_multiplier = 0.5;
    run.settings.trailing_min_lock_pips = 0.0;
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0, 0],
        confidences: &[1.; 4],
        sl_pips: &[2.; 4],
        tp_pips: &[20.; 4],
        ml_multipliers: &[1.; 4],
    };
    let out = run
        .evaluate_joint(
            &[JointBarLaneV1 {
                expert_id: "a",
                tape,
            }],
            joint_policy(),
            None,
            false,
        )
        .unwrap();
    assert!(
        out.closed_trades.is_empty(),
        "cannot invent a later profitable exit at refused stop103"
    );
    assert_eq!(out.terminal_open.len(), 1);
}

#[test]
fn joint_gross_notional_limit_scales_the_whole_entry_batch() {
    let run = prepared(&[100.; 2], &[100.; 2], &[100.; 2]);
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0],
        confidences: &[1.; 2],
        sl_pips: &[2.; 2],
        tp_pips: &[4.; 2],
        ml_multipliers: &[1.; 2],
    };
    let policy = JointBarAccountPolicyV1 {
        max_gross_notional_multiple: 0.5,
        ..joint_policy()
    };
    let out = run
        .evaluate_joint(
            &[
                JointBarLaneV1 {
                    expert_id: "a",
                    tape,
                },
                JointBarLaneV1 {
                    expert_id: "b",
                    tape,
                },
            ],
            policy,
            None,
            false,
        )
        .unwrap();
    near(out.terminal_open[0].terms.lots, 0.25);
    near(out.terminal_open[1].terms.lots, 0.25);
    near(out.admission.max_admitted_gross_notional_multiple, 0.5);
    near(out.admission.max_admitted_stop_risk_fraction, 0.01);
}

#[test]
fn varying_brackets_confidence_and_ml_once_compound_hand_calculated_account_pnl() {
    let run = prepared(
        &[100.0; 5],
        &[100.0, 100.0, 104.0, 100.0, 103.0],
        &[100.0; 5],
    );
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 1, 0, 0],
        confidences: &[0.5, 0.0, 1.0, 0.0, 0.0],
        sl_pips: &[2.0, 0.0, 1.0, 0.0, 0.0],
        tp_pips: &[4.0, 0.0, 3.0, 0.0, 0.0],
        ml_multipliers: &[0.5, 0.0, 0.25, 0.0, 0.0],
    };
    let out = run.evaluate(tape, None).unwrap();
    // First: 1000 * (1% + (3%-1%)*.5) / (2*10) * .5 = .5 lots.
    // (4 pips*10 - 2 commission)*.5 = 19. Second uses ACTUAL 1019 equity:
    // 1019*.03/(1*10)*.25 = .76425 lots; (3*10-2)*.76425 = 21.399.
    assert_eq!(out.closed_trades.len(), 2);
    near(out.closed_trades[0].pnl, 19.0);
    near(out.closed_trades[1].pnl, 21.399);
    near(out.closed_trades[0].r_multiple, 1.9);
    near(out.closed_trades[1].r_multiple, 2.8);
    near(out.ending_realized_balance, 1040.399);
    near(out.metrics.net_profit, 40.399);
    assert_eq!(out.closed_trades[0].entry_time, run.timestamps[1]);
    assert_eq!(out.closed_trades[1].entry_time, run.timestamps[3]);
    assert!(out.terminal_open.is_none());
    assert!(!out.promotion_eligible);
    assert!(out.execution_basis.contains("ohlc_screening"));
}

#[test]
fn below_min_is_non_entry_and_next_signal_can_occupy_the_position() {
    let mut run = prepared(&[100.0; 4], &[100.0, 100.0, 100.0, 104.0], &[100.0; 4]);
    // Keep the requested 1.0 lot exact: this case tests occupancy rather than
    // a floating-point value immediately below a broker grid boundary.
    run.settings.risk_per_trade_min = 0.02;
    run.settings.risk_per_trade_max = 0.02;
    let out = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1, 1, 0, 0],
                confidences: &[0.5; 4],
                sl_pips: &[2.0; 4],
                tp_pips: &[4.0; 4],
                ml_multipliers: &[0.25, 1.0, 0.0, 0.0],
            },
            Some(BarReplayLotGridV1::new(0.5, 100.0, 0.5).unwrap()),
        )
        .unwrap();
    assert_eq!(out.below_min_entries, 1);
    assert_eq!(out.closed_trades.len(), 1);
    assert_eq!(out.closed_trades[0].entry_time, run.timestamps[2]);
    near(out.closed_trades[0].pnl, 38.0);
    near(out.ending_realized_balance, 1038.0);
    assert!(out.terminal_open.is_none());
}

#[test]
fn terminal_position_is_retained_without_fabricated_close_or_realized_profit() {
    let run = prepared(
        &[100.0, 100.0, 101.0],
        &[100.0, 100.0, 101.0],
        &[100.0, 100.0, 101.0],
    );
    let out = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1, 0, 0],
                confidences: &[0.5; 3],
                sl_pips: &[10.0; 3],
                tp_pips: &[20.0; 3],
                ml_multipliers: &[0.5, 0.0, 0.0],
            },
            None,
        )
        .unwrap();
    assert!(out.closed_trades.is_empty());
    assert_eq!(out.metrics.trade_count, 0);
    near(out.metrics.net_profit, 0.0);
    near(out.ending_realized_balance, 1000.0);
    let open = out.terminal_open.unwrap();
    assert_eq!(open.direction, 1);
    assert_eq!(open.entry_bar_index, 1);
    assert_eq!(open.entry_timestamp_ms, run.timestamps[1]);
    assert_eq!(open.mark_timestamp_ms, run.timestamps[2]);
    near(open.lots, 0.1);
    near(open.stop_pips, 10.0);
    near(open.target_pips, 20.0);
    assert!(open.active_trailing_stop_price.is_none());
    near(open.gross_unrealized_account, 1.0);
    near(open.pending_round_trip_commission_account, 0.2);
    near(open.marked_equity_before_pending_costs, 1001.0);
}

#[test]
fn future_tape_cannot_change_a_prior_entry_and_ml_veto_cannot_disable_protection() {
    let run = prepared(
        &[100.0; 5],
        &[100.0, 100.0, 104.0, 100.0, 100.0],
        &[100.0; 5],
    );
    let baseline = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1, 0, 0, 0, 0],
                confidences: &[0.5; 5],
                sl_pips: &[2.0; 5],
                tp_pips: &[4.0; 5],
                ml_multipliers: &[1.0, 0.0, 0.0, 0.0, 0.0],
            },
            None,
        )
        .unwrap();
    let changed_future = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1, -1, -1, -1, -1],
                confidences: &[0.5, 1.0, 1.0, 1.0, 1.0],
                sl_pips: &[2.0, 8.0, 8.0, 8.0, 8.0],
                tp_pips: &[4.0, 16.0, 16.0, 16.0, 16.0],
                ml_multipliers: &[1.0, 0.0, 0.0, 0.0, 0.0],
            },
            None,
        )
        .unwrap();
    assert_eq!(baseline.closed_trades.len(), 1);
    assert_eq!(changed_future.closed_trades.len(), 1);
    assert_eq!(
        serde_json::to_value(&baseline.closed_trades).unwrap(),
        serde_json::to_value(&changed_future.closed_trades).unwrap()
    );
    near(changed_future.closed_trades[0].pnl, 38.0);
    assert_eq!(
        changed_future.closed_trades[0].exit_time,
        Some(run.timestamps[2])
    );
    assert!(changed_future.terminal_open.is_none());
}

#[test]
fn all_one_multipliers_and_fixed_brackets_match_legacy_core_bit_for_bit() {
    let run = prepared(
        &[100.0; 5],
        &[100.0, 100.0, 104.0, 100.0, 104.0],
        &[100.0; 5],
    );
    let signals = [1, 0, 1, 0, 0];
    let confidences = [0.5; 5];
    let out = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &signals,
                confidences: &confidences,
                sl_pips: &[2.0; 5],
                tp_pips: &[4.0; 5],
                ml_multipliers: &[1.0; 5],
            },
            None,
        )
        .unwrap();
    let (legacy_metrics, legacy_trades) =
        crate::eval::evaluate_strategy_with_confidence_and_ledger_core(
            &run.close,
            &run.high,
            &run.low,
            &signals,
            &confidences,
            &run.months,
            &run.days,
            &run.timestamps,
            &run.settings,
            None,
            None,
        )
        .unwrap();
    assert_eq!(
        out.metrics.to_metric_array().map(f64::to_bits),
        legacy_metrics.map(f64::to_bits)
    );
    assert_eq!(
        serde_json::to_value(&out.closed_trades).unwrap(),
        serde_json::to_value(&legacy_trades).unwrap()
    );
}

#[test]
fn malformed_tapes_and_bracket_overflow_fail_before_statistics() {
    let run = prepared(&[100.0; 3], &[100.0; 3], &[100.0; 3]);
    let signals = [1, 0, 0];
    let confidence = [0.5; 3];
    let stops = [2.0; 3];
    let targets = [4.0; 3];
    let multipliers = [1.0; 3];
    let tape = NettedBarDecisionTapeV1 {
        signals: &signals,
        confidences: &confidence,
        sl_pips: &stops,
        tp_pips: &targets,
        ml_multipliers: &multipliers,
    };
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                confidences: &[0.5],
                ..tape
            },
            None
        )
        .is_err()
    );
    for value in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(
            run.evaluate(
                NettedBarDecisionTapeV1 {
                    ml_multipliers: &[value; 3],
                    ..tape
                },
                None
            )
            .is_err()
        );
    }
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                sl_pips: &[f64::MAX; 3],
                ..tape
            },
            None
        )
        .is_err()
    );
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                tp_pips: &[f64::MAX; 3],
                ..tape
            },
            None
        )
        .is_err()
    );
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[2, 0, 0],
                ..tape
            },
            None
        )
        .is_err()
    );
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                sl_pips: &[0.0; 3],
                ..tape
            },
            None
        )
        .is_err()
    );
}

#[test]
fn broker_grid_never_rounds_up_to_force_a_trade_and_zero_risk_is_not_below_min() {
    let grid = BarReplayLotGridV1::new(0.5, 2.0, 0.25).unwrap();
    assert_eq!(grid.normalize_down(0.49), None);
    assert_eq!(grid.normalize_down(0.99), Some(0.75));
    assert_eq!(grid.normalize_down(3.0), Some(2.0));
    assert_eq!(grid.normalize_down(f64::INFINITY), None);
    assert!(BarReplayLotGridV1::new(0.0, 2.0, 0.25).is_err());
    let mut run = prepared(&[100.0; 3], &[100.0; 3], &[100.0; 3]);
    run.settings.risk_per_trade_min = 0.0;
    run.settings.risk_per_trade_max = 0.0;
    let out = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1; 3],
                confidences: &[0.5; 3],
                sl_pips: &[2.0; 3],
                tp_pips: &[4.0; 3],
                ml_multipliers: &[1.0; 3],
            },
            Some(grid),
        )
        .unwrap();
    assert_eq!(out.below_min_entries, 0);
    assert!(out.closed_trades.is_empty());
    assert!(out.terminal_open.is_none());
}

#[test]
fn bound_risky_grid_is_applied_reported_and_cannot_be_overridden() {
    let mut run = prepared(&[100.0; 3], &[100.0; 3], &[100.0, 100.0, 97.0]);
    run.settings.risky_account_experiment = Some(crate::eval::RiskyAccountExperimentV1 {
        target_balance: 50_000.0,
        continuous_exact_stop_risk: false,
        lot_constraints: Some(
            crate::QuoteReplayLotConstraintsV3::new(1.0, 10.0, 1.0, "a".repeat(64)).unwrap(),
        ),
    });
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0],
        confidences: &[1.0; 3],
        sl_pips: &[2.0; 3],
        tp_pips: &[4.0; 3],
        ml_multipliers: &[1.0; 3],
    };
    let out = run.evaluate(tape, None).unwrap();
    // 1.5 requested lots -> 1 admitted lot; SL20 plus commission2.
    near(out.closed_trades[0].pnl, -22.0);
    near(out.ending_realized_balance, 978.0);
    near(out.metrics.max_drawdown, 0.022);
    assert_eq!(
        out.lot_grid,
        Some(BarReplayLotGridV1::new(1.0, 10.0, 1.0).unwrap())
    );
    assert!(
        run.evaluate(tape, Some(BarReplayLotGridV1::new(0.5, 2.0, 0.25).unwrap()))
            .is_err()
    );
    run.settings
        .risky_account_experiment
        .as_mut()
        .unwrap()
        .lot_constraints =
        Some(crate::QuoteReplayLotConstraintsV3::new(2.0, 10.0, 1.0, "a".repeat(64)).unwrap());
    let refused = run.evaluate(tape, None).unwrap();
    assert_eq!(refused.below_min_entries, 1);
    assert!(refused.closed_trades.is_empty());
    near(refused.ending_realized_balance, 1000.0);
}

#[test]
fn public_preparation_evaluates_exact_published_canonical_holdout_and_rejects_rebinding() {
    use crate::data_selection::CanonicalSearchInput;
    use crate::{CanonicalSearchEvaluatedWindowV1, CanonicalTrendbarResearchCostAssumptionsV2};
    use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
    use neoethos_data::{
        BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
        CanonicalTimeframe, CanonicalVolumeRef, FeatureCellValidity, FeatureColumnF64,
        FeatureFrame, publish_canonical_ohlcv_generation,
    };
    use neoethos_feature_contracts::{
        DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOutputV1, FeaturePlanV1,
    };

    // Actual captured fixture prices, persisted and reopened by the real
    // canonical generation path. The explicitly external fixture identity is
    // not broker-truth or live-account evidence.
    let root = tempfile::tempdir().unwrap();
    let anchor = CanonicalDatasetIdentity::external(
        "netted-public-canonical-fixture",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .unwrap();
    let bars = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
    let source = ProducerProvenanceEnvelopeV1::new(
        "neoethos.netted-public-canonical-fixture.v1",
        anchor.canonical_bytes(),
    )
    .unwrap();
    publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root.path(),
        identity: &anchor,
        expected_generation: None,
        provenance: &source,
        ohlcv: &bars,
        volume: CanonicalVolumeRef::Absent,
        rows_per_chunk: 128,
    })
    .unwrap();
    let dataset =
        neoethos_data::load_dataset_for_identity_with_timeframes(root.path(), &anchor, &["M1"])
            .unwrap();
    let base = dataset.canonical_frame("M1").unwrap();
    let source_id = "source:netted-public-canonical-fixture";
    let plan = FeaturePlanV1::new(
        vec![
            FeatureNodeV1::source(
                source_id,
                anchor.clone(),
                "neoethos.netted-public-fixture.close.v1",
                1,
                vec![FeatureOutputV1::f64("close", 1).unwrap()],
                [19; 32],
            )
            .unwrap(),
        ],
        vec!["close".into()],
    )
    .unwrap();
    let provenance = DatasetFeatureArtifactProvenanceV1::new(
        &plan,
        vec![base.source_binding(source_id).unwrap()],
    )
    .unwrap();
    let features = FeatureFrame::from_columns(
        base.ohlcv().timestamp.clone().unwrap(),
        vec![
            FeatureColumnF64::new(
                "close",
                base.ohlcv().close.clone(),
                vec![FeatureCellValidity::Valid; base.len()],
            )
            .unwrap(),
        ],
        plan,
        provenance,
    )
    .unwrap();
    let input =
        CanonicalSearchInput::from_prepared_canonical_frame(anchor, base, features).unwrap();
    let run = input.as_run_input().unwrap();
    let scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::Holdout,
        &run,
        20..28,
    )
    .unwrap();
    let contract = CanonicalTrendbarResearchExecutionContractV3::new(
        run.receipt().clone(),
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "explicit-public-connection-test-costs",
            assumption_source_sha256: &"c".repeat(64),
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.5,
            slippage_pips_per_fill_assumption: 0.5,
            commission_account_per_lot_per_fill_assumption: 7.0,
            swap_long_pips_per_day: -0.25,
            swap_short_pips_per_day: 0.1,
            pnl_conversion_fee_rate: 0.01,
        },
    )
    .unwrap();
    // The published symbol rate above is nonzero, but EURUSD price P&L is
    // already USD. The sealed account-specific assumption must charge no
    // currency-conversion fee when quote and account currencies match.
    assert_eq!(contract.pnl_conversion_fee_rate(), 0.0);
    let evaluation = EvaluationConfig {
        risky_limits_policy: None,
        risky_account_experiment: None,
        symbol: "EURUSD".into(),
        account_currency: "USD".into(),
        initial_equity: 1_000.0,
        max_hold_bars: 1,
        trailing_enabled: false,
        trailing_atr_multiplier: 1.0,
        trailing_be_trigger_r: 1.0,
        trailing_min_lock_pips: 0.0,
        pip_value: 0.0001,
        pip_value_per_lot: 10.0,
        spread_pips: 2.5,
        commission_per_trade: 14.0,
        swap_long_pips_per_day: -0.25,
        swap_short_pips_per_day: 0.1,
        pnl_conversion_fee_rate: contract.pnl_conversion_fee_rate(),
        kill_zones_enabled: false,
        session_spread_pips: None,
        risk_per_trade_min: 0.02,
        risk_per_trade_max: 0.02,
        high_quality_confidence: 1.0,
        smc_gate_threshold: 0.0,
        smc_weight_ob: 1.0,
        smc_weight_fvg: 1.0,
        smc_weight_liq: 1.0,
        smc_weight_mtf: 1.0,
        smc_weight_premium: 1.0,
        smc_weight_inducement: 1.0,
        smc_weight_bos: 1.0,
        smc_weight_choch: 1.0,
        smc_weight_eqh: 1.0,
        smc_weight_eql: 1.0,
        smc_weight_displacement: 1.0,
        growth_objective: false,
        growth_goal: None,
        weekly_objective: None,
        fitness_table: Default::default(),
    };
    contract.validate_evaluation_costs(&evaluation).unwrap();
    let prepared =
        PreparedNettedCanonicalBarResearchV1::from_input(&run, &scope, &contract, &evaluation)
            .unwrap();
    assert_eq!(
        prepared.timestamps(),
        &bars.timestamp.as_ref().unwrap()[20..28]
    );
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0, 0, 0, 0, 0, 0],
        confidences: &[0.5; 8],
        sl_pips: &[100.0; 8],
        tp_pips: &[200.0; 8],
        ml_multipliers: &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    let out = prepared.evaluate(tape, None).unwrap();
    // This tests explicit *bar-assumption* accounting, not broker settlement.
    // Size =1000*.02/(100*10)*.5=.01 lots.
    // Two half-spreads sum to2.5 pips; two fill commissions sum to14/lot.
    // Carry is prorated elapsed days. No USD-to-USD conversion takes place:
    // neither a price loss nor commission/swap may receive the old 1% discount.
    let elapsed_days = (bars.timestamp.as_ref().unwrap()[22] - bars.timestamp.as_ref().unwrap()[21])
        as f64
        / 86_400_000.0;
    let price_gross_account = ((bars.close[22] - bars.close[21]) / 0.0001 * 10.0 - 25.0) * 0.01;
    let commission_account = 14.0 * 0.01;
    let swap_account = -0.25 * elapsed_days * 10.0 * 0.01;
    let expected = price_gross_account - commission_account + swap_account;
    assert_eq!(out.closed_trades.len(), 1);
    assert_eq!(
        out.closed_trades[0].entry_time,
        bars.timestamp.as_ref().unwrap()[21]
    );
    assert_eq!(
        out.closed_trades[0].exit_time,
        Some(bars.timestamp.as_ref().unwrap()[22])
    );
    near(out.closed_trades[0].pnl, expected);
    near(out.ending_realized_balance, 1000.0 + expected);
    assert_eq!(out.scope_identity_sha256, scope.identity_sha256().unwrap());
    assert_eq!(
        out.cost_contract_identity_sha256,
        contract.identity_sha256().unwrap()
    );
    assert_eq!(out.account_currency, "USD");
    assert!(out.terminal_open.is_none());
    assert!(!out.promotion_eligible);

    // The existing fitting adapter must use the same cost-charged lifecycle,
    // keep its declared scope identity, and refuse the final-holdout role.
    assert!(
        PreparedNettedCanonicalBarResearchV1::from_training_scope(
            &run,
            &scope,
            &contract,
            &evaluation,
        )
        .is_err()
    );
    for role in [
        CanonicalSearchWindowRoleV1::InSample,
        CanonicalSearchWindowRoleV1::SelectionValidation,
    ] {
        let fitting_scope =
            CanonicalSearchArtifactScopeV2::from_run_input_range(role, &run, 20..28).unwrap();
        let fitting = PreparedNettedCanonicalBarResearchV1::from_training_scope(
            &run,
            &fitting_scope,
            &contract,
            &evaluation,
        )
        .expect("explicit non-final scope");
        let fitting_out = fitting.evaluate(tape, None).unwrap();
        assert_eq!(fitting_out.metrics, out.metrics);
        near(fitting_out.ending_realized_balance, 1000.0 + expected);
        assert_eq!(
            fitting_out.scope_identity_sha256,
            fitting_scope.identity_sha256().unwrap()
        );
        assert_ne!(fitting_out.scope_identity_sha256, out.scope_identity_sha256);
        assert_eq!(
            fitting_out.cost_contract_identity_sha256,
            out.cost_contract_identity_sha256
        );
        assert!(!fitting_out.promotion_eligible);
        assert!(
            PreparedNettedCanonicalBarResearchV1::from_input(
                &run,
                &fitting_scope,
                &contract,
                &evaluation,
            )
            .is_err()
        );
    }

    let wrong_role = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::InSample,
        &run,
        20..28,
    )
    .unwrap();
    assert!(PreparedNettedCanonicalBarResearchV1::from_input(
        &run, &wrong_role, &contract, &evaluation,
    ).is_err());
    let wrong_rows = CanonicalSearchArtifactScopeV2::new(
        run.receipt().clone(),
        CanonicalSearchEvaluatedWindowV1::new(
            CanonicalSearchWindowRoleV1::Holdout,
            21,
            29,
            bars.timestamp.as_ref().unwrap()[20],
            bars.timestamp.as_ref().unwrap()[27],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(PreparedNettedCanonicalBarResearchV1::from_input(
        &run, &wrong_rows, &contract, &evaluation,
    ).is_err());
    for (field, field_name) in [
        "account_currency",
        "symbol",
        "pip_value",
        "pip_value_per_lot",
        "spread_pips",
        "commission_per_trade",
        "swap_long_pips_per_day",
        "swap_short_pips_per_day",
        "pnl_conversion_fee_rate",
    ]
    .iter()
    .enumerate()
    {
        let mut changed = evaluation.clone();
        match field {
            0 => changed.account_currency = "EUR".into(),
            1 => changed.symbol = "GBPUSD".into(),
            2 => changed.pip_value = 0.01,
            3 => changed.pip_value_per_lot = 20.0,
            4 => changed.spread_pips += 0.1,
            5 => changed.commission_per_trade += 0.1,
            6 => changed.swap_long_pips_per_day += 0.1,
            7 => changed.swap_short_pips_per_day += 0.1,
            _ => changed.pnl_conversion_fee_rate = f64::EPSILON,
        }
        let expected_error = contract
            .validate_evaluation_costs(&changed)
            .unwrap_err()
            .to_string();
        assert!(expected_error.contains(&format!(": {field_name} (evaluation ")));
        let preparation_error =
            PreparedNettedCanonicalBarResearchV1::from_input(&run, &scope, &contract, &changed)
                .err()
                .expect("financial field must not detach from its contract");
        assert_eq!(preparation_error.to_string(), expected_error);
        contract.validate_evaluation_costs(&evaluation).unwrap();
    }
    // Preparation owns only the bounded held-out data: subsequent evaluation
    // remains possible after both source dataset and feature cube are gone.
    drop(run);
    drop(input);
    drop(dataset);
    let detached = prepared.evaluate(tape, None).unwrap();
    assert_eq!(detached.metrics, out.metrics);
    near(detached.ending_realized_balance, 1000.0 + expected);
}
