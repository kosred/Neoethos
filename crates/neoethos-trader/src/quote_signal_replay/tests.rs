use super::*;
use neoethos_broker_truth::{
    CompleteBidAskQuoteReplayEvidenceV1, CompleteQuoteSideCoverageV1, EvidenceWindowV1,
    ExactHistoricalQuoteV1, ExactQuoteSourceOrdinalV1, LockedFinalistOosReplayScopeV1, QuoteSideV1,
    QuoteValidatedResearchAuthorityV1, QuoteValidatedResearchExitReasonV1,
    QuoteValidatedResearchLedgerV1, QuoteValidatedResearchReplayErrorCodeV1,
    QuoteValidatedResearchReplayErrorV1, ReviewedQuoteReplayRuleIdentityV2,
    VersionedLatencySlippagePolicyV1, preview_quote_validated_research_entry_v1,
    replay_quote_validated_research_v1,
};

const START: i64 = 60_000;
const BAR: i64 = 60_000;
const ROWS: usize = 8;
const PIP: f64 = 0.0001;

impl LaneLedger for QuoteValidatedResearchLedgerV1 {
    fn positions(&self) -> &[QuoteValidatedResearchPositionV1] {
        self.positions()
    }
    fn non_entries(&self) -> &[QuoteValidatedResearchNonEntryV1] {
        self.entry_unavailable()
    }
}

fn binding() -> QuoteValidatedResearchReplayBindingV1 {
    QuoteValidatedResearchReplayBindingV1::new(
        "11".repeat(32),
        "22".repeat(32),
        7,
        42,
        "EURUSD",
        LockedFinalistOosReplayScopeV1::new(
            EvidenceWindowV1::new(START, START + ROWS as i64 * BAR).unwrap(),
            1_000,
            100_000,
        )
        .unwrap(),
        ReviewedQuoteReplayRuleIdentityV2::new("33".repeat(32), "44".repeat(32), "55".repeat(32))
            .unwrap(),
        "66".repeat(32),
    )
    .unwrap()
}

fn policy(entry_wait: i64, exit_wait: i64) -> QuoteValidatedResearchReplayPolicyV1 {
    QuoteValidatedResearchReplayPolicyV1::new(
        entry_wait,
        1_000,
        exit_wait,
        VersionedLatencySlippagePolicyV1::new("synthetic-lane-policy", 9, 5, 0.5, PIP).unwrap(),
        None,
    )
    .unwrap()
}

fn gene() -> neoethos_search::Gene {
    neoethos_search::Gene {
        strategy_id: "synthetic-locked-lane".to_owned(),
        sl_pips: 20.0,
        tp_pips: 100.0,
        ..Default::default()
    }
}

fn bars() -> Vec<LiveBar> {
    (0..ROWS)
        .map(|index| LiveBar {
            symbol: "EURUSD".to_owned(),
            tf: "M1".to_owned(),
            o: 1.0,
            h: 1.0005,
            l: 0.9995,
            c: 1.0,
            volume: 0.0,
            ts: START + index as i64 * BAR,
        })
        .collect()
}

fn records() -> (Vec<(i64, f64)>, Vec<(i64, f64)>) {
    let bid = (1..=ROWS)
        .map(|index| (START + index as i64 * BAR + 9, 1.0))
        .collect();
    let ask = (1..=ROWS)
        .flat_map(|index| {
            [
                (START + index as i64 * BAR + 8, 1.0002),
                (START + index as i64 * BAR + 10, 1.0002),
            ]
        })
        .collect();
    (bid, ask)
}

fn evidence(
    mut bid: Vec<(i64, f64)>,
    mut ask: Vec<(i64, f64)>,
) -> CompleteBidAskQuoteReplayEvidenceV1 {
    bid.sort_by_key(|row| row.0);
    ask.sort_by_key(|row| row.0);
    let side = |side, rows: Vec<(i64, f64)>| {
        CompleteQuoteSideCoverageV1::new(
            side,
            7,
            42,
            binding().replay_scope().required_quote_coverage_window(),
            "77".repeat(32),
            "88".repeat(32),
            rows.into_iter()
                .enumerate()
                .map(|(index, (timestamp, price))| {
                    ExactHistoricalQuoteV1::new(
                        timestamp,
                        price,
                        ExactQuoteSourceOrdinalV1::new(0, 0, index as u64).unwrap(),
                    )
                    .unwrap()
                })
                .collect(),
            false,
        )
        .unwrap()
    };
    CompleteBidAskQuoteReplayEvidenceV1::new(
        binding(),
        side(QuoteSideV1::Bid, bid),
        side(QuoteSideV1::Ask, ask),
    )
    .unwrap()
}

fn run(
    lane: &CanonicalSignalQuoteLaneV1<'_>,
    policy: &QuoteValidatedResearchReplayPolicyV1,
    evidence: &CompleteBidAskQuoteReplayEvidenceV1,
) -> Result<Vec<(usize, QuoteValidatedResearchLedgerV1)>> {
    drive_lane(
        lane,
        &binding(),
        policy,
        |plan| Ok(preview_quote_validated_research_entry_v1(plan, evidence)?),
        |plan| Ok(replay_quote_validated_research_v1(plan, evidence.clone())?),
    )
}

fn lane<'a>(
    bars: &'a [LiveBar],
    signals: &'a [i8],
    gene: &'a neoethos_search::Gene,
    max_hold_bars: usize,
) -> CanonicalSignalQuoteLaneV1<'a> {
    CanonicalSignalQuoteLaneV1 {
        bars,
        signals,
        gene,
        pip_size: PIP,
        max_hold_bars,
        trailing: None,
    }
}

#[test]
fn borrowed_canonical_columns_preserve_live_bar_decisions_fills_and_trailing() {
    let mut bars = bars();
    bars[2].h = 1.006;
    bars[2].l = 0.994;
    let timestamps: Vec<_> = bars.iter().map(|bar| bar.ts).collect();
    let columns = neoethos_data::Ohlcv {
        timestamp: None,
        open: bars.iter().map(|bar| bar.o).collect(),
        high: bars.iter().map(|bar| bar.h).collect(),
        low: bars.iter().map(|bar| bar.l).collect(),
        close: bars.iter().map(|bar| bar.c).collect(),
        volume: None,
    };
    let symbol = String::from("EURUSD");
    let timeframe = String::from("M1");
    let borrowed = QuoteBars::Canonical {
        ohlcv: &columns,
        timestamps: &timestamps,
        symbol: &symbol,
        timeframe: &timeframe,
    };
    let first = borrowed.get(0).unwrap();
    assert!(std::ptr::eq(first.symbol.as_ptr(), symbol.as_ptr()));
    assert!(std::ptr::eq(first.tf.as_ptr(), timeframe.as_ptr()));
    let QuoteBars::Canonical {
        ohlcv,
        timestamps: view_times,
        ..
    } = borrowed
    else {
        unreachable!()
    };
    assert!(std::ptr::eq(ohlcv.open.as_ptr(), columns.open.as_ptr()));
    assert!(std::ptr::eq(view_times.as_ptr(), timestamps.as_ptr()));

    let gene = gene();
    let (mut bid, mut ask) = records();
    // Price-triggered exits need a later executable quote after the explicit
    // five-ms latency. The trigger quote itself cannot be reused as the fill.
    for index in 1..=ROWS {
        bid.push((START + index as i64 * BAR + 20, 1.0));
        ask.push((START + index as i64 * BAR + 20, 1.0002));
    }
    let evidence = evidence(bid, ask);
    for direction in [-1, 1] {
        for trailing in [None, TrailingPolicy::new(1.0, 1.0, 2.0, PIP)] {
            let signals = [direction; ROWS];
            let mut expected = None;
            for view in [QuoteBars::Live(&bars), borrowed] {
                let lane = BorrowedSignalQuoteLane {
                    bars: view,
                    signals: &signals,
                    confidences: Some(&[0.25; ROWS]),
                    entry_eligibility: None,
                    entry_brackets: None,
                    gene: &gene,
                    pip_size: PIP,
                    max_hold_bars: 3,
                    trailing,
                };
                let mut plans = Vec::new();
                let ledgers = drive_borrowed_lane(
                    &lane,
                    &binding(),
                    &policy(1_000, 1_000),
                    |plan| Ok(preview_quote_validated_research_entry_v1(plan, &evidence)?),
                    |plan| {
                        plans.push(serde_json::to_value(plan)?);
                        Ok(replay_quote_validated_research_v1(plan, evidence.clone())?)
                    },
                )
                .unwrap();
                assert!(!ledgers.is_empty());
                let observed = (plans, serde_json::to_value(ledgers).unwrap());
                if let Some(expected) = &expected {
                    assert_eq!(
                        &observed, expected,
                        "borrowed representation changed a decision or ledger"
                    );
                } else {
                    expected = Some(observed);
                }
            }
        }
    }
}

#[test]
fn borrowed_canonical_columns_reject_short_or_extra_history_before_any_quote_access() {
    let bars = bars();
    let timestamps: Vec<_> = bars.iter().map(|bar| bar.ts).collect();
    let gene = gene();
    for bad_length in [ROWS - 1, ROWS + 1] {
        let columns = neoethos_data::Ohlcv {
            timestamp: None,
            open: vec![1.0; ROWS],
            high: vec![1.0005; bad_length],
            low: vec![0.9995; ROWS],
            close: vec![1.0; ROWS],
            volume: None,
        };
        let lane = BorrowedSignalQuoteLane {
            bars: QuoteBars::Canonical {
                ohlcv: &columns,
                timestamps: &timestamps,
                symbol: "EURUSD",
                timeframe: "M1",
            },
            signals: &[1; ROWS],
            confidences: Some(&[0.25; ROWS]),
            entry_eligibility: None,
            entry_brackets: None,
            gene: &gene,
            pip_size: PIP,
            max_hold_bars: 2,
            trailing: None,
        };
        let error = drive_borrowed_lane::<QuoteValidatedResearchLedgerV1>(
            &lane,
            &binding(),
            &policy(1_000, 1_000),
            |_| panic!("invalid OHLC reached entry preview"),
            |_| panic!("invalid OHLC reached quote replay"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("mismatched OHLC column lengths"));
    }
}

#[test]
fn signals_open_real_quote_positions_and_hold_flat_or_opposite_until_their_time_exit() {
    let bars = bars();
    let gene = gene();
    let (bid, ask) = records();
    let lane = lane(&bars, &[1, -1, 1, -1, 0, 1, 1, 1], &gene, 2);
    let outcomes = run(&lane, &policy(1_000, 1_000), &evidence(bid, ask)).unwrap();
    assert_eq!(
        outcomes.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
        [0, 3, 6]
    );
    for (index, (_, ledger)) in outcomes.iter().enumerate() {
        assert_eq!(
            ledger.authority(),
            QuoteValidatedResearchAuthorityV1::UnverifiedCallerSuppliedQuotes,
            "a synthetic integration fixture must never manufacture a broker seal"
        );
        let position = &ledger.positions()[0];
        if index < 2 {
            assert_eq!(
                position.exit_reason(),
                Some(QuoteValidatedResearchExitReasonV1::MaxHold)
            );
            let expected = START + (3 + index as i64 * 3) * BAR + if index == 0 { 9 } else { 8 };
            assert_eq!(
                position.exit_reference().unwrap().timestamp_unix_ms(),
                expected
            );
        } else {
            assert!(
                position.exit_reference().is_none(),
                "terminal open position must not be liquidated synthetically"
            );
        }
    }
    assert_eq!(
        outcomes[1].1.positions()[0].direction(),
        ResearchPositionDirectionV1::Short
    );
}

#[test]
fn relative_gene_bracket_is_centered_on_the_modeled_quote_entry_not_last_bar_close() {
    let bars = bars();
    let gene = gene();
    let (mut bid, mut ask) = records();
    for row in bid.iter_mut().chain(ask.iter_mut()) {
        row.1 += 0.02;
    }
    let evidence = evidence(bid, ask);
    let mut plans = Vec::new();
    let outcomes = drive_lane(
        &lane(&bars, &[1, 0, 0, 0, 0, 0, 0, 0], &gene, 1),
        &binding(),
        &policy(1_000, 1_000),
        |plan| Ok(preview_quote_validated_research_entry_v1(plan, &evidence)?),
        |plan| {
            plans.push(serde_json::to_value(plan).unwrap());
            Ok(replay_quote_validated_research_v1(plan, evidence.clone())?)
        },
    )
    .unwrap();
    let position = &outcomes[0].1.positions()[0];
    assert!((position.modeled_entry_price() - 1.02025).abs() < 1e-12);
    assert!((plans[0]["decisions"][0]["stop_price"].as_f64().unwrap() - 1.01825).abs() < 1e-12);
    assert!((plans[0]["decisions"][0]["target_price"].as_f64().unwrap() - 1.03025).abs() < 1e-12);
}

#[test]
fn pending_non_entry_blocks_intervening_signals_until_its_deadline() {
    let bars = bars();
    let gene = gene();
    let (mut bid, mut ask) = records();
    bid.retain(|row| row.0 >= START + 3 * BAR);
    ask.retain(|row| row.0 >= START + 3 * BAR);
    let outcomes = run(
        &lane(&bars, &[1, 1, -1, 0, 0, 0, 0, 0], &gene, 1),
        &policy(90_000, 1_000),
        &evidence(bid, ask),
    )
    .unwrap();
    assert_eq!(
        outcomes.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(
        outcomes[0].1.entry_unavailable()[0].deadline_unix_ms(),
        START + BAR + 90_000
    );
    assert_eq!(
        outcomes[1].1.positions()[0].direction(),
        ResearchPositionDirectionV1::Short
    );
}

#[test]
fn entry_bar_pre_fill_extreme_cannot_arm_a_trailing_stop() {
    let mut bars = bars();
    bars[1].h = 1.5;
    let gene = gene();
    let (mut bid, ask) = records();
    bid.push((START + BAR + 2, 1.5));
    let mut lane = lane(&bars, &[1, 0, 0, 0, 0, 0, 0, 0], &gene, 2);
    lane.trailing = TrailingPolicy::new(1.0, 1.0, 2.0, PIP);
    let outcomes = run(&lane, &policy(1_000, 1_000), &evidence(bid, ask)).unwrap();
    assert_eq!(
        outcomes[0].1.positions()[0].exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::MaxHold)
    );
}

#[test]
fn actual_post_entry_excursion_protects_profit_before_the_far_target_for_long_and_short() {
    for direction in [1, -1] {
        let bars = bars();
        let gene = gene();
        let (mut bid, mut ask) = records();
        if direction == 1 {
            bid.push((START + BAR + 1_000, 1.0030));
            for row in &mut bid {
                if row.0 == START + 2 * BAR + 9 {
                    row.1 = 1.0012;
                }
            }
            bid.push((START + 2 * BAR + 1_000, 1.0008));
            bid.push((START + 2 * BAR + 1_010, 1.0008));
        } else {
            bid.push((START + BAR + 1_000, 0.9972));
            for row in &mut ask {
                if row.0 / BAR == (START + 2 * BAR) / BAR {
                    row.1 = 0.9990;
                }
            }
            ask.push((START + 2 * BAR + 1_000, 0.9994));
            ask.push((START + 2 * BAR + 1_010, 0.9994));
        }
        let signals = [direction, 0, 0, 0, 0, 0, 0, 0];
        let mut lane = lane(&bars, &signals, &gene, 6);
        lane.trailing = TrailingPolicy::new(1.0, 1.0, 2.0, PIP);
        let outcomes = run(&lane, &policy(1_000, 1_000), &evidence(bid, ask)).unwrap();
        let position = &outcomes[0].1.positions()[0];
        assert_eq!(
            position.exit_reason(),
            Some(QuoteValidatedResearchExitReasonV1::TrailingStop)
        );
        assert_eq!(
            position.exit_reference().unwrap().timestamp_unix_ms(),
            START + 2 * BAR + 1_010
        );
        let expected_exit = if direction == 1 { 1.00075 } else { 0.99945 };
        assert!((position.modeled_exit_price().unwrap() - expected_exit).abs() < 1e-12);
        assert!(
            (position.modeled_exit_price().unwrap() - position.modeled_entry_price())
                * f64::from(direction)
                > 0.0
        );
    }
}

#[test]
fn missing_quote_by_the_time_exit_deadline_is_an_error_not_a_late_or_ohlc_fill() {
    let bars = bars();
    let gene = gene();
    let (mut bid, ask) = records();
    for row in &mut bid {
        if row.0 == START + 2 * BAR + 9 {
            row.0 += 100;
        }
    }
    let error = run(
        &lane(&bars, &[1, 0, 0, 0, 0, 0, 0, 0], &gene, 1),
        &policy(1_000, 10),
        &evidence(bid, ask),
    )
    .unwrap_err();
    assert_eq!(
        error
            .downcast_ref::<QuoteValidatedResearchReplayErrorV1>()
            .unwrap()
            .code(),
        QuoteValidatedResearchReplayErrorCodeV1::ExitReferenceUnavailable
    );
}

#[test]
fn unsupported_or_invalid_lane_inputs_refuse_before_entry_or_quote_replay() {
    for mutation in 0..9 {
        let mut bars = bars();
        let mut gene = gene();
        let mut signals = [1; ROWS];
        match mutation {
            0 => gene.sl_pips = f64::NAN,
            1 => gene.tp_pips = 0.0,
            2 => gene.stop_vol_mult = 1.0,
            3 => signals[4] = 2,
            4 => bars[3].ts += 1,
            5 => bars[2].symbol = "USDJPY".to_owned(),
            6 => bars[2].h = 0.5,
            7 => bars[0].tf = "D1".to_owned(),
            _ => {}
        }
        let mut lane = lane(&bars, &signals, &gene, 2);
        if mutation == 8 {
            lane.pip_size = 0.01;
        }
        let error = drive_lane::<QuoteValidatedResearchLedgerV1>(
            &lane,
            &binding(),
            &policy(1_000, 1_000),
            |_| panic!("invalid lane reached entry preview"),
            |_| panic!("invalid lane reached quote execution"),
        )
        .unwrap_err();
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn bar_gap_is_not_filled_with_invented_time_or_prices() {
    let mut bars = bars();
    bars.remove(1);
    let gene = gene();
    let (bid, ask) = records();
    let error = run(
        &lane(&bars, &[1, 0, 0, 0, 0, 0, 0], &gene, 1),
        &policy(1_000, 1_000),
        &evidence(bid, ask),
    )
    .unwrap_err();
    assert!(error.to_string().contains("canonical bar gap"));
}

fn single_plan() -> QuoteValidatedResearchReplayPlanV1 {
    QuoteValidatedResearchReplayPlanV1::new(
        binding(),
        policy(1_000, 1_000),
        vec![
            CanonicalBarSignalResearchDecisionV1::new(
                START,
                START + BAR,
                ResearchPositionDirectionV1::Long,
                0.998,
                1.01,
            )
            .unwrap(),
        ],
        Vec::new(),
    )
    .unwrap()
}

#[test]
fn legacy_plan_wire_stays_unchanged_and_only_the_explicit_timer_closes_the_position() {
    let (bid, ask) = records();
    let evidence = evidence(bid, ask);
    let plan = single_plan();
    let old_wire = serde_json::to_value(&plan).unwrap();
    assert!(old_wire.get("time_exit").is_none());
    let decoded: QuoteValidatedResearchReplayPlanV1 = serde_json::from_value(old_wire).unwrap();
    let open = replay_quote_validated_research_v1(&decoded, evidence.clone()).unwrap();
    assert!(open.positions()[0].exit_reference().is_none());
    let timed = decoded
        .with_time_exit(ClosedCanonicalBarTimeExitV1::new(START + BAR, START + 2 * BAR).unwrap())
        .unwrap();
    let closed = replay_quote_validated_research_v1(&timed, evidence).unwrap();
    assert_eq!(
        closed.positions()[0].exit_reason(),
        Some(QuoteValidatedResearchExitReasonV1::MaxHold)
    );
    assert_ne!(open.ledger_sha256(), closed.ledger_sha256());
}

#[test]
fn price_trigger_wins_at_the_timer_but_not_after_it_and_fills_at_the_quote_not_threshold() {
    for (offset, expected) in [
        (0, QuoteValidatedResearchExitReasonV1::Stop),
        (1, QuoteValidatedResearchExitReasonV1::MaxHold),
    ] {
        let (mut bid, ask) = records();
        let timer = START + 2 * BAR;
        bid.push((timer + offset, 0.997));
        for quote in &mut bid {
            if quote.0 == timer + 9 {
                quote.1 = 0.9968;
            }
        }
        let timed = single_plan()
            .with_time_exit(ClosedCanonicalBarTimeExitV1::new(START + BAR, timer).unwrap())
            .unwrap();
        let ledger = replay_quote_validated_research_v1(&timed, evidence(bid, ask)).unwrap();
        let position = &ledger.positions()[0];
        assert_eq!(position.exit_reason(), Some(expected));
        assert_eq!(
            position.exit_reference().unwrap().timestamp_unix_ms(),
            timer + 9
        );
        assert!((position.modeled_exit_price().unwrap() - 0.99675).abs() < 1e-12);
    }
}

#[test]
fn raw_entry_preview_rejects_tampered_quotes_instead_of_treating_them_as_a_sealed_snapshot() {
    let (bid, ask) = records();
    let mut wire = serde_json::to_value(evidence(bid, ask)).unwrap();
    wire["bid"]["quote_records"][0]["price"] = serde_json::json!(1.00001);
    let tampered = serde_json::from_value(wire).unwrap();
    let error = match preview_quote_validated_research_entry_v1(&single_plan(), &tampered) {
        Ok(_) => panic!("modified row without its digest was accepted by raw entry preview"),
        Err(error) => error,
    };
    assert_eq!(
        error.code(),
        QuoteValidatedResearchReplayErrorCodeV1::ArtifactDigestMismatch
    );
}

#[test]
fn deserialized_timer_cannot_precede_entry_or_leave_the_locked_window() {
    let (bid, ask) = records();
    let evidence = evidence(bid, ask);
    for (source, effective) in [
        (START, START + BAR),
        (START + BAR, START + BAR),
        (START + ROWS as i64 * BAR, START + (ROWS as i64 + 1) * BAR),
    ] {
        let mut wire = serde_json::to_value(single_plan()).unwrap();
        wire["time_exit"] = serde_json::json!({
            "source_bar_open_unix_ms": source, "effective_at_next_bar_open_unix_ms": effective,
        });
        let plan = serde_json::from_value(wire).unwrap();
        let error = replay_quote_validated_research_v1(&plan, evidence.clone()).unwrap_err();
        assert_eq!(
            error.code(),
            QuoteValidatedResearchReplayErrorCodeV1::InvalidDecision
        );
    }
}

#[test]
fn zero_max_hold_disables_timer_without_fabricating_a_terminal_fill() {
    let bars = bars();
    let gene = gene();
    let (bid, ask) = records();
    let outcomes = run(
        &lane(&bars, &[1; ROWS], &gene, 0),
        &policy(1_000, 1_000),
        &evidence(bid, ask),
    )
    .unwrap();
    assert_eq!(
        outcomes.len(),
        1,
        "an open position owns its lane until the quote window ends"
    );
    let position = &outcomes[0].1.positions()[0];
    assert!(position.exit_reference().is_none());
    assert!(position.exit_reason().is_none());
}

#[test]
fn adaptive_decision_distances_skip_warmup_and_center_on_the_actual_quote_entry() {
    let bars = bars();
    let mut gene = gene();
    gene.stop_vol_mult = 1.5;
    let (bid, ask) = records();
    let evidence = evidence(bid, ask);
    let brackets = |row| if row == 0 { None } else { Some((30.0, 120.0)) };
    let eligible = |row| row > 0;
    let lane = BorrowedSignalQuoteLane {
        bars: QuoteBars::Live(&bars),
        signals: &[1, 1, 0, 0, 0, 0, 0, 0],
        confidences: Some(&[0.25; ROWS]),
        entry_eligibility: Some(&eligible),
        entry_brackets: Some(&brackets),
        gene: &gene,
        pip_size: PIP,
        max_hold_bars: 1,
        trailing: None,
    };
    let mut plans = Vec::new();
    let outcomes = drive_borrowed_lane(
        &lane,
        &binding(),
        &policy(1_000, 1_000),
        |plan| Ok(preview_quote_validated_research_entry_v1(plan, &evidence)?),
        |plan| {
            plans.push(serde_json::to_value(plan)?);
            Ok(replay_quote_validated_research_v1(plan, evidence.clone())?)
        },
    )
    .unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].0, 1);
    // Ask 1.0002 plus 0.5pip slippage; 30pip stop and 120pip target.
    assert!((outcomes[0].1.positions()[0].modeled_entry_price() - 1.00025).abs() < 1e-12);
    assert!((plans[0]["decisions"][0]["stop_price"].as_f64().unwrap() - 0.99725).abs() < 1e-12);
    assert!((plans[0]["decisions"][0]["target_price"].as_f64().unwrap() - 1.01225).abs() < 1e-12);
}
