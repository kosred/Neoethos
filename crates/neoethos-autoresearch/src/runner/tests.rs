//! State-machine tests that do not need a card, a dataset or a feature cube.
//!
//! The end-to-end crash/resume behaviour is asserted in `session.rs`
//! (`a_crash_loses_one_sweep_not_the_session`,
//! `the_fold_equals_the_incremental_apply_after_every_record`) and in
//! `journal.rs` (`a_partial_final_line_is_truncated_and_counted_not_absorbed`),
//! because those are properties of the journal and the fold and can be proved
//! without a proposer or a judge. What is left here is the runner's own
//! arithmetic and its refusals.

use super::*;
use crate::journal::OosWindow;

fn base_config() -> DiscoveryConfig {
    DiscoveryConfig {
        evaluation_symbol: "EURUSD".to_string(),
        evaluation_spread_pips: 1.5,
        evaluation_commission_per_trade: 14.0,
        swap_long_pips_per_day: -0.4,
        swap_short_pips_per_day: 0.1,
        cost_band_pips: Some((2.0, 5.0)),
        min_trades_per_day: 0.5,
        ..DiscoveryConfig::default()
    }
}

#[test]
fn the_cost_hash_moves_when_the_cost_model_moves() {
    // A resumed session whose cost model moved is judging two sets of numbers
    // that do not mean the same thing. The hash is what catches it.
    let a = hash_named(&frozen_cost_fields(&base_config(), 10.0));
    let b = hash_named(&frozen_cost_fields(&base_config(), 10.0));
    assert_eq!(a, b);

    let mut moved = base_config();
    moved.cost_band_pips = Some((2.0, 6.0));
    assert_ne!(a, hash_named(&frozen_cost_fields(&moved, 10.0)));

    // And when only the pip value moves — the conversion every currency number
    // passes through — it must still move.
    assert_ne!(a, hash_named(&frozen_cost_fields(&base_config(), 9.0)));
}

#[test]
fn the_cost_hash_covers_the_validation_geometry_too() {
    // A loop that could widen its own CPCV folds could reach any number, so the
    // geometry is frozen alongside the cost model and hashed with it.
    let mut widened = base_config();
    widened.cpcv_n_splits += 1;
    assert_ne!(
        hash_named(&frozen_cost_fields(&base_config(), 10.0)),
        hash_named(&frozen_cost_fields(&widened, 10.0))
    );
}

#[test]
fn n_min_oos_is_derived_from_the_window_not_fixed() {
    // §15: derived from min_trades_per_month x months_in_window, so it means the
    // same thing on a two-month window and on a two-year one.
    const DAY: i64 = 86_400_000;
    let short = OosWindow {
        start_ms: 0,
        end_ms: 60 * DAY,
    };
    let long = OosWindow {
        start_ms: 0,
        end_ms: 600 * DAY,
    };
    let config = base_config();
    let n_short = ctx_n_min_oos(&config, short);
    let n_long = ctx_n_min_oos(&config, long);
    assert!(n_long > n_short, "{n_long} should exceed {n_short}");
    // 0.5 trades/day x 30 = 15 per month, x 2 months = 30.
    assert_eq!(n_short, 30);
}

#[test]
fn the_session_seed_is_a_function_of_the_session_id_alone() {
    // A proposal has to be replayable from (session_seed, sweep_index), and the
    // session id is what an operator has in front of them.
    let a = SessionId::parse("ar-20260810T101500Z-1a2b3c4d").unwrap();
    let b = SessionId::parse("ar-20260810T101500Z-1a2b3c4d").unwrap();
    let c = SessionId::parse("ar-20260810T101501Z-1a2b3c4d").unwrap();
    assert_eq!(session_seed_for(&a), session_seed_for(&b));
    assert_ne!(session_seed_for(&a), session_seed_for(&c));
}

#[test]
fn the_identity_source_says_out_loud_that_replicates_need_an_external_seed() {
    // §14.3: without `ga_seed` in `ResolvedConfigStamp`, two proposals differing
    // only in `replicate_seed` hash identically. The header must say so rather
    // than let the dedupe quietly eat the replicate dimension the judge needs.
    let source = identity_source();
    assert!(source.contains("replicate_seed"));
    assert!(source.contains("ga_seed"));
    assert!(source.contains("§14.3"));
}

#[test]
fn the_refusing_executor_refuses_rather_than_returning_an_empty_sweep() {
    // An empty outcome is indistinguishable from a sweep that found nothing,
    // which is exactly the silent no-op §13 exists to forbid.
    let mut executor = RefusingExecutor::new(
        (0, 800),
        OosWindow {
            start_ms: 801,
            end_ms: 1000,
        },
        1_000,
        100.0,
    );
    let config = base_config();
    let err = executor
        .execute(&SearchRequest {
            sweep: SweepId(1),
            slot: 0,
            config: &config,
            config_hash: "fnv64:test",
            trial_returns_path: std::path::PathBuf::from("unused"),
            permutation: None,
        })
        .expect_err("it must refuse");
    assert!(format!("{err:#}").contains("indistinguishable from a sweep that found nothing"));

    assert!(
        executor
            .evaluate_oos(SweepId(1), 0, &config)
            .is_err(),
        "it must never touch the out-of-sample window"
    );
    // But it does state its windows, because a silent zero there would read as
    // "no overlap".
    assert_eq!(executor.windows().unwrap().1.start_ms, 801);
}

#[test]
fn the_champion_is_the_best_pessimistic_edge_survivor() {
    let outcomes = vec![
        outcome_with(0, Some(0.10), "a"),
        outcome_with(1, Some(0.42), "b"),
        outcome_with(2, Some(-0.30), "c"),
        // An errored search never becomes the champion, however good its number
        // looks: the number came from a run that did not complete.
        SearchOutcome {
            error: Some("the card fell over".into()),
            ..outcome_with(3, Some(9.99), "d")
        },
    ];
    let champion = best_champion(SweepId(7), &outcomes).expect("a champion");
    assert_eq!(champion.strategy_id, "b");
    assert!((champion.e_screen_pess - 0.42).abs() < 1e-12);
    assert_eq!(champion.sweep, SweepId(7));
}

#[test]
fn a_sweep_with_no_survivors_contributes_no_champion_row() {
    // A champion row that is fabricated when nothing survived would put a
    // meaningless series into the session champion matrix and therefore into
    // `pbo_session` — the statistic that judges the loop's OWN selection.
    let outcomes = vec![SearchOutcome {
        champion_returns: Vec::new(),
        e_screen_pess: None,
        ..outcome_with(0, None, "")
    }];
    assert!(best_champion(SweepId(1), &outcomes).is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// A SLOT NAMES ITS PROPOSAL
//
// `per_search` used to DROP a row for every proposal the runner could not
// resolve, while `proposals` and the per-slot ledger directory stayed dense. The
// screen's position was then used against both — and refusals are the NORM, not
// an edge case: the proposer draws `streaming_batches` from {0,4,12,32} and the
// cursor from {Start, Resumed, RandomOffset}, and `Proposal::resolve` bails on
// every non-zero and every non-Start draw. The mismatch therefore fired
// routinely, and it landed on the single out-of-sample touch.
// ─────────────────────────────────────────────────────────────────────────────

fn caps() -> crate::space::Capabilities {
    crate::space::Capabilities::all_for_tests()
}

fn proposal_at(slot: usize, hash: &str) -> crate::proposal::Proposal {
    let delta =
        crate::space::SearchConfigDelta::new([0u8; crate::space::FACTOR_COUNT], &caps()).unwrap();
    let mut proposal = crate::proposal::build(
        delta,
        crate::objective::ObjectiveVariant::B5TerminalWealth,
        0,
        crate::objective::RefusalLevels::new([0u8; 4]).unwrap(),
        7,
        crate::proposal::ProposalOrigin::ExplorationFloor,
        slot,
        None,
        &caps(),
        crate::objective::ScenarioKind::Risky,
    )
    .expect("a buildable proposal");
    proposal.config_hash = hash.to_string();
    proposal
}

fn record_at(slot: usize, hash: &str, e: Option<f64>) -> SearchRecord {
    let mut outcome = outcome_with(slot, e, "x");
    outcome.config_hash = hash.to_string();
    outcome.to_record()
}

fn evidence_of(sweep: SweepId, per_search: Vec<SearchRecord>) -> SweepEvidence {
    let statistics = per_search
        .iter()
        .map(|_| TrialStatisticsReport::unreadable("test fixture".to_string()))
        .collect();
    SweepEvidence {
        sweep,
        kind: SweepKind::Live,
        sweep_hash: "fnv64:sweep".to_string(),
        per_search,
        statistics,
        champion: None,
        trials_offered: 0,
        wall_ms: 0,
    }
}

fn failed() -> crate::judge::ScreenResult {
    crate::judge::ScreenResult::Failed {
        conjunct: crate::judge::ScreenConjunct::S1Pbo,
        detail: "test fixture".to_string(),
    }
}

fn screen_of(
    per_slot: Vec<crate::judge::ScreenResult>,
    champion_position: Option<usize>,
) -> crate::judge::SweepScreen {
    crate::judge::SweepScreen {
        per_slot,
        champion: None,
        champion_slot: champion_position,
        champion_n_trades: 0,
        champion_dsr: None,
        champion_pbo: None,
    }
}

#[test]
fn a_refused_proposal_never_shifts_the_slot_that_names_the_out_of_sample_touch() {
    let proposals: Vec<_> = (0usize..5)
        .map(|s| proposal_at(s, &format!("fnv64:h{s}")))
        .collect();

    // The evidence as the judge sees it if slots 1 and 3 were refused — the
    // COMPACTED shape this defect lived in. Even in that shape, every row still
    // carries the slot that names its proposal, so the mapping is recoverable.
    let evidence = evidence_of(
        SweepId(3),
        vec![
            record_at(0, "fnv64:h0", Some(0.10)),
            record_at(2, "fnv64:h2", Some(0.90)),
            record_at(4, "fnv64:h4", Some(0.20)),
        ],
    );
    // The judge names POSITION 1 as its champion.
    let screen = screen_of(vec![failed(), failed(), failed()], Some(1));

    let aligned = align_screen(SweepId(3), &proposals, &evidence, &screen, &[]).unwrap();

    assert_eq!(
        aligned.rows.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        vec![0, 2, 4],
        "the Screened records must be keyed by the slot that names the proposal"
    );
    // Position 1 is SLOT 2. Indexing the dense proposals vector with the
    // position would have reached h1 — a configuration that was never screened.
    assert_eq!(aligned.champion_slot, Some(2));
    assert_eq!(aligned.credit_proposals[1].config_hash, "fnv64:h2");

    // And the promotion lookup, the one that spends the OOS window, resolves
    // the same configuration the screen actually judged.
    let promoted = proposal_named_by(SweepId(3), &proposals, 2, Some("fnv64:h2")).unwrap();
    assert_eq!(promoted.slot, 2);
    assert_eq!(promoted.config_hash, "fnv64:h2");
}

#[test]
fn two_configurations_wearing_one_slot_refuse_the_touch_instead_of_guessing() {
    let proposals: Vec<_> = (0usize..3)
        .map(|s| proposal_at(s, &format!("fnv64:h{s}")))
        .collect();

    let err = proposal_named_by(SweepId(1), &proposals, 1, Some("fnv64:h2")).unwrap_err();
    assert!(format!("{err:#}").contains("wearing one slot"), "{err:#}");

    // A slot nobody drew is named, never silently absent.
    let err = proposal_named_by(SweepId(1), &proposals, 9, None).unwrap_err();
    assert!(
        format!("{err:#}").contains("no proposal naming slot 9"),
        "{err:#}"
    );

    // The honest case still resolves.
    assert_eq!(
        proposal_named_by(SweepId(1), &proposals, 1, Some("fnv64:h1"))
            .unwrap()
            .slot,
        1
    );
}

#[test]
fn a_gap_in_the_draw_order_is_refused_before_a_bar_is_read() {
    // With a gap, `session.proposals_of()` back-fills the hole with a CLONE of a
    // neighbour, so a slot that was never drawn would answer with somebody
    // else's configuration.
    let dense: Vec<_> = (0usize..3).map(|s| proposal_at(s, "fnv64:h")).collect();
    assert!(assert_draw_order_is_dense(SweepId(1), &dense).is_ok());

    let gapped = vec![proposal_at(0, "fnv64:h"), proposal_at(2, "fnv64:h")];
    let err = assert_draw_order_is_dense(SweepId(1), &gapped).unwrap_err();
    assert!(format!("{err:#}").contains("names slot 2"), "{err:#}");
}

#[test]
fn a_slot_that_never_ran_is_named_kept_and_never_credited_to_the_posterior() {
    let proposals: Vec<_> = (0usize..3)
        .map(|s| proposal_at(s, &format!("fnv64:h{s}")))
        .collect();

    let refused = SearchOutcome::refused_before_running(
        1,
        "fnv64:h1",
        "the proposal carries a streaming plan, which is not a DiscoveryConfig field",
    );
    assert_eq!(refused.slot, 1);
    assert_eq!(refused.trials_offered, 0, "a row that never ran offers no trials");
    assert!(refused.e_screen_pess.is_none());
    assert!(!refused.cost_band.discriminates);
    assert!(
        refused.error.as_deref().unwrap().starts_with(NOT_RUN_PREFIX),
        "a census reader must be able to tell 'failed' from 'never happened'"
    );
    // It never becomes the champion, whatever else is in the sweep.
    assert!(best_champion(SweepId(1), std::slice::from_ref(&refused)).is_none());

    let evidence = evidence_of(
        SweepId(1),
        vec![
            record_at(0, "fnv64:h0", Some(0.10)),
            refused.to_record(),
            record_at(2, "fnv64:h2", Some(0.30)),
        ],
    );
    let screen = screen_of(vec![failed(), failed(), failed()], None);
    let aligned = align_screen(
        SweepId(1),
        &proposals,
        &evidence,
        &screen,
        &[(1, "not resolvable".to_string())],
    )
    .unwrap();

    // The row is KEPT, so the census sees it and no slot after it shifts…
    assert_eq!(aligned.rows.len(), 3);
    assert_eq!(
        aligned.rows.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    // …and the arm is not credited a failure it never earned.
    assert_eq!(aligned.credit_proposals.len(), 2);
    assert_eq!(aligned.credit_screens.len(), 2);
    assert!(aligned.credit_proposals.iter().all(|p| p.slot != 1));
}

#[test]
fn an_outcome_that_renames_its_slot_fails_the_sweep() {
    assert!(assert_outcome_names_its_request(SweepId(1), 7, 7).is_ok());
    let err = assert_outcome_names_its_request(SweepId(1), 7, 8).unwrap_err();
    assert!(format!("{err:#}").contains("out-of-sample"), "{err:#}");
}

#[test]
fn a_screen_that_does_not_cover_every_search_is_refused_rather_than_zipped() {
    // A shorter screen zipped against the searches would file every verdict
    // after the first gap against a search it did not judge.
    let proposals: Vec<_> = (0usize..2)
        .map(|s| proposal_at(s, &format!("fnv64:h{s}")))
        .collect();
    let evidence = evidence_of(
        SweepId(1),
        vec![
            record_at(0, "fnv64:h0", None),
            record_at(1, "fnv64:h1", None),
        ],
    );
    assert!(
        align_screen(
            SweepId(1),
            &proposals,
            &evidence,
            &screen_of(vec![failed()], None),
            &[]
        )
        .is_err()
    );

    // And a champion position with no search behind it is refused too: that
    // position is what the best-ever record and the promotion candidate are
    // keyed by.
    assert!(
        align_screen(
            SweepId(1),
            &proposals,
            &evidence,
            &screen_of(vec![failed(), failed()], Some(9)),
            &[]
        )
        .is_err()
    );
}

fn outcome_with(slot: usize, e: Option<f64>, id: &str) -> SearchOutcome {
    SearchOutcome {
        slot,
        config_hash: format!("fnv64:{id}"),
        trials_offered: 100,
        statistics: neoethos_search::deflated::TrialStatisticsReport::unreadable(
            "test fixture".to_string(),
        ),
        cost_band: CostBandCounts::default(),
        rejections: Vec::new(),
        survivors: 1,
        e_screen_pess: e,
        n_trades: 250,
        champion_returns: vec![0.01, -0.02, 0.03],
        champion_period_keys: vec![1, 2, 3],
        champion_strategy_id: id.to_string(),
        streamed: true,
        batch_columns: 512,
        next_cursor: 100,
        wall_ms: 10,
        error: None,
    }
}
