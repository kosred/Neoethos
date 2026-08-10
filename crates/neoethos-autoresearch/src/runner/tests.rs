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
