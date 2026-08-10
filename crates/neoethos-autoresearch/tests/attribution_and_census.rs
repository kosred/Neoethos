//! **The three unverified attributions and the census that reported zero.**
//!
//! C1, C2 and C3 were all the same shape: a number that belonged to one
//! configuration arriving under another configuration's name, with the
//! relabelling being what hid it. C4–C6 were the counting: a total that counted
//! some sweeps twice, and a refusal counter pinned at zero by a fold that only
//! incremented on success.
//!
//! Round 2 says each is now refused or counted. This file constructs the exact
//! mismatch each defect describes and demands a refusal — not a corrected
//! answer. A statistic that repairs a mismatch is worse than one that refuses
//! it, because a repair is invisible to the operator.
//!
//! These are all pure folds and pure judgements, so unlike the end-to-end tests
//! they cost milliseconds and do not touch the store. That matters: the loop's
//! own defence is that every verdict is re-derivable from `journal.jsonl`, and a
//! test that can only reach these paths through a six-minute session is a test
//! nobody runs while changing them.

mod support;

use neoethos_autoresearch::journal::{CostBandCounts, Record, SearchRecord, SweepKind};
use neoethos_autoresearch::judge::{JudgeThresholds, ScreenConjunct, ScreenResult, screen_sweep};
use neoethos_autoresearch::session::{BlockId, ChampionRow, Session, SweepEvidence, SweepId};
use neoethos_autoresearch::shuffle::{ControlKind, ShuffleNull};
use neoethos_search::deflated::analyse_bytes;

use support::{Evidence, TRIALS, matrix_for};

/// The identity of the slot that FAILS the screen while carrying the highest
/// expectancy in the sweep — the overfit outlier.
const OUTLIER: &str = "fnv64:the-overfit-outlier";
/// The identity of the slot that actually passes.
const SURVIVOR: &str = "fnv64:the-slot-that-passed";

fn thresholds() -> JudgeThresholds {
    // The two DERIVED floors, at values this fixture's 400 trades clear. They
    // are set explicitly rather than left at `frozen()`'s placeholders because
    // the whole point of §15 is that the enforced floor is the derived one.
    JudgeThresholds::frozen().with_derived_floors(30, 30)
}

/// A null the survivor's +0.30 beats and the fixture's controls sit well under.
fn null() -> ShuffleNull {
    let mut null = ShuffleNull::new();
    for e in [-0.05, -0.04, -0.06] {
        null.observe(ControlKind::CircularRotation, Some(e));
    }
    assert!(
        null.quantile_95().is_some(),
        "the null must be AVAILABLE or S6 returns Unavailable and nothing can pass"
    );
    null
}

fn band(discriminates: bool) -> CostBandCounts {
    CostBandCounts {
        survives: 1,
        optimistic_edge_only: 0,
        fails: TRIALS - 1,
        unmeasured: 0,
        not_discriminating: 0,
        discriminates,
    }
}

fn record(slot: usize, config_hash: &str, e: f64, discriminates: bool) -> SearchRecord {
    SearchRecord {
        slot,
        config_hash: config_hash.to_string(),
        trials_offered: TRIALS,
        survivors: 1,
        e_screen_pess: Some(e),
        n_trades: 400,
        dsr: None,
        pbo: None,
        dsr_refusal: None,
        pbo_refusal: None,
        cost_band: band(discriminates),
        rejections: Vec::new(),
        streamed: false,
        batch_columns: 0,
        next_cursor: 0,
        wall_ms: 1,
        error: None,
    }
}

/// A sweep of exactly two slots:
///
/// * slot 0 — the OUTLIER. Strong statistics and the **highest** expectancy in
///   the sweep, but its cost band cannot discriminate, so it fails S3. This is
///   the routine case, not an edge case: the highest-expectancy configuration
///   in a sweep is typically the one the screen is there to reject.
/// * slot 1 — the SURVIVOR. Lower expectancy, passes every conjunct.
///
/// The old runner took the argmax over ALL non-errored slots (slot 0), the judge
/// took the argmax over PASSING slots (slot 1), and then the row was stamped
/// with slot 1's identity while keeping slot 0's monthly returns.
fn two_slot_sweep(survivor_row_hash: &str) -> (SweepEvidence, Vec<f64>, Vec<f64>) {
    let sweep = SweepId(1);
    let (outlier_bytes, outlier_series) = matrix_for(Evidence::Strong, 11);
    let (survivor_bytes, survivor_series) = matrix_for(Evidence::Strong, 22);
    let period_keys: Vec<i64> = (0..survivor_series.len() as i64).map(|i| 24_300 + i).collect();

    let evidence = SweepEvidence {
        sweep,
        kind: SweepKind::Live,
        sweep_hash: "fnv64:two-slot".to_string(),
        per_search: vec![
            record(0, OUTLIER, 0.90, false),
            record(1, SURVIVOR, 0.30, true),
        ],
        statistics: vec![
            analyse_bytes(&outlier_bytes, TRIALS),
            analyse_bytes(&survivor_bytes, TRIALS),
        ],
        champion_rows: vec![
            Some(ChampionRow {
                sweep,
                config_hash: OUTLIER.to_string(),
                strategy_id: "outlier".to_string(),
                period_keys: period_keys.clone(),
                monthly_returns: outlier_series.clone(),
                e_screen_pess: 0.90,
            }),
            Some(ChampionRow {
                sweep,
                // Parameterised so the same builder can produce the HONEST case
                // and the case where the row and the record disagree.
                config_hash: survivor_row_hash.to_string(),
                strategy_id: "survivor".to_string(),
                period_keys,
                monthly_returns: survivor_series.clone(),
                e_screen_pess: 0.30,
            }),
        ],
        trials_offered: TRIALS,
        wall_ms: 1,
    };
    (evidence, outlier_series, survivor_series)
}

// ─────────────────────────────────────────────────────────────────────────────
// C2 — two argmaxes, one row
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_champion_row_comes_from_the_slot_that_passed_and_never_from_the_outlier() {
    let (evidence, outlier_series, survivor_series) = two_slot_sweep(SURVIVOR);
    let screen = screen_sweep(&evidence, &thresholds(), TRIALS, &null());

    // The premise: the two argmaxes really do disagree here. Without this the
    // test would pass on a sweep where the defect could not have shown.
    assert!(
        matches!(
            screen.per_slot[0],
            ScreenResult::Failed {
                conjunct: ScreenConjunct::S3CostBand,
                ..
            }
        ),
        "the outlier must FAIL the screen while holding the highest expectancy, or this test \
         asserts nothing: {:?}",
        screen.per_slot[0]
    );
    assert!(
        matches!(screen.per_slot[1], ScreenResult::Passed { .. }),
        "the survivor must pass: {:?}",
        screen.per_slot[1]
    );

    let champion = screen
        .champion
        .as_ref()
        .expect("a sweep with a passing slot contributes a row");
    assert_eq!(
        screen.champion_slot,
        Some(1),
        "the row must be taken from the PASSING slot"
    );
    assert_eq!(
        champion.config_hash, SURVIVOR,
        "the row must carry the passing slot's identity"
    );

    // THE ASSERTION THE DEFECT WOULD FAIL.
    //
    // The old path kept the OUTLIER's monthly returns and overwrote only the
    // hash and the expectancy, so this is the one check the relabelling could
    // not survive — and the one nothing checked. `pbo_session` is computed from
    // exactly these series.
    assert_eq!(
        champion.monthly_returns, survivor_series,
        "the champion row carries a return series that is not the passing slot's own. The \
         identity was re-stamped and the SERIES came from somewhere else — which is precisely \
         what makes pbo_session, the statistic gating every promotion, a measurement of a \
         configuration the screen rejected."
    );
    assert_ne!(
        champion.monthly_returns, outlier_series,
        "the champion row is the REJECTED outlier's series wearing the survivor's name"
    );
    assert_eq!(champion.e_screen_pess, 0.30);
}

#[test]
fn a_row_whose_identity_disagrees_with_its_slot_is_refused_and_never_relabelled() {
    // The row materialised for slot 1 names a DIFFERENT configuration. The two
    // vectors are index-aligned by contract, so this is a wiring fault — and the
    // only safe answer is that the sweep contributes NO row.
    let (evidence, _, _) = two_slot_sweep("fnv64:some-other-configuration");
    let screen = screen_sweep(&evidence, &thresholds(), TRIALS, &null());

    assert!(
        matches!(screen.per_slot[1], ScreenResult::Passed { .. }),
        "the slot still passes the screen — the disagreement is about the ROW"
    );
    assert!(
        screen.champion.is_none(),
        "a row whose identity disagrees with the slot it is aligned to must NOT enter the session \
         champion matrix under either name"
    );
    let refusal = screen
        .champion_refusal
        .as_ref()
        .expect("a sweep that drops out of pbo_session's input must say WHY");
    assert!(
        refusal.contains(SURVIVOR) && refusal.contains("some-other-configuration"),
        "the refusal must name BOTH identities so the operator can see which two disagreed: \
         {refusal}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// C3 — a gap in the screen vector
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn a_gap_in_the_screen_vector_is_filled_with_a_refusal_and_never_with_a_pass() {
    // Only slot 3 was screened. Slots 0–2 have NO screen at all. Filling them
    // with a clone of the arriving result made each of them a PASS carrying slot
    // 3's statistics — and `promotion_candidate` selects out of exactly this
    // vector, so the fabricated passes competed for the one irreplaceable
    // out-of-sample touch.
    let sweep = SweepId(1);
    let session = Session::fold(&[Record::Screened {
        sweep,
        slot: 3,
        screen_result: ScreenResult::Passed {
            e_screen_pess: 0.30,
            dsr: 0.99,
            pbo_sweep: 0.05,
            excess_over_expected_max_per_period: 0.10,
            n_trades: 400,
            q_shuffle_95: -0.05,
        },
        failing_conjunct: None,
    }])
    .expect("folding a single Screened record");

    let screens = session
        .screens_of(sweep)
        .expect("the sweep has a screen vector");
    assert_eq!(screens.len(), 4, "the vector is keyed by SLOT: 0..=3");
    for (slot, result) in screens.iter().enumerate().take(3) {
        match result {
            ScreenResult::Unavailable { detail } => {
                assert!(
                    !detail.trim().is_empty(),
                    "slot {slot}'s gap is unnamed. A refusal with no reason is the silent drop \
                     wearing a refusal's costume."
                );
            }
            other => panic!(
                "slot {slot} was NEVER SCREENED and the fold recorded {other:?}. A gap is not a \
                 repeat: this is slot 3's verdict fabricated for a configuration that was never \
                 judged, and it is selectable for the one out-of-sample touch."
            ),
        }
    }
    assert!(
        matches!(screens[3], ScreenResult::Passed { .. }),
        "slot 3's own result must survive intact"
    );

    // And the fabricated slots must not be promotable. There is no search
    // record for any of them, so a candidate naming one cannot be identified —
    // which must be a NAMED refusal, never `unwrap_or_default()` handing the
    // touch an empty config_hash.
    match neoethos_autoresearch::verdict::promotion_candidate(&session) {
        Ok(None) => {}
        Ok(Some(candidate)) => assert!(
            !candidate.config_hash.is_empty(),
            "a promotion candidate was selected with an EMPTY config_hash — the one \
             irreplaceable touch spent on a configuration nobody can name"
        ),
        Err(detail) => assert!(
            !detail.trim().is_empty(),
            "the refusal must name what it refused"
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C4 / C5 — the census arithmetic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn a_control_sweeps_trials_are_counted_once_and_a_control_with_no_number_is_counted_as_a_refusal() {
    let control = SweepId(2);
    let records = vec![
        Record::SweepStarted {
            sweep: SweepId(1),
            sweep_hash: "fnv64:live".to_string(),
            kind: SweepKind::Live,
            started_ms: 0,
        },
        Record::SweepCompleted {
            sweep: SweepId(1),
            trials_offered: 100,
            per_search: Vec::new(),
            wall_ms: 1,
        },
        Record::SweepStarted {
            sweep: control,
            sweep_hash: "fnv64:control".to_string(),
            kind: SweepKind::Control {
                control: ControlKind::CircularRotation,
                source_sweep: SweepId(1),
                block: BlockId(1),
            },
            started_ms: 0,
        },
        // The control IS a sweep: this is where its trials are credited.
        Record::SweepCompleted {
            sweep: control,
            trials_offered: 25,
            per_search: Vec::new(),
            wall_ms: 1,
        },
        // …and this record RESTATES the same 25 for a human reader. Summing it
        // too is what inflated N by one control per block.
        Record::ShuffleControlCompleted {
            block: BlockId(1),
            kind: ControlKind::CircularRotation,
            tau: 0.5,
            source_sweep: SweepId(1),
            trials_offered: 25,
            // A control that produced NO number.
            e_screen_pess: None,
            dsr: None,
        },
    ];
    let session = Session::fold(&records).expect("folding");

    assert_eq!(
        session.n_session(),
        125,
        "N_session must be 100 + 25. Crediting ShuffleControlCompleted as well as the control's \
         own SweepCompleted counted every control TWICE — a headline N no other artifact can \
         reproduce."
    );

    // C5 — the control produced nothing, and that is COUNTED.
    assert_eq!(
        session.null_refusals, 1,
        "a shuffle control that produced no E_screen_pess must join the null as a REFUSAL. The \
         fold only pushed when the value was Some, so this counter was structurally zero under \
         every headline the loop printed and a session whose controls were all broken read as a \
         session with a clean null."
    );
    let null = ShuffleNull::from_session(&session);
    assert_eq!(
        null.refused(),
        1,
        "the refusal must survive the rebuild that S6 actually consults"
    );

    // …and it is NAMED, not merely counted.
    let rendered = session.census.render();
    assert!(
        rendered.contains("shuffle control") || rendered.contains("REFUSAL"),
        "the refusal is counted but never named, so nobody can act on it:\n{rendered}"
    );
}

#[test]
fn garbage_collection_is_journalled_and_the_census_accumulates_every_pass() {
    use neoethos_autoresearch::session::GcCensus;

    // The two counters that were rendered by every verdict and assigned by
    // nobody: there was no Record variant for a GC pass, so the fold had
    // nothing to fold and the number was structurally zero while the session
    // deleted every matrix it had.
    let pass = |sweep: u64, deleted: usize, bytes: u64| Record::GarbageCollected {
        sweep: SweepId(sweep),
        census: GcCensus {
            kept: 1,
            deleted,
            bytes_reclaimed: bytes,
            rule: "trial matrices outside the keep-set are collected after every Continue"
                .to_string(),
            free_bytes_after: None,
        },
    };
    let session =
        Session::fold(&[pass(1, 90, 4_096), pass(2, 7, 8_192)]).expect("folding two GC passes");

    assert_eq!(
        session.census.gc_deleted_matrices, 97,
        "the census is a SESSION TOTAL: a GC that ran twice must not report only the second pass"
    );
    assert_eq!(session.census.gc_bytes_reclaimed, 12_288);
    assert!(
        session.census.render().contains("keep-set"),
        "the RULE the GC applied must be named once, not left as a bare count:\n{}",
        session.census.render()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The census's own honesty: every abandoned configuration NAMED
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn no_abandoned_configuration_is_counted_under_an_empty_name() {
    // §11.2: counted AND named. A census entry with an empty rule, or an empty
    // example beside it, is a drop that has learned to render.
    let records = vec![
        Record::ProposalRefused {
            sweep: SweepId(1),
            slot: 0,
            config_hash: "fnv64:aaaa".to_string(),
            reason: "the payoff floor is unreachable under this proposal's own geometry"
                .to_string(),
        },
        Record::SweepAbandoned {
            sweep: SweepId(1),
            trials_offered: 40,
            reason: "the operator stopped the run".to_string(),
        },
    ];
    // `SweepAbandoned` closes an OPEN sweep, so the intent record must precede
    // it — the fold is total and refuses an outcome with no intent.
    let mut all = vec![Record::SweepStarted {
        sweep: SweepId(1),
        sweep_hash: "fnv64:live".to_string(),
        kind: SweepKind::Live,
        started_ms: 0,
    }];
    all.extend(records);
    let session = Session::fold(&all).expect("folding");

    assert!(
        !session.census.examples.is_empty(),
        "two configurations left the loop without a result and the census named NEITHER"
    );
    for (rule, config_hash) in &session.census.examples {
        assert!(
            !rule.trim().is_empty(),
            "a census entry is counted under an EMPTY rule, beside {config_hash:?}. A drop that \
             renders as a blank line is still a silent drop."
        );
        assert!(
            !config_hash.trim().is_empty(),
            "the census names rule {rule:?} with an EMPTY example beside it. A counter with no \
             example is a number nobody can act on."
        );
    }
}
