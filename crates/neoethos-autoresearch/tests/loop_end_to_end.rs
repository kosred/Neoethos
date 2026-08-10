//! **The loop, driven end to end to a verdict — the test that did not exist.**
//!
//! `runner::run` was invoked from NOWHERE in the workspace: no CLI subcommand,
//! no UI route, no caller outside the crate. The only two `SweepExecutor`
//! implementations were `RefusingExecutor`, which refuses by design, and
//! `StreamingSweepExecutor`, which needs a card, a dataset and a feature cube.
//! So the state machine had never once been driven from `SessionOpened` to
//! `SessionStopped`, and every defect that survived two adversarial reviews
//! lived in exactly that stretch.
//!
//! This file closes that. It asserts on the **journal**, not on internal state,
//! because the journal is the artifact — the loop's own claim is that every
//! verdict it reaches is re-derivable from `journal.jsonl` alone.
//!
//! ## It is slow, and that is not a bug
//!
//! §11.4 re-folds the WHOLE journal after every appended record (a debug
//! assertion, and the one that caught round 1's worst defect), so a session of a
//! few thousand records costs O(n²) folds. Sixteen sweeps is also the SHORTEST
//! run in which a pass is even possible: `MIN_NULL_OBS = 3` control observations
//! are needed before screen conjunct S6 can be evaluated at all, and a control
//! runs once per `SHUFFLE_PERIOD = 4` live sweeps. Expect minutes, not seconds.

mod support;

use support::{ScriptedExecutor, tag_counts};

use neoethos_autoresearch::runner::{RunArgs, run_with_executor};

/// Sixteen sweeps reach the first possible pass (12 live + 3 controls, then the
/// live sweep that finally has a null to beat). The extra budget is headroom, not
/// expectation: the loop stops itself at the first passing screen.
const MAX_SWEEPS: usize = 24;

#[test]
fn the_loop_runs_to_a_verdict_and_the_journal_proves_every_claim_in_it() {
    let root = support::fresh_root("end-to-end");
    let settings = support::settings_in(&root);
    let base = support::base_config(&settings);

    let mut executor = ScriptedExecutor::new();
    let verdict = run_with_executor(
        RunArgs {
            max_sweeps: MAX_SWEEPS,
            ..RunArgs::new("EURUSD")
        },
        &settings,
        base,
        &mut executor,
    )
    .expect(
        "the loop must reach a VERDICT. An abort that produces no verdict.json is the multi-hour \
         run that ended with no artifact — the failure mode this crate exists to prevent, and a \
         session that can report neither 'goal reached' nor 'goal unreachable' has no output at \
         all.",
    );

    let dir = support::only_session_dir();
    let lines = support::journal_lines(&dir);
    let tags = tag_counts(&lines);
    let census = || {
        format!(
            "\n  session   : {}\n  tags      : {tags:#?}\n  verdict   : {}\n  screen failures: \
             {:#?}\n  matrices left on disk: {:#?}",
            dir.display(),
            verdict.render(),
            support::distinct_screen_failures(&lines),
            support::matrices_on_disk(&dir),
        )
    };

    // ── the session has ONE history and ONE ending ──────────────────────────
    assert_eq!(
        tags.get("SessionOpened").copied().unwrap_or(0),
        1,
        "a session has exactly one frame — one goal set, one judge, one OOS window{}",
        census()
    );
    assert_eq!(
        tags.get("SessionStopped").copied().unwrap_or(0),
        1,
        "the verdict must be journalled, not only returned{}",
        census()
    );
    assert!(
        dir.join("verdict.json").exists(),
        "verdict.json is the artifact an operator reads when the run is over{}",
        census()
    );

    // ── AT LEAST ONE PASSING SCREEN ─────────────────────────────────────────
    //
    // This is the assertion that catches an unsatisfiable screen contract. The
    // second adversary drove 71 searches through this loop and ZERO passed:
    // every one failed S2 with "deflated against trials_offered = 40 but the
    // session-wide trial count is 1080", because the executor deflates against
    // its OWN manifest while the judge demands the session fold. No pass means
    // no champion, which means `best_ever` stays None forever, which means R2's
    // U2 condition can never be satisfied — the loop can report NEITHER goal
    // reached NOR goal unreachable.
    let passed = support::passed_screens(&lines);
    assert!(
        !passed.is_empty(),
        "NOT ONE SCREEN PASSED in {} sweeps. A loop that can never pass its own in-sample screen \
         can never record a champion, can never advance best_ever, and can therefore report \
         neither GOAL REACHED nor GOAL UNREACHABLE. The distinct failures below name why.{}",
        verdict.sweeps_run,
        census()
    );

    // ── the champion and the best-ever pointer, both FOLDED FROM THE JOURNAL ─
    assert!(
        tags.get("ChampionRecorded").copied().unwrap_or(0) >= 1,
        "a passing screen must contribute its row to the session champion matrix — the input to \
         pbo_session, which gates every promotion. A matrix that restarts empty makes pbo_session \
         refuse forever.{}",
        census()
    );
    assert!(
        tags.get("BestEverAdvanced").copied().unwrap_or(0) >= 1,
        "the session's best screened result must be JOURNALLED. Held outside the journal it is \
         lost on every restart, and with None on the left of R2's U2 comparison the loop can no \
         longer report the goal unreachable.{}",
        census()
    );
    assert!(
        verdict.best_ever.is_some(),
        "the verdict must carry the best-ever result it was folded from{}",
        census()
    );
    assert!(
        dir.join("session_champions.json").exists(),
        "the champion matrix is written beside the journal{}",
        census()
    );

    // ── THE GC CENSUS IS NON-EMPTY AND AGREES WITH THE JOURNAL ──────────────
    //
    // `AbandonCensus::gc_deleted_matrices` and `gc_bytes_reclaimed` are rendered
    // by every verdict and were ASSIGNED BY NOBODY: there was no `Record`
    // variant for a garbage-collection pass, so the fold had nothing to fold and
    // the operator was told zero matrices had been deleted by a session that had
    // deleted all of them.
    let passes = support::records_tagged(&lines, "GarbageCollected");
    assert!(
        !passes.is_empty(),
        "the GC ran on every Continue decision and journalled nothing{}",
        census()
    );
    let journalled_deleted: usize = passes
        .iter()
        .map(|r| {
            r["census"]["deleted"]
                .as_u64()
                .expect("a GC census carries its deleted count") as usize
        })
        .sum();
    let journalled_bytes: u64 = passes
        .iter()
        .map(|r| {
            r["census"]["bytes_reclaimed"]
                .as_u64()
                .expect("a GC census carries its reclaimed bytes")
        })
        .sum();
    assert!(
        journalled_deleted > 0,
        "the session ran {} sweeps and the GC deleted nothing. Either the keep-set is keeping \
         everything, or the pass is not running.{}",
        verdict.sweeps_run,
        census()
    );
    assert_eq!(
        verdict.census.gc_deleted_matrices, journalled_deleted,
        "the number the verdict PRINTS must be the number the journal RECORDS. They were 0 and \
         'all of them' respectively.{}",
        census()
    );
    assert_eq!(
        verdict.census.gc_bytes_reclaimed, journalled_bytes,
        "same for the reclaimed bytes{}",
        census()
    );

    // ── the honest N is a fold, and it never resets ─────────────────────────
    // `ShuffleControlCompleted` is NOT in this list: a control IS a sweep and
    // its trials are already credited by its own `SweepCompleted`. The summary
    // record restates the same number for a reader; summing it too counted
    // every control twice, inflating N by about one sweep per block.
    let folded: usize = ["SweepCompleted", "SweepFailed", "SweepAbandoned"]
        .iter()
        .flat_map(|tag| support::records_tagged(&lines, tag))
        .map(|r| r["trials_offered"].as_u64().unwrap_or(0) as usize)
        .sum();
    assert_eq!(
        verdict.n_trials_session, folded,
        "N_session is defined as a fold over the journal's terminal sweep records and nothing \
         else{}",
        census()
    );
    assert!(
        verdict.n_trials_session > 0,
        "a session that offered no trials judged nothing{}",
        census()
    );

    // ── the single out-of-sample touch ──────────────────────────────────────
    assert!(
        tags.get("OosTouchSpent").copied().unwrap_or(0) <= 1,
        "there is ONE out-of-sample touch per session, ever{}",
        census()
    );
    assert!(
        executor.oos_calls.len() <= 1,
        "the executor was asked to evaluate the OOS window {} times{}",
        executor.oos_calls.len(),
        census()
    );
    // A passing screen makes S9 decide PROMOTE, so this session must have spent
    // it — and stopped either way.
    assert_eq!(
        tags.get("OosTouchSpent").copied().unwrap_or(0),
        1,
        "a passing screen must be carried to the out-of-sample window exactly once{}",
        census()
    );

    // ── non-negotiable 5, observed rather than asserted in prose ────────────
    let stray = support::files_named(&root, "live_portfolio.json");
    assert!(
        stray.is_empty(),
        "the loop PROPOSES; the operator promotes. It wrote {stray:?}{}",
        census()
    );

    // Everything it did write is under its own session directory.
    for artifact in ["journal.jsonl", "session.json", "verdict.json"] {
        assert!(
            dir.join(artifact).exists(),
            "{artifact} is missing from the session store{}",
            census()
        );
    }
}
