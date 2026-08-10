//! **Non-negotiable 1, driven through the REAL runner: a crash loses ONE SWEEP,
//! not the session.**
//!
//! That property is already asserted at the fold level (`session.rs`) and at the
//! file level (`journal.rs`), and both of those are tests of a hand-built record
//! sequence. Neither of them ever ran the state machine. This one kills the
//! process from **inside `SweepExecutor::execute`**, where a real crash happens —
//! after `SweepStarted` was fsynced and before its outcome exists — and then
//! resumes the same session id and requires it to finish.
//!
//! The difference matters: a hand-built sequence proves the fold handles an open
//! sweep. Only this proves that the runner WRITES one, that the partial ledger
//! survives with real trials in it, that the resume path finds the session, and
//! that nothing already on disk is rewritten on the way through.

mod support;

use std::panic::AssertUnwindSafe;

use neoethos_autoresearch::runner::{RunArgs, run_with_executor};
use support::{ScriptedExecutor, tag_counts};

/// Die inside sweep 3, after three of its searches have completed: two sweeps
/// are already terminal, one is open, and its partial ledger is non-empty.
const DIE_FROM_SWEEP: u64 = 3;
const DIE_AFTER: usize = 3;

#[test]
fn a_crash_mid_sweep_loses_one_sweep_and_not_the_session() {
    let root = support::fresh_root("crash-resume");
    let settings = support::settings_in(&root);
    let base = support::base_config(&settings);

    // ── Phase 1 — the process dies inside a sweep ───────────────────────────
    //
    // The panic is expected, so the default hook's backtrace would only make the
    // test output look like a failure. Silenced for the duration and restored
    // immediately, because a swallowed panic anywhere else in this file would be
    // a test that cannot fail.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut dying = ScriptedExecutor::dying_in_sweep(DIE_FROM_SWEEP, DIE_AFTER);
    let died = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = run_with_executor(
            RunArgs {
                max_sweeps: 8,
                ..RunArgs::new("EURUSD")
            },
            &settings,
            base.clone(),
            &mut dying,
        );
    }));
    std::panic::set_hook(previous_hook);
    assert!(
        died.is_err(),
        "the scripted executor was asked to die inside sweep {DIE_FROM_SWEEP} and did not. \
         Without a real death there is no crash to recover from and this test proves nothing."
    );

    let dir = support::only_session_dir();
    let session_id = dir
        .file_name()
        .and_then(|n| n.to_str())
        .expect("the session directory names the session")
        .to_string();

    let before = support::journal_lines(&dir);
    let tags_before = tag_counts(&before);
    let count = |tags: &std::collections::BTreeMap<String, usize>, tag: &str| {
        tags.get(tag).copied().unwrap_or(0)
    };
    let terminal_before = count(&tags_before, "SweepCompleted")
        + count(&tags_before, "SweepFailed")
        + count(&tags_before, "SweepAbandoned");

    // The intent record reached the disk and its outcome did not. That asymmetry
    // IS two-phase journalling; without it a resume cannot tell that a sweep had
    // begun.
    assert_eq!(
        count(&tags_before, "SweepStarted"),
        terminal_before + 1,
        "exactly one sweep must be OPEN at the crash. tags: {tags_before:#?}"
    );
    assert!(
        terminal_before >= 2,
        "the crash must land after at least two terminal sweeps, or 'loses ONE sweep' is not \
         being tested. tags: {tags_before:#?}"
    );

    let open_sweep = support::records_tagged(&before, "SweepStarted")
        .last()
        .and_then(|r| r["sweep"].as_u64())
        .expect("the open sweep names itself");

    // The partial ledger — what makes `SweepAbandoned { trials_offered }` honest
    // rather than a zero.
    let ledger = dir
        .join("sweeps")
        .join(format!("sweep-{open_sweep}"))
        .join("partial.jsonl");
    assert!(
        ledger.exists(),
        "the open sweep {open_sweep} left no partial ledger at {}, so its trials can only be \
         recorded as zero — which UNDERSTATES N_session and therefore FLATTERS every deflated \
         Sharpe that follows it",
        ledger.display()
    );

    // ── Phase 2 — resume the SAME session ───────────────────────────────────
    let mut healthy = ScriptedExecutor::new();
    let verdict = run_with_executor(
        RunArgs {
            max_sweeps: 8,
            session: Some(session_id.clone()),
            ..RunArgs::new("EURUSD")
        },
        &settings,
        base,
        &mut healthy,
    )
    .expect("the resumed session must reach a verdict");

    let after = support::journal_lines(&dir);
    let tags_after = tag_counts(&after);
    let report = || {
        format!(
            "\n  session: {session_id}\n  before : {tags_before:#?}\n  after  : {tags_after:#?}"
        )
    };

    // ── NOT THE SESSION ─────────────────────────────────────────────────────
    //
    // Every record written before the crash is still there, unchanged, in order.
    // The journal is append-only and the resume path must not rewrite a byte of
    // it — a resume that re-derived the history would be free to derive a
    // different one.
    assert!(
        after.len() > before.len(),
        "the resumed session appended nothing{}",
        report()
    );
    assert_eq!(
        &after[..before.len()],
        &before[..],
        "the resume REWROTE part of the history. The journal is append-only: everything before \
         the crash happened, and a resume that can edit it can reach any conclusion it likes.{}",
        report()
    );
    assert_eq!(
        count(&tags_after, "SessionOpened"),
        1,
        "a resume must continue ONE history, not open a second frame inside it{}",
        report()
    );
    assert_eq!(
        support::only_session_dir(),
        dir,
        "the resume must reuse the session directory, not start a second one{}",
        report()
    );

    // ── ONE SWEEP ───────────────────────────────────────────────────────────
    let abandoned = support::records_tagged(&after, "SweepAbandoned");
    assert_eq!(
        abandoned.len(),
        1,
        "exactly one sweep is lost to a crash — the one that was open{}",
        report()
    );
    assert_eq!(
        abandoned[0]["sweep"].as_u64(),
        Some(open_sweep),
        "the abandoned sweep must be the one that was open at the crash{}",
        report()
    );

    // A TRIAL THAT RAN IS A TRIAL THAT RAN. The abandoned sweep's searches
    // completed; forgetting them would understate N_session and flatter every
    // deflated Sharpe that follows.
    let abandoned_trials = abandoned[0]["trials_offered"]
        .as_u64()
        .expect("the abandoned record carries its recovered trial count");
    assert!(
        abandoned_trials > 0,
        "the crashed sweep ran {DIE_AFTER} searches before it died and was recorded as offering \
         zero trials. That understates N_session, which FLATTERS every deflated Sharpe in the \
         rest of the session.{}",
        report()
    );
    assert_eq!(
        abandoned_trials as usize,
        DIE_AFTER * support::TRIALS,
        "the recovered count must be the partial ledger's own sum{}",
        report()
    );

    // ── it finished ─────────────────────────────────────────────────────────
    assert!(
        verdict.sweeps_run > terminal_before,
        "the resumed session ran no further sweeps{}",
        report()
    );
    assert_eq!(
        count(&tags_after, "SessionStopped"),
        1,
        "the resumed session must end with a verdict, in the same shape a clean run ends in{}",
        report()
    );
    assert!(
        dir.join("verdict.json").exists(),
        "verdict.json is missing after the resume{}",
        report()
    );

    // N_session across the crash: monotone, and equal to the fold over the
    // journal's terminal records. There is no code path that decrements it.
    // `ShuffleControlCompleted` is NOT in this list: a control IS a sweep and
    // its trials are already credited by its own `SweepCompleted`. The summary
    // record restates the same number for a reader, and summing it too counted
    // every control twice.
    let folded: usize = ["SweepCompleted", "SweepFailed", "SweepAbandoned"]
        .iter()
        .flat_map(|tag| support::records_tagged(&after, tag))
        .map(|r| r["trials_offered"].as_u64().unwrap_or(0) as usize)
        .sum();
    assert_eq!(
        verdict.n_trials_session,
        folded,
        "N_session is a fold over the journal and nothing else{}",
        report()
    );

    // The crash census is NAMED, not merely counted.
    assert_eq!(
        verdict.census.sweeps_abandoned, 1,
        "the abandon census must carry the crashed sweep{}",
        report()
    );
}
