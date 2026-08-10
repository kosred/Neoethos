//! **The irreplaceable resource, proven rather than described.**
//!
//! Round 2 claims the promotion path no longer spends the single out-of-sample
//! touch on a path guaranteed to fail. The claim is easy to make in a comment
//! and the comments are excellent; this file is the part that cannot be written
//! by asserting it. Each scenario damages the promotion evidence in one named
//! way and then demands, from the JOURNAL and not from internal state, that:
//!
//! 1. `OosTouchSpent` is **absent** — the window is still clean for a resume;
//! 2. the executor's `evaluate_oos` was **never called** — no bar was read;
//! 3. the session still reached a **verdict**, so the run produced an artifact;
//! 4. the refusal is **NAMED**, and names the configuration it refused.
//!
//! Claim 4 is why this is not one assertion. A session that refuses correctly
//! but reports `Inconclusive { reason: "" }` has moved the defect rather than
//! fixed it: the operator learns the run produced nothing and cannot learn why.
//!
//! ## Why all three scenarios live in ONE `#[test]`
//!
//! `support::settings_in` sets `NEOETHOS_USER_DATA_DIR`, which
//! `config::user_config_path` reads on EVERY call — it is not cached. Two tests
//! in one binary would therefore race for the store root and each would read
//! the other's sessions. Cargo gives each integration-test FILE its own process
//! but runs the tests inside a file on threads, so the scenarios are sequential
//! statements in a single test rather than three `#[test]` functions.

mod support;

use support::{Sabotage, ScriptedExecutor};

use neoethos_autoresearch::runner::{RunArgs, run_with_executor};
use neoethos_autoresearch::verdict::Verdict;

/// Same budget as the end-to-end test: the first PASSING screen is what drives
/// S9 to `Promote`, and without a promotion decision none of this is exercised.
/// A scenario that never reached the promotion path would pass every assertion
/// below vacuously, so each one re-checks that a screen did pass.
const MAX_SWEEPS: usize = 24;

#[test]
fn a_promotion_that_cannot_succeed_refuses_by_name_and_leaves_the_window_unspent() {
    // ── the three ways the evidence is wrong ────────────────────────────────
    //
    // Each is a real provenance, not a contrived corruption:
    //   * OmitEvidence  — THE ORIGINAL DEFECT. One site read `survivors.json`;
    //                     nothing in the workspace wrote it.
    //   * ForeignStamp  — a scratch root keyed by SYMBOL while sweep ids restart
    //                     at 1 per session: session B's sweep-1/slot-007 file IS
    //                     session A's.
    //   * EmptyGenes    — a search that selected nothing, written honestly.
    for (scenario, sabotage, must_name) in [
        ("missing", Sabotage::OmitEvidence, "NOT spent"),
        ("foreign-stamp", Sabotage::ForeignStamp, "stamped"),
        ("empty-genes", Sabotage::EmptyGenes, "no genes"),
    ] {
        let root = support::fresh_root(&format!("oos-unspent-{scenario}"));
        let settings = support::settings_in(&root);
        let base = support::base_config(&settings);

        let mut executor = ScriptedExecutor::sabotaging(sabotage);
        let verdict = run_with_executor(
            RunArgs {
                max_sweeps: MAX_SWEEPS,
                ..RunArgs::new("EURUSD")
            },
            &settings,
            base,
            &mut executor,
        )
        .unwrap_or_else(|e| {
            panic!(
                "[{scenario}] the loop returned Err instead of a verdict: {e:#}\n\nAn Err out of \
                 `run` is the multi-hour session that ends with no SessionStopped and no \
                 verdict.json. Unusable promotion evidence must STOP the session with a named \
                 refusal, not lose it."
            )
        });

        let dir = support::only_session_dir();
        let lines = support::journal_lines(&dir);
        let tags = support::tag_counts(&lines);
        let census = || {
            format!(
                "\n  scenario : {scenario}\n  session  : {}\n  tags     : {tags:#?}\n  verdict  : \
                 {}",
                dir.display(),
                verdict.render()
            )
        };

        // The scenario is only meaningful if it REACHED the promotion path.
        assert!(
            !support::passed_screens(&lines).is_empty(),
            "[{scenario}] no screen passed, so S9 never decided Promote and this scenario \
             asserted nothing. The window being unspent would be vacuous.{}",
            census()
        );

        // ── 1. THE WINDOW IS NOT SPENT ──────────────────────────────────────
        assert_eq!(
            tags.get("OosTouchSpent").copied().unwrap_or(0),
            0,
            "[{scenario}] the single out-of-sample touch was journalled as SPENT on a promotion \
             that could not possibly succeed. This is the irreplaceable resource: once the record \
             is on disk, `oos_spent()` bails on every resume and no later build can ever evaluate \
             this window out of sample again.{}",
            census()
        );

        // ── 2. NO BAR WAS READ ──────────────────────────────────────────────
        assert!(
            executor.oos_calls.is_empty(),
            "[{scenario}] the executor's evaluate_oos was called {} times. The journal record and \
             the actual read must agree — a window that was read but not journalled is worse than \
             one that was journalled but not read.{}",
            executor.oos_calls.len(),
            census()
        );
        assert!(
            verdict.oos.is_none(),
            "[{scenario}] the verdict carries a promotion outcome for a touch that never \
             happened{}",
            census()
        );

        // ── 3. THE SESSION STILL PRODUCED AN ARTIFACT ───────────────────────
        assert_eq!(
            tags.get("SessionStopped").copied().unwrap_or(0),
            1,
            "[{scenario}] the session did not journal a verdict{}",
            census()
        );
        assert!(
            dir.join("verdict.json").exists(),
            "[{scenario}] verdict.json is what the operator reads when the run is over{}",
            census()
        );

        // ── 4. THE REFUSAL IS NAMED ─────────────────────────────────────────
        //
        // Not merely non-empty: it must name the failure AND the artifact, so
        // the operator can act. "Inconclusive" with an empty reason is the
        // silent drop this crate's §11.2 forbids, wearing a verdict's costume.
        let Verdict::Inconclusive { reason } = &verdict.verdict else {
            panic!(
                "[{scenario}] expected an Inconclusive verdict naming the refusal, got {:?}{}",
                verdict.verdict.tag(),
                census()
            )
        };
        assert!(
            reason.contains(must_name),
            "[{scenario}] the refusal does not name why it refused — it must contain \
             {must_name:?}. A counter with no example beside it is a number nobody can act on, \
             and so is a verdict with no cause.\n  reason: {reason}{}",
            census()
        );
        assert!(
            reason.contains("slot"),
            "[{scenario}] the refusal does not name WHICH configuration it refused.\n  reason: \
             {reason}{}",
            census()
        );

        // ── and the window really is resumable ──────────────────────────────
        //
        // The whole point of "not spent" is that a later run can still use it.
        // Asserted against the folded session rather than the record count, so
        // this checks the thing `promote` actually gates on.
        let journal = neoethos_autoresearch::journal::Journal::open(dir.join("journal.jsonl"))
            .unwrap_or_else(|e| panic!("[{scenario}] reopening the journal: {e:#}"));
        let session = neoethos_autoresearch::session::Session::fold(journal.records())
            .unwrap_or_else(|e| panic!("[{scenario}] folding the journal: {e:#}"));
        assert!(
            !session.oos_spent(),
            "[{scenario}] the folded session believes the window is spent, so a resume would bail \
             even though no bar was ever read{}",
            census()
        );
    }
}
