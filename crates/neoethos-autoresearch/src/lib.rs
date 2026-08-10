//! # The autoresearch loop
//!
//! Implements `docs/autoresearch-loop.md`. This module has **no logic**: it
//! declares the module map, re-exports the public surface, and states the
//! invariants in one place so a reader does not have to reconstruct them from
//! twelve files.
//!
//! ## What it is
//!
//! A loop that proposes `(search configuration, objective)` pairs, runs them as
//! sweeps of `SWEEP_SEARCHES = 100` searches, judges each sweep against a
//! **frozen** judge, records every one in an append-only journal, and stops with
//! one of three verdicts. It optimises *toward* the operator's goal constants;
//! it can never redefine them.
//!
//! ## What it is not
//!
//! It never places an order, never contacts a broker, never writes
//! `live_portfolio.json`, never writes `config.yaml`. Its only outputs are a
//! proposal artifact and a verdict report, both under
//! `<store>/autoresearch/<session_id>/`. **The operator promotes.**
//!
//! ## The five non-negotiables and where each is enforced
//!
//! | # | Non-negotiable | Enforcement (mechanical, not procedural) |
//! |---|---|---|
//! | 1 | The loop may never write the goals, the cost model or the gates | [`goals::GoalSet`] has private fields and getters only; [`space::FROZEN_FIELDS`] plus a test asserting disjointness with everything `SearchConfigDelta::apply` writes; the CLI has no flag naming any of them (§14.2) |
//! | 2 | Every sweep recorded with its `config_hash`, its result and its verdict | [`journal`] is two-phase — [`journal::Record::SweepStarted`] is fsynced BEFORE any work, the outcome after; every proposal carries a `ResolvedConfigStamp` |
//! | 3 | Resumable — a crash loses one sweep, not the session | append-only JSONL, fsync per record, [`session::Session`] is a **pure fold** over the journal, a `SweepStarted` with no outcome becomes [`journal::Record::SweepAbandoned`] on resume |
//! | 4 | No silent drops | [`session::AbandonCensus`] in every report; `ProposalRefused` / `ProposalDuplicate` / `PosteriorRetracted` records |
//! | 5 | Never places an order, never touches the broker | the crate does not depend on `neoethos-app` or `neoethos-trader` (see `Cargo.toml`); its only writes are under the session directory |
//!
//! ## The invariant that carries the most weight
//!
//! > **The in-memory [`session::Session`] is a pure fold over the journal. No
//! > state exists anywhere else.**
//!
//! `N_session` — the honest trial count the deflated Sharpe deflates against —
//! is therefore [`session::n_session`], a fold, and never a stored counter.
//! There is no code path that decrements it and no field that holds it, which
//! makes *"a loop that resets its own N between sweeps"* structurally
//! impossible rather than forbidden by policy.
//!
//! ## Module ownership
//!
//! Three builders wrote this crate against `docs/autoresearch-loop.md` §2.
//! The boundary is by module, and it is worth stating because it is what makes
//! the seams reviewable:
//!
//! * **the loop** owns the session and the state machine — [`goals`],
//!   [`contracts`], [`journal`], [`session`], [`runner`];
//! * **the proposer** owns proposal and validation — [`space`], [`proposal`],
//!   [`proposer`], [`objective`];
//! * **the judge** owns statistics and stopping — [`judge`], [`shuffle`],
//!   [`verdict`].
//!
//! The exact symbols the loop consumes across that boundary are listed in
//! [`contracts`], which exists so that an absent symbol fails at COMPILE time
//! naming itself, and an absent behaviour fails at STARTUP naming itself —
//! never a silent no-op that makes the loop appear to run while doing nothing.

// ── the loop: session + state machine ───────────────────────────────────────
pub mod contracts;
pub mod goals;
pub mod journal;
pub mod runner;
pub mod session;

// ── the proposer: proposal + validation ─────────────────────────────────────
pub mod objective;
pub mod proposal;
pub mod proposer;
pub mod space;

// ── the judge: statistics + stopping ────────────────────────────────────────
pub mod judge;
pub mod shuffle;
pub mod verdict;

pub use goals::{GoalRefusal, GoalSet, Scenario, ScenarioKind};
pub use journal::{Journal, Record, SweepKind};
pub use runner::{RunArgs, SearchOutcome, SweepExecutor, run};
pub use session::{
    AbandonCensus, BestEver, BlockId, ChampionRow, CoverageCounters, Session, SessionHeader,
    SessionId, SessionStore, SweepId, n_session,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants — `docs/autoresearch-loop.md` §15, in ONE place.
//
// The judge's own thresholds are NOT here: they live in `judge::JudgeThresholds`
// because they are hashed into `judge_hash` and verified on every resume, and a
// constant that two modules can both spell is a constant that can drift.
// ─────────────────────────────────────────────────────────────────────────────

/// Searches per sweep. **The operator's own unit**: *"it does a SWEEP with 100
/// searches on the card, logs the good ones, and passes them through validation
/// ONCE."* All 100 are drawn before any of them runs, so a sweep is one
/// experiment with one `sweep_hash`, reproducible from `(session_seed,
/// sweep_index)` alone.
pub const SWEEP_SEARCHES: usize = 100;

/// Live sweeps per block. At each block boundary the block's best-screening
/// sweep is re-run against rotated features (§9).
pub const SHUFFLE_PERIOD: usize = 4;

/// Control sweeps per block. One per four live sweeps = **20% of the budget**,
/// which is defended plainly: the alternative is a session whose entire output
/// is unfalsifiable, and that is worth 0% of the budget.
pub const SHUFFLE_REPLICATES: usize = 1;

/// Blocks before the shuffle null is usable, and therefore before screen
/// conjunct S6 is available at all. Until then every reward is recorded as 0
/// and the posterior is not updated — the first three blocks are pure
/// exploration BY CONSTRUCTION, not by a tuned schedule.
pub const MIN_NULL_OBS: usize = 3;

/// Sweeps before the "provably unreachable" verdict (R2/U1) may even be
/// claimed. 40 sweeps = 4,000 searches ≥ 8 blocks ≥ 8 null observations.
pub const K_MIN: usize = 40;

/// Fraction of blocks that must be INDISTINGUISHABLE from their shuffle control
/// before R2/U3 is satisfied.
pub const U3_THRESHOLD: f64 = 0.75;

/// Default sweep budget = 20,000 searches.
pub const DEFAULT_MAX_SWEEPS: usize = 200;

/// Schema tag on every journal record. Bumping it starts a new session: two
/// journals with different schemas are not one history.
pub const JOURNAL_SCHEMA: &str = "neoethos.autoresearch.journal.v1";

/// Schema tag on `session.json`.
pub const SESSION_SCHEMA: &str = "neoethos.autoresearch.session.v1";

// ─────────────────────────────────────────────────────────────────────────────
// The one hash function this crate uses for its own identity values.
// ─────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64, rendered `fnv64:<16 hex>` — the SAME shape and the same algorithm
/// as `neoethos_search::run_identity`'s `config_hash`, so an operator reading a
/// `goal_hash` beside a `config_hash` is reading two values of one kind.
///
/// Used for `goal_hash`, `judge_hash`, `cost_hash` and `sweep_hash`. Not a
/// cryptographic claim and not asked to be one: these values answer *"is this
/// the same experiment?"*, and they are compared against a header the loop
/// itself wrote and fsynced.
pub fn fnv64(input: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    format!("fnv64:{hash:016x}")
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn fnv64_is_stable_and_prefixed() {
        // Pinned: a change here silently invalidates every session header on
        // disk, and the resume path would then refuse every existing session
        // with a hash mismatch it could not explain.
        assert_eq!(fnv64(""), "fnv64:cbf29ce484222325");
        assert!(fnv64("anything").starts_with("fnv64:"));
        assert_ne!(fnv64("a"), fnv64("b"));
    }

    #[test]
    fn the_budget_arithmetic_in_the_design_holds() {
        // §15: K_MIN sweeps = 4,000 searches, and at SHUFFLE_PERIOD = 4 that is
        // at least 8 blocks, hence at least 8 null observations — which is what
        // makes MIN_NULL_OBS reachable well before R2 may be claimed.
        assert_eq!(K_MIN * SWEEP_SEARCHES, 4_000);
        assert!(K_MIN / SHUFFLE_PERIOD >= MIN_NULL_OBS);
        // 1 control per 4 live sweeps = 20% of the budget.
        let control_share =
            SHUFFLE_REPLICATES as f64 / (SHUFFLE_PERIOD + SHUFFLE_REPLICATES) as f64;
        assert!((control_share - 0.20).abs() < 1e-12);
    }
}
