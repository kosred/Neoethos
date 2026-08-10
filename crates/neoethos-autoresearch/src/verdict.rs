//! **The one report shape, and both stopping rules.**
//!
//! `docs/autoresearch-loop.md` §10. Two facts drive every decision here:
//!
//! 1. **A success and a failure produce the same struct and the same
//!    renderer.** Not "a similar struct". The same one. A failure report that
//!    looks different from a success report is a failure report nobody reads,
//!    and the thing this project has never been able to say is *"no
//!    configuration in this space reaches the goal"*. It is said here, in the
//!    same format, with the same fields filled in, as a success.
//!
//! 2. **"Unreachable" is only ever a claim about a SEARCHED space.** U4 exists
//!    because a space is not refuted if it was not searched, and it is written
//!    so that an EMPTY coverage summary FAILS it rather than passing it
//!    vacuously — the classic "for all x in {}" trap, which here would let a
//!    loop that drew nothing declare the goal impossible.
//!
//! [`decide`] is a pure function of the folded session, so S9 has no hidden
//! state and the stop can be re-derived from the journal by a reader.

use serde::{Deserialize, Serialize};

use crate::goals::{GoalSet, Scenario};
use crate::journal::{OosWindow, RiskOutcomeRecord};
use crate::judge::{JudgeThresholds, PromotionOutcome, ScreenResult};
use crate::session::{AbandonCensus, BestEver, Session, SweepId};
use crate::shuffle::{ShuffleNull, ShuffleSummary};

/// Schema tag on every emitted verdict. Bump only with the struct.
pub const VERDICT_SCHEMA: &str = "neoethos.autoresearch.verdict.v1";

/// U3: the fraction of blocks that must be indistinguishable from their shuffle
/// control before the space is called refuted.
pub const U3_THRESHOLD: f64 = 0.75;

// ─────────────────────────────────────────────────────────────────────────────
// The promotion candidate
// ─────────────────────────────────────────────────────────────────────────────

/// A sweep slot that passed the whole in-sample screen — the only thing the one
/// out-of-sample touch may ever be spent on.
///
/// Every statistic the promotion inequality re-checks travels on the candidate,
/// so `judge::promote` re-derives nothing and cannot silently read a different
/// sweep's numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionCandidate {
    pub sweep: SweepId,
    pub slot: usize,
    pub config_hash: String,
    pub e_screen_pess: f64,
    pub dsr: Option<f64>,
    pub pbo_sweep: f64,
    pub excess_over_expected_max: f64,
    /// Filled in AFTER the out-of-sample evaluation, so the verdict can report
    /// the goal metric and the whole frontier.
    ///
    /// `None` means the numbers exist only in the journal's `Promoted` record.
    /// The renderer says so rather than printing a hollow "GOAL REACHED" with no
    /// number beside it.
    #[serde(default)]
    pub outcome: Option<PromotionOutcome>,
}

impl PromotionCandidate {
    /// Attach the out-of-sample evaluation before stopping.
    pub fn with_outcome(mut self, outcome: PromotionOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// R2 — the goal is provably unreachable in this space
// ─────────────────────────────────────────────────────────────────────────────

/// One of the four R2 conditions, with the numbers that decided it.
///
/// Carried in the report whether or not R2 fired, because §10.3 requires an
/// inconclusive verdict to say *which* of U1–U4 were not met. A stop that cannot
/// name what was missing is the sixteen-month failure mode again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnreachableCondition {
    pub id: String,
    pub satisfied: bool,
    pub detail: String,
}

/// The full R2 evaluation. `satisfied` is the conjunction of U1..U4.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnreachableCheck {
    pub satisfied: bool,
    pub conditions: Vec<UnreachableCondition>,
}

impl UnreachableCheck {
    /// The conditions that did NOT hold.
    pub fn unmet(&self) -> Vec<&UnreachableCondition> {
        self.conditions.iter().filter(|c| !c.satisfied).collect()
    }

    fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "R2 — is the goal provably unreachable in the space that was searched? {}",
            if self.satisfied { "YES" } else { "no" }
        );
        for c in &self.conditions {
            let _ = writeln!(
                s,
                "  [{}] {}: {}",
                if c.satisfied { "x" } else { " " },
                c.id,
                c.detail
            );
        }
        s
    }
}

/// Evaluate U1–U4, each checkable, each with its numbers.
///
/// **Every condition is evaluated even after one fails.** A short-circuit would
/// save nothing and would rob the inconclusive report of exactly the information
/// §10.3 requires it to carry.
pub fn check_unreachable(session: &Session, _th: &JudgeThresholds) -> UnreachableCheck {
    let mut conditions = Vec::with_capacity(4);
    let live = session.live_sweeps_run();

    // U1 — enough sweeps that the null itself means something.
    conditions.push(UnreachableCondition {
        id: "U1".to_string(),
        satisfied: live >= crate::K_MIN,
        detail: format!(
            "{live} live sweep(s); K_MIN = {} (= {} searches, >= {} block(s))",
            crate::K_MIN,
            crate::K_MIN * crate::SWEEP_SEARCHES,
            crate::K_MIN / crate::SHUFFLE_PERIOD.max(1)
        ),
    });

    // U2 — the loop never once beat its own noise.
    //
    // A missing number on EITHER side leaves U2 unsatisfied. Without a null
    // there is nothing to fail to beat; without a best there is nothing that
    // failed to beat it. In both cases "we never beat noise" is unsupported, and
    // an unsupported claim is not a refutation.
    let null = ShuffleNull::from_session(session);
    let best = session.best_ever.as_ref().map(|b| b.e_screen_pess);
    conditions.push(match (best, null.quantile_95()) {
        (Some(b), Some(q)) => UnreachableCondition {
            id: "U2".to_string(),
            satisfied: b <= q,
            detail: format!(
                "best live E_screen_pess {b:+.4} pips vs Q_0.95(shuffle null) {q:+.4} pips"
            ),
        },
        (None, Some(q)) => UnreachableCondition {
            id: "U2".to_string(),
            satisfied: false,
            detail: format!(
                "no sweep ever produced a screened E_screen_pess, so there is nothing to compare \
                 against Q_0.95 = {q:+.4}. That is a wiring or data finding, NOT evidence that \
                 the space holds no edge."
            ),
        },
        (_, None) => UnreachableCondition {
            id: "U2".to_string(),
            satisfied: false,
            detail: format!(
                "the shuffle null holds {} observation(s) and needs {}, so 'never beat its own \
                 noise' is not yet a statement that can be made.",
                null.len(),
                crate::MIN_NULL_OBS
            ),
        },
    });

    // U3 — the blocks were mostly indistinguishable from their controls.
    let blocks = session.blocks.len();
    let indist = session.indistinguishable_blocks();
    let frac = if blocks == 0 {
        0.0
    } else {
        indist as f64 / blocks as f64
    };
    conditions.push(UnreachableCondition {
        id: "U3".to_string(),
        satisfied: blocks > 0 && frac >= U3_THRESHOLD,
        detail: format!(
            "{indist}/{blocks} block(s) indistinguishable from their shuffle control = {:.0}%; \
             threshold {:.0}%",
            frac * 100.0,
            U3_THRESHOLD * 100.0
        ),
    });

    // U4 — the space was actually searched.
    //
    // The empty check is FIRST and is not folded into the `under_covered` calls:
    // those return an empty vector when nothing was declared, and "no level is
    // under-drawn" would then be indistinguishable from full coverage. A loop
    // that drew nothing must never be able to declare a space refuted.
    conditions.push(if session.coverage.is_empty() {
        UnreachableCondition {
            id: "U4".to_string(),
            satisfied: false,
            detail: "no factor level was ever credited, so the coverage counters are empty. A \
                     space is not refuted if it was not searched, and an empty coverage summary \
                     satisfies 'every level was drawn' only vacuously."
                .to_string(),
        }
    } else {
        let b_under = session
            .coverage
            .under_covered("objective", crate::B_MIN_DRAWS);
        let a_under: Vec<(String, String, usize)> = session
            .coverage
            .iter()
            .filter(|(f, _, _)| *f != "objective")
            .filter(|(_, _, c)| c.sweeps < 1)
            .map(|(f, l, c)| (f.to_string(), l.to_string(), c.sweeps))
            .collect();
        let satisfied = b_under.is_empty() && a_under.is_empty();
        UnreachableCondition {
            id: "U4".to_string(),
            satisfied,
            detail: if satisfied {
                format!(
                    "every credited axis-A level was drawn at least once and every axis-B variant \
                     at least {} sweep(s)' worth",
                    crate::B_MIN_DRAWS
                )
            } else {
                format!(
                    "under-drawn — axis B (< {} sweeps): [{}]; axis A (< 1 sweep): [{}]",
                    crate::B_MIN_DRAWS,
                    b_under
                        .iter()
                        .map(|(l, c)| format!("{l} ({})", c.sweeps))
                        .collect::<Vec<_>>()
                        .join(", "),
                    a_under
                        .iter()
                        .map(|(f, l, n)| format!("{f}={l} ({n})"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            },
        }
    });

    UnreachableCheck {
        satisfied: conditions.iter().all(|c| c.satisfied),
        conditions,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// S9 — DECIDE
// ─────────────────────────────────────────────────────────────────────────────

/// What the state machine does next.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Draw another sweep.
    Continue,
    /// Spend the one out-of-sample touch on this candidate, then stop whatever
    /// the outcome.
    ///
    /// R1 is `∃ s : PROMOTE(s)` and `PROMOTE` has out-of-sample conjuncts, so R1
    /// cannot be evaluated without spending the touch. This is the loop asking
    /// permission from its own budget, not a claim of success.
    Promote(PromotionCandidate),
    Stop(StopReason),
}

/// Why a session stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stop", rename_all = "snake_case")]
pub enum StopReason {
    /// R1 — the promotion inequality held out of sample, once.
    GoalReached(PromotionCandidate),
    /// The touch was spent and the inequality did not hold. STOP either way:
    /// the out-of-sample window is spent.
    PromotionRefused {
        sweep: SweepId,
        failing_conjunct: String,
    },
    /// R2/U5 — the reachable space is enumerated. An independent terminator, and
    /// also a "this space was searched" result.
    ProposerExhausted,
    /// R2/U1–U4.
    GoalUnreachable,
    BudgetExhausted {
        sweeps_run: usize,
        max_sweeps: usize,
    },
    WallClock {
        hours: f64,
    },
    OperatorStop,
}

/// **S9 — DECIDE.** Pure, so the stop is re-derivable from the journal.
///
/// Order, and why:
///
/// 1. **R1 before R2.** A sweep that passed the whole in-sample screen deserves
///    its out-of-sample test even if U1–U4 have also come true; refusing to
///    spend the touch on a survivor and reporting "unreachable" instead would be
///    the loop declining to check its own best answer.
/// 2. R2.
/// 3. Otherwise continue — the wall-clock and sweep budgets belong to the
///    runner, which owns the clock.
pub fn decide(session: &Session, _goals: &GoalSet, th: &JudgeThresholds) -> Decision {
    if let Some(candidate) = promotion_candidate(session)
        && session.oos_touches_spent < th.oos_touches_total
    {
        return Decision::Promote(candidate);
    }
    if check_unreachable(session, th).satisfied {
        return Decision::Stop(StopReason::GoalUnreachable);
    }
    Decision::Continue
}

/// The best PASSING screen in the whole session, or `None`.
///
/// Only a `ScreenResult::Passed` qualifies: `Failed` and `Unavailable` are both
/// failures, and a candidate drawn from either would spend the one out-of-sample
/// touch on something the in-sample judge already refused.
pub fn promotion_candidate(session: &Session) -> Option<PromotionCandidate> {
    let mut best: Option<PromotionCandidate> = None;
    for sweep in &session.sweeps {
        let Some(screens) = session.screens_of(sweep.sweep) else {
            continue;
        };
        for (slot, result) in screens.iter().enumerate() {
            let ScreenResult::Passed {
                e_screen_pess,
                dsr,
                pbo_sweep,
                excess_over_expected_max_per_period,
                ..
            } = result
            else {
                continue;
            };
            if best
                .as_ref()
                .map(|b| *e_screen_pess > b.e_screen_pess)
                .unwrap_or(true)
            {
                best = Some(PromotionCandidate {
                    sweep: sweep.sweep,
                    slot,
                    config_hash: sweep
                        .per_search
                        .get(slot)
                        .map(|p| p.config_hash.clone())
                        .unwrap_or_default(),
                    e_screen_pess: *e_screen_pess,
                    dsr: Some(*dsr),
                    pbo_sweep: *pbo_sweep,
                    excess_over_expected_max: *excess_over_expected_max_per_period,
                    outcome: None,
                });
            }
        }
    }
    best
}

// ─────────────────────────────────────────────────────────────────────────────
// The verdict
// ─────────────────────────────────────────────────────────────────────────────

/// The three ways a session can end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// R1.
    GoalReached {
        sweep: SweepId,
        slot: usize,
        goal_metric: Option<f64>,
        goal_bar: Option<f64>,
        frontier: Vec<RiskOutcomeRecord>,
    },
    /// R2. The result this project has never been able to state.
    GoalUnreachableInSearchedSpace {
        distinct_configurations: usize,
        /// Whether U5 (proposer exhaustion) rather than U1–U4 terminated it.
        proposer_exhausted: bool,
    },
    /// Neither. Explicitly NOT evidence that no edge exists.
    Inconclusive { reason: String },
}

impl Verdict {
    pub fn tag(&self) -> &'static str {
        match self {
            Self::GoalReached { .. } => "GOAL_REACHED",
            Self::GoalUnreachableInSearchedSpace { .. } => "GOAL_UNREACHABLE_IN_SEARCHED_SPACE",
            Self::Inconclusive { .. } => "INCONCLUSIVE",
        }
    }

    /// The headline line. One sentence, the same shape for all three.
    pub fn headline(&self, unmet: usize) -> String {
        match self {
            Self::GoalReached {
                sweep,
                slot,
                goal_metric,
                goal_bar,
                ..
            } => match (goal_metric, goal_bar) {
                (Some(m), Some(b)) => format!(
                    "GOAL REACHED — {sweep} slot {slot} cleared the promotion inequality out of \
                     sample: goal metric {m:.4} >= bar {b:.4}"
                ),
                _ => format!(
                    "GOAL REACHED — {sweep} slot {slot} cleared the promotion inequality out of \
                     sample. THE GOAL METRIC IS NOT ATTACHED TO THIS VERDICT: read it from the \
                     journal's Promoted record. A headline without its number is not a result."
                ),
            },
            Self::GoalUnreachableInSearchedSpace {
                distinct_configurations,
                proposer_exhausted,
            } => format!(
                "GOAL UNREACHABLE IN THE SEARCHED SPACE — {distinct_configurations} distinct \
                 configuration(s) were run and none ever beat its own shuffle control{}.",
                if *proposer_exhausted {
                    "; the proposer then exhausted the reachable space"
                } else {
                    ""
                }
            ),
            Self::Inconclusive { reason } => format!(
                "INCONCLUSIVE ({reason}) — this is NOT evidence that no edge exists: {unmet} of \
                 the R2 conditions were not met."
            ),
        }
    }
}

/// Everything the runner resolved that is not in the session fold.
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictContext {
    pub session_id: String,
    pub symbol: String,
    pub cost_hash: String,
    pub judge_hash: String,
    pub oos_window: OosWindow,
}

/// **The report. One struct, one renderer, three verdicts.**
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionVerdict {
    pub schema: String,
    pub verdict: Verdict,
    pub session_id: String,
    pub symbol: String,
    /// Resume invalidation, and the proof that the goals were not edited
    /// mid-session.
    pub goal_hash: String,
    pub judge_hash: String,
    pub cost_hash: String,
    /// The goal, BY FIELD PATH, so a reader can check that the loop optimised
    /// toward the operator's constants and not toward something it invented.
    pub scenario_label: String,
    pub scenario_fields: Vec<(String, String)>,
    pub scenarios_source: String,
    pub oos_window: OosWindow,
    pub sweeps_run: usize,
    pub live_sweeps_run: usize,
    pub searches_run: usize,
    /// The honest N — every trial the session ever offered, live plus control
    /// plus failed plus abandoned. Never reset, because it is a fold over the
    /// journal.
    pub n_trials_session: usize,
    pub best_ever: Option<BestEver>,
    /// §9.4. ALWAYS present, and rendered immediately under the headline.
    pub shuffle: ShuffleSummary,
    /// U1–U4 with their numbers, in every verdict.
    pub unreachable: UnreachableCheck,
    /// Per `(factor, level)`: `(sweeps, draws)`.
    pub coverage: Vec<(String, String, usize, usize)>,
    pub census: AbandonCensus,
    /// Present exactly when the out-of-sample touch was spent.
    pub oos: Option<PromotionOutcome>,
    /// The exact command that re-derives this verdict.
    pub reproduction: String,
}

/// Build the report from the stopped session.
pub fn build(
    session: &Session,
    goals: &GoalSet,
    scenario: &Scenario,
    th: &JudgeThresholds,
    reason: StopReason,
    ctx: VerdictContext,
) -> SessionVerdict {
    let unreachable = check_unreachable(session, th);
    let distinct = session.config_hashes.len();

    let (verdict, oos) = match reason {
        StopReason::GoalReached(candidate) => {
            let outcome = candidate.outcome.clone();
            (
                Verdict::GoalReached {
                    sweep: candidate.sweep,
                    slot: candidate.slot,
                    goal_metric: outcome.as_ref().map(|o| o.goal_metric),
                    goal_bar: outcome.as_ref().map(|o| o.goal_bar),
                    frontier: outcome
                        .as_ref()
                        .map(|o| o.frontier.clone())
                        .unwrap_or_default(),
                },
                outcome,
            )
        }
        StopReason::PromotionRefused {
            sweep,
            failing_conjunct,
        } => (
            // The touch is spent and the inequality did not hold. If U1–U4 also
            // hold, that IS the refutation; otherwise it is inconclusive — and
            // saying so is the difference between "we proved there is nothing
            // here" and "we ran out of window".
            if unreachable.satisfied {
                Verdict::GoalUnreachableInSearchedSpace {
                    distinct_configurations: distinct,
                    proposer_exhausted: false,
                }
            } else {
                Verdict::Inconclusive {
                    reason: format!(
                        "the single out-of-sample touch was spent on {sweep} and promotion was \
                         refused at {failing_conjunct}"
                    ),
                }
            },
            None,
        ),
        StopReason::ProposerExhausted => (
            Verdict::GoalUnreachableInSearchedSpace {
                distinct_configurations: distinct,
                proposer_exhausted: true,
            },
            None,
        ),
        StopReason::GoalUnreachable => (
            Verdict::GoalUnreachableInSearchedSpace {
                distinct_configurations: distinct,
                proposer_exhausted: false,
            },
            None,
        ),
        StopReason::BudgetExhausted {
            sweeps_run,
            max_sweeps,
        } => (
            Verdict::Inconclusive {
                reason: format!("the sweep budget ran out at {sweeps_run}/{max_sweeps}"),
            },
            None,
        ),
        StopReason::WallClock { hours } => (
            Verdict::Inconclusive {
                reason: format!("the wall-clock budget of {hours:.1} h ran out"),
            },
            None,
        ),
        StopReason::OperatorStop => (
            Verdict::Inconclusive {
                reason: "the operator stopped the session".to_string(),
            },
            None,
        ),
    };

    SessionVerdict {
        schema: VERDICT_SCHEMA.to_string(),
        verdict,
        session_id: ctx.session_id.clone(),
        symbol: ctx.symbol,
        goal_hash: goals.goal_hash().to_string(),
        judge_hash: ctx.judge_hash,
        cost_hash: ctx.cost_hash,
        scenario_label: scenario.label().to_string(),
        scenario_fields: goals
            .field_values()
            .iter()
            .map(|(p, v)| ((*p).to_string(), v.clone()))
            .collect(),
        scenarios_source: goals.scenarios_source().to_string(),
        oos_window: ctx.oos_window,
        sweeps_run: session.sweeps_run(),
        live_sweeps_run: session.live_sweeps_run(),
        searches_run: session.searches_run(),
        n_trials_session: session.n_session(),
        best_ever: session.best_ever.clone(),
        shuffle: ShuffleSummary::from_session(session),
        unreachable,
        coverage: session
            .coverage
            .iter()
            .map(|(f, l, c)| (f.to_string(), l.to_string(), c.sweeps, c.draws))
            .collect(),
        census: session.census.clone(),
        oos,
        reproduction: format!(
            "neoethos autoresearch --symbol {} --session {}",
            ctx.symbol_for_reproduction(),
            ctx.session_id
        ),
    }
}

impl VerdictContext {
    fn symbol_for_reproduction(&self) -> &str {
        &self.symbol
    }
}

impl SessionVerdict {
    /// **The only renderer.** Success, refutation and inconclusive all come out
    /// of this function, in this order, with the same sections.
    ///
    /// The `ShuffleSummary` is emitted immediately under the headline in every
    /// case. That placement is the mechanical form of *"if the loop's winners
    /// score like the shuffle's winners it must say so in the same breath as its
    /// best result"* — rendering it last, or only on success, would let the one
    /// falsifying measurement be skipped by a reader who stopped at the number
    /// they wanted.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();

        let _ = writeln!(s, "{}", "=".repeat(72));
        let _ = writeln!(
            s,
            "{}",
            self.verdict.headline(self.unreachable.unmet().len())
        );
        let _ = writeln!(s, "{}", "=".repeat(72));

        // IMMEDIATELY under the headline. Not a section further down.
        let _ = writeln!(s, "{}", self.shuffle.render());

        let _ = writeln!(s, "SESSION");
        let _ = writeln!(s, "  session   : {}  ({})", self.session_id, self.symbol);
        let _ = writeln!(s, "  goal_hash : {}", self.goal_hash);
        let _ = writeln!(s, "  judge_hash: {}", self.judge_hash);
        let _ = writeln!(s, "  cost_hash : {}", self.cost_hash);
        let _ = writeln!(
            s,
            "  sweeps {} ({} live) / searches {} / N_session {}",
            self.sweeps_run, self.live_sweeps_run, self.searches_run, self.n_trials_session
        );
        let _ = writeln!(
            s,
            "  N_session is the trial count the Deflated Sharpe was deflated against — every \
             trial the session offered, live and control and failed and abandoned."
        );

        let _ = writeln!(s, "\nGOAL (read-only, by field path)");
        let _ = writeln!(s, "  scenario  : {}", self.scenario_label);
        for (path, value) in &self.scenario_fields {
            let _ = writeln!(s, "    {path} = {value}");
        }
        let _ = writeln!(s, "  source    : {}", self.scenarios_source);

        let _ = writeln!(s, "\nBEST EVER SCREENED");
        match &self.best_ever {
            Some(b) => {
                let _ = writeln!(
                    s,
                    "  {} slot {}  config_hash {}",
                    b.sweep, b.slot, b.config_hash
                );
                let _ = writeln!(
                    s,
                    "  E_screen_pess {:+.4} pips/trade over {} trade(s)   DSR {}   PBO {}",
                    b.e_screen_pess,
                    b.n_trades,
                    b.dsr
                        .map(|d| format!("{d:.4}"))
                        .unwrap_or_else(|| "<refused>".to_string()),
                    b.pbo
                        .map(|p| format!("{p:.4}"))
                        .unwrap_or_else(|| "<refused>".to_string()),
                );
            }
            None => {
                let _ = writeln!(
                    s,
                    "  none — no sweep in this session ever passed the screen. That is a result, \
                     not an empty field."
                );
            }
        }

        // The R2 ledger, in EVERY verdict: on a success it shows what the
        // refutation would have needed; on an inconclusive it is §10.3's
        // required "here is which ones were not met".
        let _ = writeln!(s);
        let _ = write!(s, "{}", self.unreachable.render());

        let _ = writeln!(s, "\nCOVERAGE — what was actually searched");
        if self.coverage.is_empty() {
            let _ = writeln!(
                s,
                "  none. Nothing was credited, so no claim about this space is supportable."
            );
        }
        for (factor, level, sweeps, draws) in &self.coverage {
            let _ = writeln!(
                s,
                "  {factor:<24} {level:<24} {sweeps} sweep(s), {draws} draw(s)"
            );
        }

        let _ = writeln!(s, "\n{}", self.census.render());

        match &self.oos {
            Some(oos) => {
                let _ = writeln!(s, "\n{}", oos.render());
            }
            None => {
                let _ = writeln!(
                    s,
                    "\nOUT OF SAMPLE: the single touch was NEVER SPENT. Every number above is \
                     in-sample and no claim here has been checked out of sample."
                );
            }
        }

        let _ = writeln!(s, "\nREPRODUCE\n  {}", self.reproduction);
        let _ = writeln!(
            s,
            "\nThis loop PROPOSES. It never places an order, never contacts a broker and never \
             writes live_portfolio.json. The operator promotes."
        );
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::BlockOutcome;

    fn thresholds() -> JudgeThresholds {
        JudgeThresholds::frozen()
    }

    fn empty_session() -> Session {
        Session::default()
    }

    #[test]
    fn an_empty_session_can_never_refute_a_space() {
        // The trap this test exists for: `under_covered` returns [] for empty
        // coverage, so a naive U4 would read "nothing is under-drawn" as full
        // coverage, and a loop that drew NOTHING could declare the goal
        // impossible.
        let check = check_unreachable(&empty_session(), &thresholds());
        assert!(!check.satisfied);
        assert_eq!(check.conditions.len(), 4);
        let u4 = check.conditions.iter().find(|c| c.id == "U4").unwrap();
        assert!(!u4.satisfied);
        assert!(u4.detail.contains("vacuously"), "{}", u4.detail);
    }

    #[test]
    fn every_condition_is_evaluated_even_after_one_fails() {
        // The inconclusive report has to name which conditions were unmet, so a
        // short-circuit would silently shrink the evidence.
        let check = check_unreachable(&empty_session(), &thresholds());
        assert_eq!(check.unmet().len(), 4);
        for id in ["U1", "U2", "U3", "U4"] {
            assert!(check.conditions.iter().any(|c| c.id == id), "missing {id}");
        }
    }

    #[test]
    fn u2_is_unsatisfied_while_the_null_is_unavailable() {
        // "The loop never beat its own noise" is not a claim that can be made
        // before there IS a noise level. A refusal must not read as confirmation.
        let mut s = empty_session();
        s.null_observations = vec![0.3];
        let check = check_unreachable(&s, &thresholds());
        let u2 = check.conditions.iter().find(|c| c.id == "U2").unwrap();
        assert!(!u2.satisfied);
        assert!(u2.detail.contains("needs"), "{}", u2.detail);
    }

    #[test]
    fn u2_fails_when_the_loop_did_beat_the_null() {
        let mut s = empty_session();
        s.null_observations = vec![0.1, 0.2, 0.3];
        s.best_ever = Some(BestEver {
            sweep: SweepId(1),
            slot: 0,
            config_hash: "fnv64:abc".to_string(),
            e_screen_pess: 0.90,
            n_trades: 400,
            dsr: Some(0.99),
            pbo: Some(0.2),
        });
        let check = check_unreachable(&s, &thresholds());
        let u2 = check.conditions.iter().find(|c| c.id == "U2").unwrap();
        assert!(!u2.satisfied, "0.90 beats Q_0.95 = 0.30");
    }

    #[test]
    fn u2_holds_when_the_loop_never_beat_the_null() {
        let mut s = empty_session();
        s.null_observations = vec![0.1, 0.2, 0.3];
        s.best_ever = Some(BestEver {
            sweep: SweepId(1),
            slot: 0,
            config_hash: "fnv64:abc".to_string(),
            e_screen_pess: 0.10,
            n_trades: 400,
            dsr: Some(0.99),
            pbo: Some(0.2),
        });
        let check = check_unreachable(&s, &thresholds());
        assert!(
            check
                .conditions
                .iter()
                .find(|c| c.id == "U2")
                .unwrap()
                .satisfied
        );
    }

    #[test]
    fn u3_needs_at_least_one_judged_block() {
        // Zero blocks gives 0/0; the fraction must not be read as satisfied.
        let check = check_unreachable(&empty_session(), &thresholds());
        assert!(
            !check
                .conditions
                .iter()
                .find(|c| c.id == "U3")
                .unwrap()
                .satisfied
        );
    }

    #[test]
    fn u3_counts_indistinguishable_blocks() {
        let mut s = empty_session();
        for i in 0..4 {
            s.blocks.push(BlockOutcome {
                block: crate::session::BlockId(i),
                delta_expectancy: -0.1,
                p_block: 1.0,
                dsr_gap: -0.1,
                indistinguishable: i < 3,
            });
        }
        let check = check_unreachable(&s, &thresholds());
        let u3 = check.conditions.iter().find(|c| c.id == "U3").unwrap();
        assert!(u3.satisfied, "3/4 = 75% meets the threshold: {}", u3.detail);
    }

    #[test]
    fn a_session_with_no_passing_screen_has_no_promotion_candidate() {
        assert!(promotion_candidate(&empty_session()).is_none());
        assert_eq!(
            decide(&empty_session(), &GoalSet::default(), &thresholds()),
            Decision::Continue
        );
    }

    #[test]
    fn the_headline_of_a_goal_reached_without_its_metric_says_so() {
        // A "GOAL REACHED" with no number beside it is exactly the hollow report
        // this design exists to prevent.
        let v = Verdict::GoalReached {
            sweep: SweepId(3),
            slot: 7,
            goal_metric: None,
            goal_bar: None,
            frontier: Vec::new(),
        };
        assert!(v.headline(0).contains("NOT ATTACHED"));
        let v = Verdict::GoalReached {
            sweep: SweepId(3),
            slot: 7,
            goal_metric: Some(0.61),
            goal_bar: Some(0.50),
            frontier: Vec::new(),
        };
        assert!(v.headline(0).contains("0.6100"));
    }

    #[test]
    fn an_inconclusive_headline_denies_being_evidence_of_no_edge() {
        let v = Verdict::Inconclusive {
            reason: "the operator stopped the session".to_string(),
        };
        let h = v.headline(3);
        assert!(h.contains("NOT evidence that no edge exists"));
        assert!(h.contains("3 of the R2 conditions"));
    }

    fn sample(verdict: Verdict) -> SessionVerdict {
        SessionVerdict {
            schema: VERDICT_SCHEMA.to_string(),
            verdict,
            session_id: "ar-test".to_string(),
            symbol: "EURUSD".to_string(),
            goal_hash: "fnv64:aaaa".to_string(),
            judge_hash: "fnv64:bbbb".to_string(),
            cost_hash: "fnv64:cccc".to_string(),
            scenario_label: "100 -> 50k (x500)".to_string(),
            scenario_fields: vec![(
                "system.risky_target_balance_usd".to_string(),
                "50000".to_string(),
            )],
            scenarios_source: "system.risky_* (single)".to_string(),
            oos_window: OosWindow {
                start_ms: 1,
                end_ms: 2,
            },
            sweeps_run: 40,
            live_sweeps_run: 32,
            searches_run: 4000,
            n_trials_session: 412_000,
            best_ever: None,
            shuffle: ShuffleSummary::default(),
            unreachable: check_unreachable(&empty_session(), &thresholds()),
            coverage: Vec::new(),
            census: AbandonCensus::default(),
            oos: None,
            reproduction: "neoethos autoresearch --symbol EURUSD --session ar-test".to_string(),
        }
    }

    #[test]
    fn all_three_verdicts_render_the_shuffle_block_immediately_under_the_headline() {
        // §9.4, rendered mechanically so it cannot be omitted. If this ever
        // fails, a report can quote a best result without its falsification.
        let verdicts = [
            Verdict::GoalReached {
                sweep: SweepId(3),
                slot: 1,
                goal_metric: Some(0.61),
                goal_bar: Some(0.50),
                frontier: Vec::new(),
            },
            Verdict::GoalUnreachableInSearchedSpace {
                distinct_configurations: 3_940,
                proposer_exhausted: false,
            },
            Verdict::Inconclusive {
                reason: "operator stop".to_string(),
            },
        ];
        for v in verdicts {
            let out = sample(v).render();
            let at = out
                .find("SHUFFLE CONTROL")
                .expect("every verdict renders the shuffle summary");
            let before = &out[..at];
            for section in [
                "SESSION\n",
                "GOAL (read-only",
                "BEST EVER",
                "COVERAGE",
                "ABANDON CENSUS",
                "OUT OF SAMPLE",
                "REPRODUCE",
                "R2 —",
            ] {
                assert!(
                    !before.contains(section),
                    "section {section:?} was rendered ABOVE the shuffle summary; the \
                     falsification must sit in the same breath as the headline"
                );
            }
            assert!(
                before.lines().count() <= 4,
                "the shuffle summary starts {} line(s) in",
                before.lines().count()
            );
        }
    }

    #[test]
    fn every_verdict_carries_the_r2_ledger_and_the_unspent_oos_notice() {
        let out = sample(Verdict::Inconclusive {
            reason: "operator stop".to_string(),
        })
        .render();
        assert!(out.contains("R2 —"));
        assert!(out.contains("NEVER SPENT"));
        assert!(out.contains("That is a result, not an empty field."));
        assert!(out.contains("The operator promotes."));
    }

    #[test]
    fn an_empty_coverage_list_says_no_claim_is_supportable() {
        let out = sample(Verdict::GoalUnreachableInSearchedSpace {
            distinct_configurations: 0,
            proposer_exhausted: false,
        })
        .render();
        assert!(out.contains("no claim about this space is supportable"));
    }

    #[test]
    fn the_verdict_round_trips_through_json() {
        let v = sample(Verdict::GoalUnreachableInSearchedSpace {
            distinct_configurations: 3_940,
            proposer_exhausted: true,
        });
        let text = serde_json::to_string(&v).expect("verdict serialises");
        let back: SessionVerdict = serde_json::from_str(&text).expect("verdict deserialises");
        assert_eq!(v, back);
    }
}
