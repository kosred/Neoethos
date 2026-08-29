//! **The judge — the spine of this design, not a filter on the end.**
//!
//! `docs/autoresearch-loop.md` §8.
//!
//! ## The one risk this module exists to contain
//!
//! An optimiser pointed at *"turn 100 into 50,000 in six months"* will find
//! configurations that appear to do it. The measured base rate says so:
//! expectancy was **−4.15 pips per trade in every exit configuration tested**
//! while the payoff ratio moved from 0.91 to 2.53
//! (`neoethos_search::run_identity`, module doc). Exit geometry redistributes
//! between win rate and payoff; the product stays at minus the cost. A loop
//! judging on in-sample results will therefore converge on spectacular overfits,
//! present them with confidence, and the operator will trade them with real
//! money.
//!
//! Four rules are wired into the code rather than written on a wall:
//!
//! 1. **Deflate against the SESSION-WIDE trial count.** `n_session` is a fold
//!    over the journal (`session::Session::n_session`), passed in. There is no
//!    field here to reset, which is what makes *"a loop that resets its own N
//!    between sweeps"* structurally impossible rather than merely forbidden.
//! 2. **PBO above 0.5 discards the sweep whatever its headline number is.**
//!    Both must clear it: `pbo_sweep` (within one sweep) and [`session_pbo`]
//!    (of the LOOP'S OWN selection procedure across sweeps).
//! 3. **One out-of-sample touch, ever.** [`promote`] refuses on a spent budget
//!    before it reads a single out-of-sample number.
//! 4. **Charge the cost BAND, not a point estimate.**
//!
//! ## A refusal is a FAIL
//!
//! `deflated::TrialStatisticsReport` returns a refusal string rather than a
//! number whenever the matrix cannot support a statistic. Every one of those
//! refusals fails its conjunct here. **No number is never good news** — that is
//! the reader module's own doctrine and this judge enforces it rather than
//! reinterpreting it.
//!
//! ## Boundary
//!
//! The loop owns `SweepEvidence`, `SearchRecord` and `OosEvidence` because it
//! owns S5 COLLECT. The judge consumes them and returns verdicts. Every function
//! here is pure, so any claim the loop makes can be re-derived by re-calling it.

use anyhow::Result;
use chrono::{Datelike, TimeZone, Utc};
use neoethos_core::utils::hashing::fnv1a64;
use neoethos_search::deflated::{
    DecodedTrialMatrix, DecodedTrialRow, PboReport, TrialStatisticsReport, pbo_cscv,
};
use neoethos_search::goal_report::{DEFAULT_RISK_LEVELS, build_report};
use serde::{Deserialize, Serialize};

use crate::QuoteValidatedOosTouchEvidenceV1;
use crate::goals::{GoalSet, Scenario};
use crate::journal::{CostBandCounts, OosWindow, RiskOutcomeRecord};
use crate::session::{ChampionRow, Session, SweepEvidence, SweepId};
use crate::shuffle::ShuffleNull;
use crate::verdict::PromotionCandidate;

// ─────────────────────────────────────────────────────────────────────────────
// Thresholds — frozen for the life of a session
// ─────────────────────────────────────────────────────────────────────────────

/// Above one half the selection procedure is worse than random. Not a tuning
/// knob, which is why it is stated here and hashed.
pub const PBO_MAX: f64 = 0.50;
/// `P(true Sharpe > SR*)` the champion must clear after deflation.
pub const DSR_MIN: f64 = 0.95;
/// The final fraction of the bar span, by time, reserved and touched once.
pub const OOS_FRACTION: f64 = 0.20;
/// One. Per `(scenario, window)`.
pub const OOS_TOUCHES_TOTAL: u32 = 1;
/// `min_expectancy_t_stat >= 2` applied OUT of sample, as `E − 2·SE > 0`.
pub const OOS_T_STAT_MIN: f64 = 2.0;
/// Below one half the MEDIAN PATH DOES NOT REACH THE TARGET, and a "goal
/// reached" claim backed by `P(reach) = 0.2` is a lottery ticket sold as a
/// result.
pub const P_REACH_MIN: f64 = 0.50;
/// Likewise. The frontier is reported in full either way, so the operator sees
/// the whole curve and not only the conjunct that passed.
pub const P_RUIN_MAX: f64 = 0.50;
/// In-sample trade floor used until the runner installs a window-derived one.
///
/// Deliberately not zero: a floor of zero lets a candidate with no trades clear
/// the trade conjunct.
pub const N_MIN_SCREEN_DEFAULT: usize = 30;
/// Out-of-sample trade floor default, same reasoning.
pub const N_MIN_OOS_DEFAULT: usize = 30;

/// The judge's thresholds. **Read-only to the loop**, exactly like the goals.
///
/// Hashed into `judge_hash`, written into the session header and verified on
/// every resume. Changing one starts a NEW SESSION ID: results judged by two
/// different judges are not comparable and must never share a history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeThresholds {
    pub pbo_max: f64,
    pub dsr_min: f64,
    pub p_reach_min: f64,
    pub p_ruin_max: f64,
    pub oos_fraction: f64,
    pub oos_touches_total: u32,
    pub oos_t_stat_min: f64,
    pub min_null_obs: usize,
    /// Derived per session from `min_trades_per_month × months_in_window` —
    /// never a constant, so it scales with the window rather than degrading
    /// differently on every dataset.
    pub n_min_screen: usize,
    pub n_min_oos: usize,
}

impl JudgeThresholds {
    /// The frozen judge.
    pub fn frozen() -> Self {
        Self {
            pbo_max: PBO_MAX,
            dsr_min: DSR_MIN,
            p_reach_min: P_REACH_MIN,
            p_ruin_max: P_RUIN_MAX,
            oos_fraction: OOS_FRACTION,
            oos_touches_total: OOS_TOUCHES_TOTAL,
            oos_t_stat_min: OOS_T_STAT_MIN,
            min_null_obs: crate::MIN_NULL_OBS,
            n_min_screen: N_MIN_SCREEN_DEFAULT,
            n_min_oos: N_MIN_OOS_DEFAULT,
        }
    }

    /// Install the two window-derived trade floors — the only two numbers here
    /// that may legitimately vary with the dataset.
    ///
    /// They still enter `judge_hash`, so a session cannot silently resume under
    /// a different floor.
    ///
    /// **[`Self::frozen`] alone is not a judge a session may run under.** Its
    /// two floors are placeholders, and a placeholder floor is the failure this
    /// method exists to prevent: on a two-year out-of-sample window at half a
    /// trade a day the derived floor is around 360 trades, so judging with the
    /// 30-trade default applies a bar twelve times weaker than the one the
    /// startup self-check told the operator it had verified — on the one window
    /// that cannot be re-read. `runner::session_thresholds` is the caller, and
    /// it is the only one outside tests.
    pub fn with_derived_floors(mut self, n_min_screen: usize, n_min_oos: usize) -> Self {
        self.n_min_screen = n_min_screen.max(1);
        self.n_min_oos = n_min_oos.max(1);
        self
    }

    /// `fnv64:…` over every field in declaration order.
    ///
    /// `f64::to_bits` rather than a formatted decimal, so the hash is exact and
    /// cannot drift with a format-precision change.
    pub fn judge_hash(&self) -> String {
        let mut b: Vec<u8> = Vec::with_capacity(88);
        for x in [
            self.pbo_max,
            self.dsr_min,
            self.p_reach_min,
            self.p_ruin_max,
            self.oos_fraction,
            self.oos_t_stat_min,
        ] {
            b.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        for n in [
            u64::from(self.oos_touches_total),
            self.min_null_obs as u64,
            self.n_min_screen as u64,
            self.n_min_oos as u64,
        ] {
            b.extend_from_slice(&n.to_le_bytes());
        }
        format!("fnv64:{:016x}", fnv1a64(&b))
    }

    /// Every threshold, by name, for the `SessionOpened` header.
    ///
    /// **In `judge_hash`'s declaration order, and covering every field it
    /// hashes.** The header is what an auditor reads to learn what judged the
    /// session; a header that listed nine of the eleven numbers would let a
    /// threshold change the hash — and therefore refuse the resume — without
    /// ever appearing in the artifact that says why.
    ///
    /// The two `usize` trade floors are included even though they are derived
    /// per session: they are derived ONCE, they enter `judge_hash`, and a reader
    /// cannot re-derive them without the window.
    pub fn named_values(&self) -> crate::journal::NamedValues {
        crate::journal::NamedValues(vec![
            ("pbo_max".into(), format!("{:?}", self.pbo_max)),
            ("dsr_min".into(), format!("{:?}", self.dsr_min)),
            ("p_reach_min".into(), format!("{:?}", self.p_reach_min)),
            ("p_ruin_max".into(), format!("{:?}", self.p_ruin_max)),
            ("oos_fraction".into(), format!("{:?}", self.oos_fraction)),
            (
                "oos_t_stat_min".into(),
                format!("{:?}", self.oos_t_stat_min),
            ),
            (
                "oos_touches_total".into(),
                self.oos_touches_total.to_string(),
            ),
            ("min_null_obs".into(), self.min_null_obs.to_string()),
            ("n_min_screen".into(), self.n_min_screen.to_string()),
            ("n_min_oos".into(), self.n_min_oos.to_string()),
            ("judge_hash".into(), self.judge_hash()),
        ])
    }
}

/// Derive a trade floor from the window it will be applied to.
///
/// The result is **never zero**.
pub fn n_min_trades(min_trades_per_month: f64, months_in_window: f64) -> usize {
    let raw = min_trades_per_month.max(0.0) * months_in_window.max(0.0);
    if !raw.is_finite() {
        return 1;
    }
    // A partial trade rounds UP — half a trade is not a trade, and the floor is
    // a floor. But `15 × (800/30)` evaluates to 400.000000000000057, and a
    // floating-point remainder of 6e-14 is not a trade either: bare `ceil` would
    // turn a clean 400 into 401 and make the bar depend on how the window
    // happens to divide. Anything within a millionth of an integer IS that
    // integer; everything else still rounds up.
    let snapped = if (raw - raw.round()).abs() < 1.0e-6 {
        raw.round()
    } else {
        raw.ceil()
    };
    (snapped as usize).max(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1 — SCREEN (in-sample only; never touches OOS)
// ─────────────────────────────────────────────────────────────────────────────

/// The conjuncts of `SCREEN`, in evaluation order.
///
/// **S2b is folded into [`ScreenConjunct::S2Dsr`]** and named in the detail
/// string, because `AbandonCensus::screen_failed_by_conjunct` is `[usize; 7]` =
/// S1..S6 plus `Unavailable` (§12) and a seventh conjunct would shift every
/// index in every journal already written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenConjunct {
    /// PBO within the sweep is present and `≤ pbo_max`.
    S1Pbo,
    /// The deflated Sharpe is present, `≥ dsr_min`, AND (S2b) its
    /// `excess_over_expected_max_per_period` is strictly positive.
    S2Dsr,
    /// The cost band discriminates, nothing survived only at the optimistic
    /// edge, and at least one candidate survived the band.
    S3CostBand,
    /// Per-trade net expectancy at the PESSIMISTIC edge is strictly positive.
    S4Expectancy,
    /// Trade count clears the derived floor.
    S5Trades,
    /// The candidate beats the accumulated shuffle null.
    S6Shuffle,
    /// The null is not yet available, so S6 cannot be evaluated. A failure for
    /// the proposer and a block on promotion — never a pass.
    Unavailable,
}

impl ScreenConjunct {
    /// Index into `AbandonCensus::screen_failed_by_conjunct`.
    pub fn index(self) -> usize {
        match self {
            Self::S1Pbo => 0,
            Self::S2Dsr => 1,
            Self::S3CostBand => 2,
            Self::S4Expectancy => 3,
            Self::S5Trades => 4,
            Self::S6Shuffle => 5,
            Self::Unavailable => 6,
        }
    }

    /// Labels in index order, for the census renderer.
    pub const ALL_LABELS: [&'static str; 7] = [
        "S1_pbo",
        "S2_dsr",
        "S3_cost_band",
        "S4_expectancy",
        "S5_trades",
        "S6_shuffle",
        "unavailable",
    ];

    pub fn label(self) -> &'static str {
        Self::ALL_LABELS[self.index()]
    }
}

/// One search's screen verdict. Carried whole in the journal, so the decision is
/// re-derivable rather than remembered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ScreenResult {
    Passed {
        e_screen_pess: f64,
        dsr: f64,
        pbo_sweep: f64,
        excess_over_expected_max_per_period: f64,
        n_trades: usize,
        q_shuffle_95: f64,
    },
    Failed {
        conjunct: ScreenConjunct,
        detail: String,
    },
    /// The shuffle null is not yet available. Counts as a failure everywhere,
    /// and is distinguished from `Failed` only so the proposer can tell "this
    /// configuration lost" from "nothing could be judged yet" — §5.3: while the
    /// null is unavailable the posterior is not updated at all.
    Unavailable { detail: String },
}

impl ScreenResult {
    pub fn passed(&self) -> bool {
        matches!(self, Self::Passed { .. })
    }

    /// The proposer's binary reward. Zero unless every conjunct held.
    ///
    /// Binary, not the continuous in-sample score: on this data that score is
    /// dominated by selection noise, so a proposer hill-climbing on it converges
    /// on the first lucky draw.
    pub fn reward(&self) -> u8 {
        u8::from(self.passed())
    }

    /// True when the posterior must not be updated at all, rather than credited
    /// a zero.
    pub fn null_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable { .. })
    }

    /// The census slot this outcome charges. `None` for a pass.
    pub fn census_index(&self) -> Option<usize> {
        match self {
            Self::Passed { .. } => None,
            Self::Failed { conjunct, .. } => Some(conjunct.index()),
            Self::Unavailable { .. } => Some(ScreenConjunct::Unavailable.index()),
        }
    }

    /// The label the journal records beside the result.
    pub fn failing_conjunct_label(&self) -> Option<String> {
        match self {
            Self::Passed { .. } => None,
            Self::Failed { conjunct, .. } => Some(conjunct.label().to_string()),
            Self::Unavailable { .. } => Some(ScreenConjunct::Unavailable.label().to_string()),
        }
    }

    /// The expectancy of a PASSING screen. `None` for anything else — a failing
    /// screen has no number worth carrying forward, and letting one through here
    /// would put rejects into the shuffle null and the best-ever record.
    pub fn e_screen_pess(&self) -> Option<f64> {
        match self {
            Self::Passed { e_screen_pess, .. } => Some(*e_screen_pess),
            _ => None,
        }
    }

    pub fn detail(&self) -> &str {
        match self {
            Self::Passed { .. } => "",
            Self::Failed { detail, .. } | Self::Unavailable { detail } => detail,
        }
    }
}

fn fail(conjunct: ScreenConjunct, detail: impl Into<String>) -> ScreenResult {
    ScreenResult::Failed {
        conjunct,
        detail: detail.into(),
    }
}

/// One search's evidence, as the judge reads it out of the loop's records.
#[derive(Debug, Clone, Copy)]
pub struct SearchEvidence<'a> {
    pub statistics: &'a TrialStatisticsReport,
    pub cost_band: CostBandCounts,
    /// `discovery::cost_band_discriminates(band, baseline_cost_pips)`, computed
    /// by the runner from the frozen cost model.
    pub band_discriminates: bool,
    /// Per-trade net expectancy of the best survivor at the PESSIMISTIC edge.
    pub e_screen_pess: Option<f64>,
    pub n_trades: usize,
    /// A per-search failure recorded by the executor.
    pub error: Option<&'a str>,
}

/// **Stage 1 — SCREEN, one search.** In-sample only. Never touches OOS.
///
/// Conjuncts are evaluated S1 → S6 and the FIRST failure is returned, so the
/// journal records a specific cause rather than "failed". S6 is evaluated last
/// on purpose: when the null is still unavailable but the candidate already
/// failed S1–S5, the recorded reason is the real one and not `Unavailable`.
pub fn screen(
    ev: SearchEvidence<'_>,
    th: &JudgeThresholds,
    n_session: usize,
    null: &ShuffleNull,
) -> ScreenResult {
    if let Some(e) = ev.error {
        return fail(
            ScreenConjunct::S1Pbo,
            format!("the search itself failed and produced no judgeable evidence: {e}"),
        );
    }
    let st = ev.statistics;

    // Unreadable: BOTH statistics refused for the same named cause. Reported
    // against S1 because that is the first conjunct it defeats, with the cause
    // spelled out so it is never mistaken for "the sweep was bad".
    //
    // **Checked BEFORE the N contract below.** An unreadable report has no
    // deflation to have a wrong N, and complaining about its denominator would
    // file a disk or wiring finding under S2 with a message about statistics
    // that were never computed — the real cause hidden behind a derived one.
    if st.deflated_sharpe.is_none() && st.pbo.is_none() && st.matrix_rows == 0 {
        return fail(
            ScreenConjunct::S1Pbo,
            format!(
                "UNREADABLE trial-return matrix — neither statistic was attempted. {} This is a \
                 wiring or disk finding, not a verdict on the configuration.",
                st.pbo_refusal.clone().unwrap_or_default()
            ),
        );
    }

    // The trial count the statistics were deflated against must BE the honest
    // N. A report built with one search's own row count would understate the
    // selection adjustment by orders of magnitude and every DSR below would be
    // flattered. Checked, not assumed — this is the quietest way this judge
    // could be defeated.
    //
    // **This is satisfiable, and the runner is what satisfies it.** `n_session`
    // is a fold over the journal and is not final until the sweep that is being
    // screened has been recorded, so the executor that ran the search CANNOT be
    // handed it. The runner therefore re-deflates every report against the fold
    // — `TrialStatisticsReport::redeflated_against` — after `SweepCompleted` is
    // appended and before the judge sees anything. When this fires it means
    // that step was skipped, which is a wiring fault and not a research result.
    if st.trials_offered != n_session {
        return fail(
            ScreenConjunct::S2Dsr,
            format!(
                "the statistics were deflated against trials_offered = {} but the session-wide \
                 trial count is {n_session}. A deflation against the wrong N is not a deflation; \
                 the runner must install the fold with \
                 TrialStatisticsReport::redeflated_against(n_session) before screening.",
                st.trials_offered
            ),
        );
    }

    // ── S1: PBO within the sweep ────────────────────────────────────────────
    let Some(pbo) = st.pbo_value().filter(|p| p.is_finite()) else {
        return fail(
            ScreenConjunct::S1Pbo,
            format!(
                "no PBO. {} A refusal is a FAIL: an unknown overfitting probability is not a low \
                 one.",
                st.pbo_refusal.clone().unwrap_or_default()
            ),
        );
    };
    if pbo > th.pbo_max {
        return fail(
            ScreenConjunct::S1Pbo,
            format!(
                "PBO {pbo:.4} > {:.2} — the selection procedure is worse than random, so the \
                 sweep is discarded whatever its headline number.",
                th.pbo_max
            ),
        );
    }

    // ── S2 + S2b: the deflated Sharpe ───────────────────────────────────────
    let Some(dsr) = st.deflated_sharpe.as_ref() else {
        return fail(
            ScreenConjunct::S2Dsr,
            format!(
                "no deflated Sharpe. {} A refusal is a FAIL.",
                st.deflated_sharpe_refusal.clone().unwrap_or_default()
            ),
        );
    };
    if !dsr.deflated_sharpe_ratio.is_finite() || dsr.deflated_sharpe_ratio < th.dsr_min {
        return fail(
            ScreenConjunct::S2Dsr,
            format!(
                "deflated Sharpe {:.4} < {:.2} at N = {} trials",
                dsr.deflated_sharpe_ratio, th.dsr_min, dsr.trials_n
            ),
        );
    }
    // S2b. The DSR's cross-trial Sharpe variance is estimated from THIS sweep's
    // rows while `trials_n` is the session-wide count. That mismatch understates
    // the variance, therefore understates SR*, therefore FLATTERS the DSR. This
    // term does not depend on the variance estimate, and it is what hardens the
    // conjunction.
    if dsr.excess_over_expected_max_per_period <= 0.0 {
        return fail(
            ScreenConjunct::S2Dsr,
            format!(
                "S2b: excess over the expected maximum by luck is {:+.4} per period — the \
                 champion did not beat what a search of {} trials produces from noise. Checked \
                 separately from the DSR because the DSR's variance estimate comes from this \
                 sweep's rows while N is session-wide, which biases the DSR UPWARD.",
                dsr.excess_over_expected_max_per_period, dsr.trials_n
            ),
        );
    }

    // ── S3: the cost band ───────────────────────────────────────────────────
    if !ev.band_discriminates {
        return fail(
            ScreenConjunct::S3CostBand,
            "the cost band cannot discriminate: its pessimistic edge is at or below the cost the \
             run already charged, so passing it is arithmetic and not evidence. A census that \
             reads clean on every run is worse than no census, because a reader takes it as \
             evidence. See docs/autoresearch-loop.md §14.4."
                .to_string(),
        );
    }
    if ev.cost_band.optimistic_edge_only != 0 {
        return fail(
            ScreenConjunct::S3CostBand,
            format!(
                "{} candidate(s) were profitable only at the CHEAP edge of the cost band. That is \
                 not a result and must not be reported as one.",
                ev.cost_band.optimistic_edge_only
            ),
        );
    }
    if ev.cost_band.survives < 1 {
        return fail(
            ScreenConjunct::S3CostBand,
            format!(
                "no candidate survived both edges of the cost band (survives {}, fails {}, \
                 unmeasured {}, not_discriminating {})",
                ev.cost_band.survives,
                ev.cost_band.fails,
                ev.cost_band.unmeasured,
                ev.cost_band.not_discriminating
            ),
        );
    }

    // ── S4 + S5: the survivor itself ────────────────────────────────────────
    let Some(e_pess) = ev.e_screen_pess else {
        return fail(
            ScreenConjunct::S4Expectancy,
            "the cost-band census reports a survivor but the search recorded no expectancy for \
             it, so there is nothing to judge. Counted as a failure rather than skipped."
                .to_string(),
        );
    };
    if !e_pess.is_finite() || e_pess <= 0.0 {
        return fail(
            ScreenConjunct::S4Expectancy,
            format!(
                "E_screen_pess = {e_pess:+.4} pips/trade at the pessimistic cost edge. The \
                 measured base rate for this instrument is -4.15 pips in every exit configuration \
                 tested; a non-positive number here is that base rate, not a near miss."
            ),
        );
    }
    if ev.n_trades < th.n_min_screen {
        return fail(
            ScreenConjunct::S5Trades,
            format!(
                "{} trade(s) < the derived floor of {} (min_trades_per_month x months_in_window)",
                ev.n_trades, th.n_min_screen
            ),
        );
    }

    // ── S6: beat the accumulated shuffle null ───────────────────────────────
    let Some(q) = null.quantile_95() else {
        return ScreenResult::Unavailable {
            detail: format!(
                "the shuffle null holds {} observation(s) and needs {}; S6 cannot be evaluated, \
                 so this screen is NOT a pass. Until the null is available every reward is zero \
                 by construction and the first blocks are pure exploration.",
                null.len(),
                th.min_null_obs
            ),
        };
    };
    if e_pess < q {
        return fail(
            ScreenConjunct::S6Shuffle,
            format!(
                "E_screen_pess {e_pess:+.4} < Q_0.95 of the shuffle null {q:+.4}: this \
                 configuration scores like a time-rotated control, which is what mining noise \
                 looks like."
            ),
        );
    }

    ScreenResult::Passed {
        e_screen_pess: e_pess,
        dsr: dsr.deflated_sharpe_ratio,
        pbo_sweep: pbo,
        excess_over_expected_max_per_period: dsr.excess_over_expected_max_per_period,
        n_trades: ev.n_trades,
        q_shuffle_95: q,
    }
}

/// One sweep's screen: a verdict per slot, plus the champion it contributes.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepScreen {
    /// Index-aligned with the sweep's proposals and per-search records.
    pub per_slot: Vec<ScreenResult>,
    /// The row this sweep contributes to the session champion matrix, taken from
    /// the best **passing** slot.
    ///
    /// A sweep whose every slot failed contributes nothing: a champion drawn
    /// from a failed screen would put the loop's own rejects into the very
    /// matrix its overfitting probability is measured from.
    pub champion: Option<ChampionRow>,
    /// Why this sweep contributes NO row although a slot passed. `None` when
    /// there is nothing to explain.
    ///
    /// A sweep dropping out of `pbo_session`'s input changes the number that
    /// gates every promotion, so the reason is carried rather than inferred from
    /// a `None`. It is never a licence to substitute another slot's series.
    pub champion_refusal: Option<String>,
    pub champion_slot: Option<usize>,
    /// The passing slot's own identity and screened expectancy.
    ///
    /// Carried separately from [`Self::champion`] because the session's
    /// BEST-EVER record needs neither a return series nor a place in the PBO
    /// matrix — it needs the slot, the hash and the number. Reading them off the
    /// champion row instead would make `best_ever` — and therefore R2's U2, and
    /// therefore the loop's ability to report GOAL UNREACHABLE — silently
    /// conditional on a matrix write that is non-fatal by design.
    pub champion_config_hash: Option<String>,
    pub champion_e_screen_pess: Option<f64>,
    pub champion_n_trades: usize,
    pub champion_dsr: Option<f64>,
    pub champion_pbo: Option<f64>,
}

impl SweepScreen {
    pub fn passed(&self) -> usize {
        self.per_slot.iter().filter(|r| r.passed()).count()
    }

    /// The sweep's best `E_screen_pess` over PASSING slots.
    pub fn best_e_screen_pess(&self) -> Option<f64> {
        self.per_slot
            .iter()
            .filter_map(ScreenResult::e_screen_pess)
            .reduce(f64::max)
    }

    /// Failure counts by census slot, for `AbandonCensus`.
    pub fn failures_by_conjunct(&self) -> [usize; 7] {
        let mut out = [0usize; 7];
        for r in &self.per_slot {
            if let Some(i) = r.census_index() {
                out[i] += 1;
            }
        }
        out
    }
}

/// **Stage 1 — SCREEN, one sweep.** Every slot, then the champion.
pub fn screen_sweep(
    evidence: &SweepEvidence,
    th: &JudgeThresholds,
    n_session: usize,
    null: &ShuffleNull,
) -> SweepScreen {
    let mut per_slot = Vec::with_capacity(evidence.per_search.len());
    let mut best: Option<(usize, f64)> = None;

    for (i, rec) in evidence.per_search.iter().enumerate() {
        let Some(statistics) = evidence.statistics.get(i) else {
            // Index misalignment is a wiring fault, and a slot with no
            // statistics must never be scored as anything but a failure.
            per_slot.push(fail(
                ScreenConjunct::S1Pbo,
                format!(
                    "slot {i} has a search record but no statistics report; the two vectors are \
                     index-aligned by contract and this pair is not."
                ),
            ));
            continue;
        };
        let result = screen(
            SearchEvidence {
                statistics,
                cost_band: rec.cost_band,
                band_discriminates: rec.cost_band.discriminates,
                e_screen_pess: rec.e_screen_pess,
                n_trades: rec.n_trades,
                error: rec.error.as_deref(),
            },
            th,
            n_session,
            null,
        );
        if let Some(e) = result.e_screen_pess()
            && best.map(|(_, b)| e > b).unwrap_or(true)
        {
            best = Some((i, e));
        }
        per_slot.push(result);
    }

    match best {
        Some((slot, e)) => {
            let rec = &evidence.per_search[slot];
            // The row is TAKEN from the slot that passed. It is never taken from
            // somewhere else and re-stamped: a re-stamp makes a mismatch
            // invisible, and the mismatch is the defect. The slot's own row
            // already carries the slot's own `config_hash` and its own
            // `e_screen_pess`, so there is nothing left to relabel.
            let (champion, champion_refusal) = match evidence.champion_rows.get(slot) {
                Some(Some(row)) if row.config_hash == rec.config_hash => (Some(row.clone()), None),
                Some(Some(row)) => (
                    None,
                    Some(format!(
                        "slot {slot} passed the screen under configuration {} but the champion row \
                         materialised for that slot names {}. The two are index-aligned by \
                         contract, so a disagreement means the row would enter the session \
                         champion matrix — the input to pbo_session, which gates every promotion — \
                         under an identity that is not its own. Refused, and this sweep \
                         contributes NO row.",
                        rec.config_hash, row.config_hash
                    )),
                ),
                Some(None) => (
                    None,
                    Some(format!(
                        "slot {slot} ({}) passed the screen but its search produced no per-period \
                         champion series, so this sweep contributes no row to the session champion \
                         matrix. The row is NOT taken from another slot: the highest-expectancy \
                         slot is typically the overfit outlier the screen just rejected, and its \
                         return series wearing this slot's identity is precisely what \
                         pbo_session would then be measuring.",
                        rec.config_hash
                    )),
                ),
                None => (
                    None,
                    Some(format!(
                        "the evidence carries {} champion rows for {} search records, so slot \
                         {slot}'s row cannot be located. The two vectors are index-aligned by \
                         contract and this pair is not.",
                        evidence.champion_rows.len(),
                        evidence.per_search.len()
                    )),
                ),
            };
            SweepScreen {
                per_slot,
                champion,
                champion_refusal,
                champion_slot: Some(slot),
                champion_config_hash: Some(rec.config_hash.clone()),
                champion_e_screen_pess: Some(e),
                champion_n_trades: rec.n_trades,
                champion_dsr: rec.dsr,
                champion_pbo: rec.pbo,
            }
        }
        None => SweepScreen {
            per_slot,
            champion: None,
            champion_refusal: None,
            champion_slot: None,
            champion_config_hash: None,
            champion_e_screen_pess: None,
            champion_n_trades: 0,
            champion_dsr: None,
            champion_pbo: None,
        },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// pbo_session — PBO of the LOOP'S OWN selection procedure
// ─────────────────────────────────────────────────────────────────────────────

/// The constructed matrix plus everything that was dropped building it.
#[derive(Debug, Clone)]
pub struct SessionChampionMatrix {
    pub matrix: DecodedTrialMatrix,
    pub rows_dropped: usize,
    pub notes: Vec<String>,
}

/// Build the session champion matrix **in memory** — one row per sweep.
///
/// `DecodedTrialMatrix`'s fields are all `pub`, so this is constructed and
/// handed straight to `pbo_cscv`. No encoder is needed and `trial_returns.rs`
/// does not change.
///
/// Rows are aligned on the **intersection** of their period keys: two sweeps
/// that spanned slightly different months are still comparable over the months
/// they share, and CSCV ranks trials against each other period by period. Every
/// row that could not be used is dropped **and named** — a champion silently
/// missing from the matrix would change PBO without changing anything a reader
/// can see.
pub fn build_session_champion_matrix(
    rows: &[ChampionRow],
    initial_balance: f64,
) -> Result<SessionChampionMatrix, String> {
    if rows.is_empty() {
        return Err(
            "REFUSED: no sweep champions, so the loop's own selection procedure has no \
                    matrix and its PBO cannot be computed."
                .to_string(),
        );
    }

    let mut notes: Vec<String> = Vec::new();
    let mut dropped = 0usize;
    let mut usable: Vec<&ChampionRow> = Vec::with_capacity(rows.len());
    for r in rows {
        let bad = if r.period_keys.len() != r.monthly_returns.len() {
            Some(format!(
                "{} period key(s) against {} return(s)",
                r.period_keys.len(),
                r.monthly_returns.len()
            ))
        } else if r.period_keys.is_empty() {
            Some("empty period grid".to_string())
        } else if r.period_keys.windows(2).any(|w| w[1] <= w[0]) {
            Some("period keys are not strictly ascending".to_string())
        } else if r.monthly_returns.iter().any(|v| !v.is_finite()) {
            Some("a non-finite monthly return".to_string())
        } else {
            None
        };
        match bad {
            Some(why) => {
                dropped += 1;
                notes.push(format!("{} dropped: {why}", r.sweep));
            }
            None => usable.push(r),
        }
    }
    if usable.is_empty() {
        return Err(format!(
            "REFUSED: all {} champion row(s) were unusable. {}",
            rows.len(),
            notes.join("; ")
        ));
    }

    let mut shared: Vec<i64> = usable[0].period_keys.clone();
    for r in usable.iter().skip(1) {
        shared.retain(|k| r.period_keys.binary_search(k).is_ok());
    }
    if shared.is_empty() {
        return Err(format!(
            "REFUSED: the {} usable champion row(s) share no calendar month, so they cannot be \
             ranked against each other period by period.",
            usable.len()
        ));
    }
    if shared.len() < usable[0].period_keys.len() {
        notes.push(format!(
            "aligned on the intersection of the champions' period grids: {} shared month(s) out \
             of {} in the first row's grid.",
            shared.len(),
            usable[0].period_keys.len()
        ));
    }

    let mut out_rows: Vec<DecodedTrialRow> = Vec::with_capacity(usable.len());
    for (i, r) in usable.iter().enumerate() {
        let mut returns = Vec::with_capacity(shared.len());
        for k in &shared {
            match r.period_keys.binary_search(k) {
                Ok(idx) => returns.push(r.monthly_returns[idx]),
                Err(_) => return Err(format!("{}: month {k} vanished from its own grid", r.sweep)),
            }
        }
        out_rows.push(DecodedTrialRow {
            candidate_index: i,
            strategy_id: format!("{}::{}", r.sweep, r.strategy_id),
            returns,
        });
    }

    let n = out_rows.len();
    Ok(SessionChampionMatrix {
        matrix: DecodedTrialMatrix {
            version: 1,
            // bit 0 = "returns are a fraction of the initial balance", which is
            // exactly what a champion's monthly series is.
            flags: 1,
            initial_balance: if initial_balance.is_finite() && initial_balance != 0.0 {
                initial_balance
            } else {
                1.0
            },
            period_keys: shared,
            rows: out_rows,
            header_trials: n,
            truncated_tail_bytes: 0,
        },
        rows_dropped: dropped,
        notes,
    })
}

/// `pbo_session` — CSCV over the session champion matrix.
///
/// PBO of **the loop's own selection procedure**, which is the object the
/// operator's instruction names; `pbo_sweep` is PBO within one sweep. Both must
/// clear `PBO_MAX`.
///
/// **An emergent constraint worth stating out loud:** `pbo_cscv` refuses below
/// `MIN_TRIALS_FOR_PBO` rows and `MIN_PERIODS_FOR_PBO` periods, and a refusal is
/// a fail. So no promotion is possible until the session has accumulated that
/// many sweep champions over that many shared months. That is not an accident to
/// be worked around — a selection procedure with nine observations has no
/// measurable overfitting probability.
pub fn session_pbo(
    rows: &[ChampionRow],
    initial_balance: f64,
) -> Result<(PboReport, SessionChampionMatrix), String> {
    let built = build_session_champion_matrix(rows, initial_balance)?;
    let report = pbo_cscv(&built.matrix)?;
    Ok((report, built))
}

/// `pbo_session` for a session: the value, or the refusal that stands in for it.
pub fn session_pbo_value(session: &Session) -> (Option<f64>, Option<String>) {
    match session_pbo(&session.champions, 1.0) {
        Ok((report, _)) => (Some(report.pbo), None),
        Err(e) => (None, Some(e)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2 — the promotion inequality
// ─────────────────────────────────────────────────────────────────────────────

/// The conjuncts of `PROMOTE`, in evaluation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionConjunct {
    OosTouchAlreadySpent,
    PboSweep,
    PboSession,
    Dsr,
    ExcessOverExpectedMax,
    OosExpectancy,
    OosTStat,
    OosCostBand,
    OosTradeCount,
    GoalMetric,
    /// Risky: `p_ruin` at the winning risk level. PropFirm: a drawdown
    /// violation, or no complete month.
    ScenarioPathConstraint,
}

impl PromotionConjunct {
    pub fn label(self) -> &'static str {
        match self {
            Self::OosTouchAlreadySpent => "oos_touch_already_spent",
            Self::PboSweep => "pbo_sweep",
            Self::PboSession => "pbo_session",
            Self::Dsr => "dsr",
            Self::ExcessOverExpectedMax => "excess_over_expected_max",
            Self::OosExpectancy => "oos_expectancy",
            Self::OosTStat => "oos_t_stat",
            Self::OosCostBand => "oos_cost_band",
            Self::OosTradeCount => "oos_trade_count",
            Self::GoalMetric => "goal_metric",
            Self::ScenarioPathConstraint => "scenario_path_constraint",
        }
    }
}

/// The out-of-sample evaluation — the decision AND the record.
///
/// One object, because a promotion decision that is not also a record is a claim
/// nobody can re-derive. Carried in the verdict whether the promotion held or
/// not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionOutcome {
    pub sweep: SweepId,
    pub slot: usize,
    pub window: OosWindow,
    pub promoted: bool,
    /// The label of the first conjunct that failed, or `None` on a promotion.
    pub failing_conjunct: Option<String>,
    pub detail: String,

    pub n_trades: usize,
    pub n_min_oos: usize,
    pub e_oos_pess_pips: Option<f64>,
    pub se_oos_pips: Option<f64>,
    pub t_stat: Option<f64>,
    pub band_survives: bool,
    pub pbo_session: Option<f64>,

    pub goal_metric: f64,
    pub goal_bar: f64,
    /// The whole frontier, always — a "goal reached" claim is only readable
    /// beside the risk levels where it was not reached.
    pub frontier: Vec<RiskOutcomeRecord>,
    pub best_risk_fraction: Option<f64>,
    pub p_ruin_at_best: Option<f64>,

    pub months_total: usize,
    pub months_clearing_target: usize,
    pub notes: Vec<String>,
}

impl PromotionOutcome {
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "OUT OF SAMPLE — the single touch, spent on {} slot {}",
            self.sweep, self.slot
        );
        let _ = writeln!(
            s,
            "  window     : [{}, {}]",
            self.window.start_ms, self.window.end_ms
        );
        let _ = writeln!(
            s,
            "  trades     : {} (floor {})",
            self.n_trades, self.n_min_oos
        );
        let _ = writeln!(
            s,
            "  E_oos_pess : {}   SE {}   t {}   (bar: E - {t}*SE > 0)",
            opt(self.e_oos_pess_pips, "pips"),
            opt(self.se_oos_pips, "pips"),
            self.t_stat
                .map(|t| format!("{t:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
            t = OOS_T_STAT_MIN
        );
        let _ = writeln!(
            s,
            "  pbo_session: {}   (the LOOP'S OWN selection across sweeps)",
            self.pbo_session
                .map(|p| format!("{p:.4}"))
                .unwrap_or_else(|| "<refused>".to_string())
        );
        let _ = writeln!(
            s,
            "  cost band  : {}",
            if self.band_survives {
                "SurvivesBand (both edges)"
            } else {
                "did NOT survive both edges — a result that survives only at the optimistic edge \
                 is not a result"
            }
        );
        let _ = writeln!(
            s,
            "  goal metric: {:.4} vs bar {:.4}",
            self.goal_metric, self.goal_bar
        );
        if !self.frontier.is_empty() {
            let _ = writeln!(
                s,
                "  risk/trade  P(reach)  P(ruin)   median-end     mean-end"
            );
            for r in &self.frontier {
                let star = match self.best_risk_fraction {
                    Some(f) if (r.risk_fraction - f).abs() < 1e-9 => " <= max P(reach)",
                    _ => "",
                };
                let _ = writeln!(
                    s,
                    "  {:>6.0}%     {:>6.1}%   {:>6.1}%  {:>11.0}  {:>11.0}{}",
                    r.risk_fraction * 100.0,
                    r.p_reach_target * 100.0,
                    r.p_ruin * 100.0,
                    r.median_terminal,
                    r.mean_terminal,
                    star
                );
            }
        }
        if self.months_total > 0 {
            let _ = writeln!(
                s,
                "  months     : {}/{} cleared the monthly target",
                self.months_clearing_target, self.months_total
            );
        }
        for n in &self.notes {
            let _ = writeln!(s, "  note: {n}");
        }
        let _ = writeln!(
            s,
            "  VERDICT    : {}",
            if self.promoted {
                "PROMOTED — this is a PROPOSAL. The loop does not size a position, does not \
                 contact a broker and does not write live_portfolio.json."
                    .to_string()
            } else {
                format!(
                    "REFUSED at {}: {}",
                    self.failing_conjunct.as_deref().unwrap_or("<unnamed>"),
                    self.detail
                )
            }
        );
        let _ = writeln!(
            s,
            "  The out-of-sample window is now SPENT. A configuration that touches it more than \
             once has spent it."
        );
        s
    }
}

fn opt(v: Option<f64>, unit: &str) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:+.4} {unit}"),
        _ => "n/a".to_string(),
    }
}

/// Mean and standard error of a sample. `None` below two observations, where the
/// standard error does not exist.
fn mean_and_se(xs: &[f64]) -> Option<(f64, f64)> {
    if xs.len() < 2 || xs.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    if !var.is_finite() || var < 0.0 {
        return None;
    }
    Some((mean, (var / n).sqrt()))
}

/// Calendar-month key: `year * 12 + month` — the grid `trial_returns` and
/// `quality::analyze_monthly_consistency` both bucket by.
pub fn month_key(ms: i64) -> Option<i64> {
    Utc.timestamp_millis_opt(ms)
        .single()
        .map(|dt| (dt.year() as i64) * 12 + dt.month() as i64)
}

/// Path statistics that ARE derivable from a month-end return series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonthlyPathStats {
    /// The single worst month, as a positive loss fraction.
    pub worst_month_loss: f64,
    /// Peak-to-trough of the compounded month-end equity curve.
    pub max_drawdown: f64,
}

/// Compound a month-end return series and measure its drawdown.
///
/// Drawdown-path constraints are **not** a function of the per-trade mean: a
/// candidate can have identical expectancy and be either inside or outside the
/// rule. That is exactly why the prop-firm objective is a different objective
/// and not a re-weighting.
///
/// It deliberately does **not** claim to measure an intraday daily loss: the
/// input has month resolution, and a limit checked at the wrong resolution reads
/// as passed — which is worse than not checking it, because a reader takes it as
/// evidence.
pub fn path_stats(monthly_returns: &[f64]) -> MonthlyPathStats {
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;
    let mut worst = 0.0_f64;
    for r in monthly_returns.iter().copied().filter(|r| r.is_finite()) {
        if -r > worst {
            worst = -r;
        }
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
        }
        if peak > 0.0 {
            let dd = ((peak - equity) / peak).max(0.0);
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    MonthlyPathStats {
        worst_month_loss: worst.max(0.0),
        max_drawdown: max_dd,
    }
}

/// **Stage 2 — the promotion inequality.** One expression; it cannot drift into
/// prose.
///
/// ```text
/// PROMOTE(s) ⟺
///       oos_touches_spent(window)              = 0        (before this evaluation)
///    ∧  SCREEN(s)                              = true     (the candidate only exists if it did)
///    ∧  pbo_sweep(s)                           ≤ 0.50
///    ∧  pbo_session                            ≤ 0.50
///    ∧  dsr(s ; N_session)                     ≥ 0.95
///    ∧  excess_over_expected_max_per_period(s) > 0
///    ∧  E_oos_pess(s)                          > 0
///    ∧  E_oos_pess(s) − 2 · SE_oos(s)          > 0
///    ∧  band_verdict_oos(s)                    = SurvivesBand
///    ∧  n_oos(s)                               ≥ N_MIN_OOS
///    ∧  goal_metric(s)                         ≥ goal_bar(scenario)
///    ∧  scenario path constraints
/// ```
///
/// **Deviation from the design's written order, deliberately:** the touch-budget
/// conjunct is evaluated FIRST rather than tenth. A conjunction's truth value is
/// order-independent, but "did I have permission to look at this window" checked
/// after looking is not a guard.
///
/// `SCREEN(s)` does not appear as a runtime check because a
/// [`PromotionCandidate`] is only ever constructed from a passing screen
/// (`verdict::decide`); its statistics are carried on the candidate and
/// re-checked here.
pub fn promote(
    candidate: &PromotionCandidate,
    oos: &QuoteValidatedOosTouchEvidenceV1,
    th: &JudgeThresholds,
    _goals: &GoalSet,
    scenario: &Scenario,
    session: &Session,
) -> Result<PromotionOutcome> {
    let mut notes: Vec<String> = Vec::new();

    let stats = mean_and_se(oos.per_trade_net_pips());
    let e_oos = stats.map(|(m, _)| m);
    let se_oos = stats.map(|(_, s)| s);
    let t_stat = stats.and_then(|(m, s)| (s > 0.0).then_some(m / s));

    // Scenario metrics are computed whatever the outcome, so a refused promotion
    // still shows the operator the whole curve.
    let mut goal_metric = 0.0_f64;
    let mut goal_metric_available = false;
    let mut frontier: Vec<RiskOutcomeRecord> = Vec::new();
    let mut best_risk_fraction: Option<f64> = None;
    let mut p_ruin_at_best: Option<f64> = None;
    let mut months_total = 0usize;
    let mut months_clearing = 0usize;
    let mut path_constraint_failure: Option<String> = None;

    let goal_bar = match scenario {
        Scenario::Risky(r) => {
            if oos.r_multiples().is_empty() || oos.trades_per_day() <= 0.0 {
                notes.push(
                    "no out-of-sample R-multiples, so no goal projection was possible. That is a \
                     refusal, not a P(reach) of zero."
                        .to_string(),
                );
            } else {
                let report = build_report(
                    oos.r_multiples(),
                    r.start_balance_usd,
                    r.target_balance_usd,
                    f64::from(r.horizon_days),
                    oos.trades_per_day(),
                    DEFAULT_RISK_LEVELS,
                    session.opened.as_ref().map(|h| h.session_seed).unwrap_or(0),
                );
                frontier = report
                    .frontier
                    .iter()
                    .map(RiskOutcomeRecord::from)
                    .collect();
                best_risk_fraction = Some(report.best_risk_fraction);
                if let Some(best) = report
                    .frontier
                    .iter()
                    .find(|o| (o.risk_fraction - report.best_risk_fraction).abs() < 1e-9)
                {
                    goal_metric = best.p_reach_target;
                    goal_metric_available = true;
                    p_ruin_at_best = Some(best.p_ruin);
                    if best.p_ruin > th.p_ruin_max {
                        path_constraint_failure = Some(format!(
                            "P(ruin) {:.4} at the P(reach)-maximising risk of {:.0}% exceeds \
                             {:.2}. A target reached on a path that mostly blows up first is not \
                             a result.",
                            best.p_ruin,
                            report.best_risk_fraction * 100.0,
                            th.p_ruin_max
                        ));
                    }
                }
            }
            th.p_reach_min
        }
        Scenario::PropFirm(p) => {
            months_total = oos.monthly_returns().len();
            months_clearing = oos
                .monthly_returns()
                .iter()
                .filter(|r| **r >= p.monthly_profit_target_pct)
                .count();
            if months_total > 0 {
                goal_metric = months_clearing as f64 / months_total as f64;
                goal_metric_available = true;
            } else {
                path_constraint_failure = Some(
                    "the out-of-sample window contains no complete month, so the monthly pass \
                     rate does not exist"
                        .to_string(),
                );
            }
            let path = path_stats(oos.monthly_returns());
            if path.worst_month_loss > p.max_overall_drawdown_pct
                && path_constraint_failure.is_none()
            {
                path_constraint_failure = Some(format!(
                    "the worst out-of-sample month lost {:.2}%, beyond \
                     max_overall_drawdown_pct {:.2}%",
                    path.worst_month_loss * 100.0,
                    p.max_overall_drawdown_pct * 100.0
                ));
            }
            if path.max_drawdown > p.max_overall_drawdown_pct && path_constraint_failure.is_none() {
                path_constraint_failure = Some(format!(
                    "the compounded month-end drawdown reached {:.2}%, beyond \
                     max_overall_drawdown_pct {:.2}%",
                    path.max_drawdown * 100.0,
                    p.max_overall_drawdown_pct * 100.0
                ));
            }
            notes.push(format!(
                "the intraday daily-loss limit (max_daily_loss_pct {:.2}%) is NOT checked here: \
                 the out-of-sample evidence carries month-end returns, not an intraday equity \
                 path, and a limit checked at the wrong resolution is worse than one not checked \
                 at all because it reads as passed. See REQUIRED ELSEWHERE in \
                 docs/autoresearch-loop.md.",
                p.max_daily_loss_pct * 100.0
            ));
            p.min_pass_rate
        }
    };

    let (pbo_session, pbo_session_refusal) = session_pbo_value(session);

    let build = |failing: Option<PromotionConjunct>, detail: String| PromotionOutcome {
        sweep: candidate.sweep,
        slot: candidate.slot,
        window: oos.window(),
        promoted: failing.is_none(),
        failing_conjunct: failing.map(|c| c.label().to_string()),
        detail,
        n_trades: oos.per_trade_net_pips().len(),
        n_min_oos: th.n_min_oos,
        e_oos_pess_pips: e_oos,
        se_oos_pips: se_oos,
        t_stat,
        band_survives: oos.band_survives(),
        pbo_session,
        goal_metric,
        goal_bar,
        frontier: frontier.clone(),
        best_risk_fraction,
        p_ruin_at_best,
        months_total,
        months_clearing_target: months_clearing,
        notes: notes.clone(),
    };

    // ── the inequality, conjunct by conjunct ────────────────────────────────

    if session.oos_touches_spent > th.oos_touches_total {
        return Ok(build(
            Some(PromotionConjunct::OosTouchAlreadySpent),
            format!(
                "{} out-of-sample touch(es) have been spent on this window and the budget is {}. \
                 A configuration that touches the OOS window more than once has spent it.",
                session.oos_touches_spent, th.oos_touches_total
            ),
        ));
    }

    if candidate.pbo_sweep > th.pbo_max {
        return Ok(build(
            Some(PromotionConjunct::PboSweep),
            format!("pbo_sweep {:.4} > {:.2}", candidate.pbo_sweep, th.pbo_max),
        ));
    }

    let Some(pbo_session_value) = pbo_session.filter(|p| p.is_finite()) else {
        return Ok(build(
            Some(PromotionConjunct::PboSession),
            format!(
                "pbo_session refused — the loop's OWN selection procedure has no measurable \
                 overfitting probability. A refusal is a FAIL: an unmeasured selection procedure \
                 is not a sound one. {}",
                pbo_session_refusal.unwrap_or_default()
            ),
        ));
    };
    if pbo_session_value > th.pbo_max {
        return Ok(build(
            Some(PromotionConjunct::PboSession),
            format!(
                "pbo_session {pbo_session_value:.4} > {:.2} — the LOOP'S OWN selection across \
                 sweeps is worse than random, whatever this one sweep's internal PBO says.",
                th.pbo_max
            ),
        ));
    }

    match candidate.dsr {
        None => {
            return Ok(build(
                Some(PromotionConjunct::Dsr),
                "no deflated Sharpe at promotion time. A refusal is a FAIL.".to_string(),
            ));
        }
        Some(d) if !d.is_finite() || d < th.dsr_min => {
            return Ok(build(
                Some(PromotionConjunct::Dsr),
                format!("deflated Sharpe {d:.4} < {:.2}", th.dsr_min),
            ));
        }
        Some(_) => {}
    }
    if candidate.excess_over_expected_max <= 0.0 {
        return Ok(build(
            Some(PromotionConjunct::ExcessOverExpectedMax),
            format!(
                "excess over the expected maximum by luck is {:+.4} per period — the champion did \
                 not beat what the search produces from noise.",
                candidate.excess_over_expected_max
            ),
        ));
    }

    let (Some(e), Some(se)) = (e_oos, se_oos) else {
        return Ok(build(
            Some(PromotionConjunct::OosExpectancy),
            format!(
                "the out-of-sample expectancy has no standard error: {} trade(s) were recorded \
                 and at least 2 are needed. No number is never good news.",
                oos.per_trade_net_pips().len()
            ),
        ));
    };
    if e <= 0.0 {
        return Ok(build(
            Some(PromotionConjunct::OosExpectancy),
            format!(
                "E_oos_pess = {e:+.4} pips/trade at the pessimistic cost edge. Exit geometry \
                 redistributes between win rate and payoff; it never creates edge, and this is \
                 what that looks like out of sample."
            ),
        ));
    }
    if e - th.oos_t_stat_min * se <= 0.0 {
        return Ok(build(
            Some(PromotionConjunct::OosTStat),
            format!(
                "E_oos_pess {e:+.4} - {:.1}*SE {se:.4} = {:+.4} <= 0. The expectancy is positive \
                 but not distinguishable from zero on {} trade(s); this is the significance bar \
                 applied OUT of sample.",
                th.oos_t_stat_min,
                e - th.oos_t_stat_min * se,
                oos.per_trade_net_pips().len()
            ),
        ));
    }

    if !oos.band_survives() {
        return Ok(build(
            Some(PromotionConjunct::OosCostBand),
            "band_verdict_oos is not SurvivesBand. A result that survives only at the optimistic \
             edge is not a result."
                .to_string(),
        ));
    }

    if oos.per_trade_net_pips().len() < th.n_min_oos {
        return Ok(build(
            Some(PromotionConjunct::OosTradeCount),
            format!(
                "{} out-of-sample trade(s) < the derived floor of {}",
                oos.per_trade_net_pips().len(),
                th.n_min_oos
            ),
        ));
    }

    if !goal_metric_available || !goal_metric.is_finite() {
        return Ok(build(
            Some(PromotionConjunct::GoalMetric),
            "the goal metric could not be computed out of sample, so the goal has NOT been \
             reached. An absent metric is never a pass."
                .to_string(),
        ));
    }
    if goal_metric < goal_bar {
        return Ok(build(
            Some(PromotionConjunct::GoalMetric),
            match scenario {
                Scenario::Risky(_) => format!(
                    "P(reach target) = {goal_metric:.4} < {goal_bar:.2}. Below one half the \
                     MEDIAN PATH DOES NOT REACH THE TARGET, and a 'goal reached' claim backed by \
                     this number would be a lottery ticket sold as a result."
                ),
                Scenario::PropFirm(_) => format!(
                    "{months_clearing}/{months_total} months cleared the monthly target = \
                     {goal_metric:.4} < models.prop_firm_min_pass_rate {goal_bar:.4}"
                ),
            },
        ));
    }

    if let Some(detail) = path_constraint_failure {
        return Ok(build(
            Some(PromotionConjunct::ScenarioPathConstraint),
            detail,
        ));
    }

    Ok(build(
        None,
        format!(
            "every conjunct held out of sample: E_oos_pess {e:+.4} pips, t {}, band survives, \
             pbo_sweep {:.4}, pbo_session {pbo_session_value:.4}, goal metric {goal_metric:.4} >= \
             {goal_bar:.4}.",
            t_stat
                .map(|t| format!("{t:.2}"))
                .unwrap_or_else(|| "undefined (zero dispersion)".to_string()),
            candidate.pbo_sweep
        ),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shuffle::ControlKind;
    use neoethos_search::deflated::DeflatedSharpeReport;

    const N: usize = 12_345;

    fn dsr_report(value: f64, excess: f64, trials_n: usize) -> DeflatedSharpeReport {
        DeflatedSharpeReport {
            trials_n,
            trials_n_source: "test".to_string(),
            trials_with_sharpe: 100,
            periods: 36,
            period_unit: "month".to_string(),
            champion_strategy_id: "s".to_string(),
            champion_candidate_index: 0,
            raw_sharpe_per_period: 0.5,
            raw_sharpe_annualised: 1.7,
            skewness: 0.0,
            kurtosis: 3.0,
            sharpe_variance_across_trials: 0.01,
            expected_max_sharpe_per_period: 0.3,
            expected_max_sharpe_annualised: 1.0,
            excess_over_expected_max_per_period: excess,
            deflated_sharpe_ratio: value,
            assumptions: Vec::new(),
        }
    }

    fn pbo_report(value: f64) -> PboReport {
        PboReport {
            pbo: value,
            combinations: 70,
            groups: 8,
            group_choice_reason: "test".to_string(),
            periods: 36,
            smallest_group_periods: 4,
            largest_group_periods: 5,
            trials_ranked: 100,
            trials_excluded_constant: 0,
            degenerate_oos_evaluations: 0,
            mean_logit: -1.0,
            median_relative_rank: 0.2,
            assumptions: Vec::new(),
        }
    }

    fn statistics(
        dsr: Option<f64>,
        excess: f64,
        pbo: Option<f64>,
        n: usize,
    ) -> TrialStatisticsReport {
        TrialStatisticsReport {
            schema: "neoethos.trial_statistics.v1".to_string(),
            matrix_rows: 100,
            trials_offered: n,
            matrix_periods: 36,
            matrix_truncated: false,
            deflated_sharpe: dsr.map(|d| dsr_report(d, excess, n)),
            deflated_sharpe_refusal: dsr.is_none().then(|| "REFUSED: too short".to_string()),
            pbo: pbo.map(pbo_report),
            pbo_refusal: pbo.is_none().then(|| "REFUSED: too few trials".to_string()),
            notes: Vec::new(),
        }
    }

    fn band() -> CostBandCounts {
        CostBandCounts {
            survives: 3,
            optimistic_edge_only: 0,
            fails: 5,
            unmeasured: 0,
            not_discriminating: 0,
            discriminates: true,
        }
    }

    fn evidence(st: &TrialStatisticsReport) -> SearchEvidence<'_> {
        SearchEvidence {
            statistics: st,
            cost_band: band(),
            band_discriminates: true,
            e_screen_pess: Some(0.80),
            n_trades: 400,
            error: None,
        }
    }

    /// A report whose DSR can survive being re-deflated from a single search's
    /// trial count all the way up to the session-wide one.
    ///
    /// The shared `dsr_report` helper carries a cross-trial Sharpe variance
    /// (0.01) large enough that SR* swamps the champion at any realistic N, so
    /// it can prove a REFUSAL but never a PASS. This one is a plausible sweep:
    /// a champion at 0.5 per period against a tight spread of trials.
    fn redeflatable(trials_offered: usize) -> TrialStatisticsReport {
        let mut st = statistics(Some(0.99), 0.20, Some(0.20), trials_offered);
        {
            let d = st.deflated_sharpe.as_mut().unwrap();
            d.trials_n = 2;
            d.periods = 36;
            d.raw_sharpe_per_period = 0.5;
            d.skewness = 0.0;
            d.kurtosis = 3.0;
            d.sharpe_variance_across_trials = 1.0e-4;
        }
        // SR*, the excess and the DSR are recomputed from those moments, so the
        // report is internally consistent at `trials_offered` rather than a set
        // of hand-written numbers that could not have come from one matrix.
        let consistent = neoethos_search::deflated::redeflate_sharpe(
            st.deflated_sharpe.as_ref().unwrap(),
            trials_offered,
        )
        .expect("a well-formed report re-deflates");
        st.deflated_sharpe = Some(consistent);
        st
    }

    /// **THE defect that made the screen unsatisfiable.**
    ///
    /// `screen` requires the report's `trials_offered` to BE `n_session`, and
    /// `n_session` is a session-wide fold that no single search can know. A
    /// measured run of 71 searches produced ZERO passes for exactly this reason
    /// — every one failed S2 with "deflated against 40 but the session-wide
    /// count is 1080" — which left `best_ever` permanently `None` and the loop
    /// unable to report either goal reached or goal unreachable.
    ///
    /// The contract stays. What this proves is that it is SATISFIABLE, through
    /// the one call the runner makes after `SweepCompleted` is appended.
    #[test]
    fn the_n_contract_is_satisfiable_by_redeflating_against_the_session_fold() {
        let as_the_executor_produced_it = redeflatable(40);

        // 1. Unfixed: refused, and named against S2 with both numbers shown.
        match screen(
            evidence(&as_the_executor_produced_it),
            &JudgeThresholds::frozen(),
            N,
            &ready_null(),
        ) {
            ScreenResult::Failed { conjunct, detail } => {
                assert_eq!(conjunct, ScreenConjunct::S2Dsr);
                assert!(detail.contains("trials_offered = 40"), "{detail}");
                assert!(detail.contains(&N.to_string()), "{detail}");
            }
            other => panic!("a report deflated against its own N must fail: {other:?}"),
        }

        // 2. Fixed, by the sanctioned call — and it PASSES. If this ever stops
        //    passing, no configuration in any session can ever be screened.
        let installed = as_the_executor_produced_it.redeflated_against(N);
        assert_eq!(installed.trials_offered, N);
        let result = screen(
            evidence(&installed),
            &JudgeThresholds::frozen(),
            N,
            &ready_null(),
        );
        assert!(
            result.passed(),
            "the screen must be reachable at all — it was not, for 71 measured searches in a \
             row: {}",
            result.detail()
        );

        // 3. And the fix deflates rather than flatters: the bar the champion had
        //    to clear by luck alone is HIGHER at the session-wide N.
        let before = as_the_executor_produced_it
            .deflated_sharpe
            .as_ref()
            .unwrap();
        let after = installed.deflated_sharpe.as_ref().unwrap();
        assert!(
            after.expected_max_sharpe_per_period > before.expected_max_sharpe_per_period,
            "SR* must rise with N"
        );
        assert!(after.deflated_sharpe_ratio <= before.deflated_sharpe_ratio);
    }

    /// An unreadable matrix is a disk or wiring finding. Reporting it as an
    /// N-contract violation files it under a cause that is not its own — and
    /// that is what happened to every not-run row, because `unreadable()` sets
    /// `trials_offered = 0` and the N check used to run first.
    #[test]
    fn an_unreadable_report_is_named_by_its_own_cause_not_by_the_n_contract() {
        let st = TrialStatisticsReport::unreadable("the ledger file is absent".to_string());
        match screen(evidence(&st), &JudgeThresholds::frozen(), N, &ready_null()) {
            ScreenResult::Failed { conjunct, detail } => {
                assert_eq!(conjunct, ScreenConjunct::S1Pbo);
                assert!(detail.contains("UNREADABLE"), "{detail}");
                assert!(detail.contains("the ledger file is absent"), "{detail}");
                assert!(
                    !detail.contains("trials_offered = 0"),
                    "the real cause must not be hidden behind the N contract: {detail}"
                );
            }
            other => panic!("an unreadable matrix is never a pass: {other:?}"),
        }
    }

    /// §15 — the derived floors must be enforced, not merely computed.
    ///
    /// `with_derived_floors` had ZERO non-test callers: both floors were derived
    /// for the startup self-check and then the judge applied the 30-trade
    /// placeholder. On a two-year OOS window at half a trade a day the derived
    /// floor is ~360, so the bar actually applied was twelve times weaker than
    /// the startup message said it had verified.
    #[test]
    fn the_derived_screen_floor_is_what_the_judge_actually_applies() {
        let st = redeflatable(N);
        let ev = SearchEvidence {
            n_trades: 100,
            ..evidence(&st)
        };

        // Under the placeholder, 100 trades clears 30 and the screen passes.
        assert!(screen(ev, &JudgeThresholds::frozen(), N, &ready_null()).passed());

        // Under the floor derived from a two-year window it does not, and the
        // failure is named against S5 with the number it had to beat.
        let derived = JudgeThresholds::frozen().with_derived_floors(360, 360);
        match screen(ev, &derived, N, &ready_null()) {
            ScreenResult::Failed { conjunct, detail } => {
                assert_eq!(conjunct, ScreenConjunct::S5Trades);
                assert!(detail.contains("360"), "{detail}");
            }
            other => panic!("100 trades must not clear a floor of 360: {other:?}"),
        }
    }

    fn ready_null() -> ShuffleNull {
        let mut null = ShuffleNull::new();
        for e in [0.10, 0.20, 0.30] {
            null.observe(ControlKind::CircularRotation, Some(e));
        }
        null
    }

    fn search_record(slot: usize, hash: &str, e: f64) -> crate::journal::SearchRecord {
        crate::journal::SearchRecord {
            slot,
            config_hash: hash.to_string(),
            trials_offered: N,
            survivors: 3,
            e_screen_pess: Some(e),
            n_trades: 400,
            dsr: Some(0.99),
            pbo: Some(0.20),
            dsr_refusal: None,
            pbo_refusal: None,
            cost_band: band(),
            rejections: Vec::new(),
            streamed: true,
            batch_columns: 0,
            next_cursor: 0,
            wall_ms: 1,
            error: None,
        }
    }

    fn champion_row(hash: &str, id: &str, returns: Vec<f64>) -> ChampionRow {
        ChampionRow {
            sweep: SweepId(4),
            config_hash: hash.to_string(),
            strategy_id: id.to_string(),
            period_keys: (0..returns.len() as i64).map(|k| 24_300 + k).collect(),
            monthly_returns: returns,
            e_screen_pess: 0.0,
        }
    }

    /// THE defect: two argmaxes, one row, and the stamping hid the difference.
    ///
    /// The highest-expectancy slot is typically the overfit outlier the screen
    /// rejects. The old code took THAT slot's return series, overwrote its
    /// `config_hash` and `e_screen_pess` with the passing slot's, and handed the
    /// result to `pbo_session` — the statistic that gates every promotion.
    #[test]
    fn the_champion_row_comes_from_the_slot_that_passed_not_from_the_outlier() {
        let passing = statistics(Some(0.99), 0.20, Some(0.20), N);
        // The outlier scores far higher and fails S1 — the routine case.
        let rejected = statistics(Some(0.999), 5.00, Some(0.99), N);

        let evidence = SweepEvidence {
            sweep: SweepId(4),
            kind: crate::journal::SweepKind::Live,
            sweep_hash: "fnv64:sweep".to_string(),
            per_search: vec![
                search_record(0, "fnv64:passed", 0.80),
                search_record(1, "fnv64:rejected", 5.00),
            ],
            statistics: vec![passing, rejected],
            champion_rows: vec![
                Some(champion_row(
                    "fnv64:passed",
                    "honest",
                    vec![0.01, 0.02, 0.03],
                )),
                Some(champion_row(
                    "fnv64:rejected",
                    "outlier",
                    vec![9.0, 9.0, 9.0],
                )),
            ],
            trials_offered: N,
            wall_ms: 1,
        };

        let screen = screen_sweep(&evidence, &JudgeThresholds::frozen(), N, &ready_null());
        assert!(screen.per_slot[0].passed(), "{:?}", screen.per_slot[0]);
        assert!(!screen.per_slot[1].passed(), "the outlier must fail S1");
        assert_eq!(screen.champion_slot, Some(0));

        let row = screen.champion.expect("the passing slot offers a row");
        assert_eq!(row.config_hash, "fnv64:passed");
        // The SERIES is what `pbo_session` is computed from, and it is the one
        // that belongs to the identity beside it.
        assert_eq!(row.strategy_id, "honest");
        assert_eq!(row.monthly_returns, vec![0.01, 0.02, 0.03]);
        assert!(screen.champion_refusal.is_none());
    }

    /// A passing slot with no return series contributes NOTHING, and says so.
    /// Substituting another slot's series is the defect above, wearing a
    /// different hat.
    #[test]
    fn a_passing_slot_with_no_series_contributes_no_row_and_names_the_reason() {
        let passing = statistics(Some(0.99), 0.20, Some(0.20), N);
        let rejected = statistics(Some(0.999), 5.00, Some(0.99), N);
        let evidence = SweepEvidence {
            sweep: SweepId(4),
            kind: crate::journal::SweepKind::Live,
            sweep_hash: "fnv64:sweep".to_string(),
            per_search: vec![
                search_record(0, "fnv64:passed", 0.80),
                search_record(1, "fnv64:rejected", 5.00),
            ],
            statistics: vec![passing, rejected],
            champion_rows: vec![
                None,
                Some(champion_row(
                    "fnv64:rejected",
                    "outlier",
                    vec![9.0, 9.0, 9.0],
                )),
            ],
            trials_offered: N,
            wall_ms: 1,
        };

        let screen = screen_sweep(&evidence, &JudgeThresholds::frozen(), N, &ready_null());
        assert!(
            screen.champion.is_none(),
            "the reject's series must not stand in"
        );
        let why = screen.champion_refusal.expect("the drop must be NAMED");
        assert!(why.contains("no per-period champion series"), "{why}");
        // best-ever is NOT gated on the row: R2's U2 still gets its number.
        assert_eq!(screen.champion_config_hash.as_deref(), Some("fnv64:passed"));
        assert_eq!(screen.champion_e_screen_pess, Some(0.80));
    }

    #[test]
    fn a_clean_search_passes_every_conjunct() {
        let st = statistics(Some(0.99), 0.20, Some(0.20), N);
        let out = screen(evidence(&st), &JudgeThresholds::frozen(), N, &ready_null());
        assert!(out.passed(), "{out:?}");
        assert_eq!(out.reward(), 1);
        assert_eq!(out.failing_conjunct_label(), None);
    }

    #[test]
    fn only_a_measured_search_census_can_clear_stage_one_cost_band_screening() {
        let st = statistics(Some(0.99), 0.20, Some(0.20), N);
        let measured = neoethos_search::discovery::CostBandCensus {
            survives: 1,
            optimistic_edge_only: 0,
            fails: 2,
            unmeasured: 0,
            not_discriminating: 0,
        };
        let mut measured_evidence = evidence(&st);
        measured_evidence.cost_band = CostBandCounts::from_census(&measured, true);
        let passed = screen(
            measured_evidence,
            &JudgeThresholds::frozen(),
            N,
            &ready_null(),
        );
        assert!(
            passed.passed(),
            "measured survivor must reach Stage-1: {passed:?}"
        );

        for refused in [
            neoethos_search::discovery::CostBandCensus::default(),
            neoethos_search::discovery::CostBandCensus {
                unmeasured: 1,
                ..neoethos_search::discovery::CostBandCensus::default()
            },
        ] {
            let mut refused_evidence = evidence(&st);
            refused_evidence.cost_band = CostBandCounts::from_census(&refused, true);
            assert!(matches!(
                screen(
                    refused_evidence,
                    &JudgeThresholds::frozen(),
                    N,
                    &ready_null(),
                ),
                ScreenResult::Failed {
                    conjunct: ScreenConjunct::S3CostBand,
                    ..
                }
            ));
        }
    }

    #[test]
    fn a_pbo_refusal_is_a_fail_and_never_a_pass() {
        // "No number is never good news." An unknown overfitting probability is
        // not a low one.
        let st = statistics(Some(0.99), 0.2, None, N);
        match screen(evidence(&st), &JudgeThresholds::frozen(), N, &ready_null()) {
            ScreenResult::Failed { conjunct, detail } => {
                assert_eq!(conjunct, ScreenConjunct::S1Pbo);
                assert!(detail.contains("A refusal is a FAIL"));
            }
            other => panic!("expected S1, got {other:?}"),
        }
    }

    #[test]
    fn pbo_above_one_half_discards_the_sweep_whatever_its_headline() {
        let st = statistics(Some(0.999), 5.0, Some(0.51), N);
        let mut ev = evidence(&st);
        ev.e_screen_pess = Some(99.0);
        match screen(ev, &JudgeThresholds::frozen(), N, &ready_null()) {
            ScreenResult::Failed { conjunct, detail } => {
                assert_eq!(conjunct, ScreenConjunct::S1Pbo);
                assert!(detail.contains("worse than random"));
            }
            other => panic!("expected S1, got {other:?}"),
        }
    }

    #[test]
    fn s2b_catches_a_champion_that_did_not_beat_luck_even_when_the_dsr_passes() {
        // The DSR's cross-trial variance comes from this sweep's rows while N is
        // session-wide, which FLATTERS it. S2b does not depend on that variance.
        let st = statistics(Some(0.99), -0.01, Some(0.2), N);
        match screen(evidence(&st), &JudgeThresholds::frozen(), N, &ready_null()) {
            ScreenResult::Failed { conjunct, detail } => {
                assert_eq!(conjunct, ScreenConjunct::S2Dsr);
                assert!(detail.contains("S2b"));
                assert!(detail.contains("biases the DSR UPWARD"));
            }
            other => panic!("expected S2b, got {other:?}"),
        }
    }

    #[test]
    fn statistics_deflated_against_the_wrong_n_are_refused_outright() {
        // The quietest way this judge could be defeated.
        let st = statistics(Some(0.99), 0.2, Some(0.2), 100);
        match screen(evidence(&st), &JudgeThresholds::frozen(), N, &ready_null()) {
            ScreenResult::Failed { detail, .. } => {
                assert!(detail.contains("deflation against the wrong N"));
            }
            other => panic!("expected the N mismatch to be refused, got {other:?}"),
        }
    }

    #[test]
    fn a_non_discriminating_cost_band_fails_rather_than_reading_clean() {
        let st = statistics(Some(0.99), 0.2, Some(0.2), N);
        let mut ev = evidence(&st);
        ev.band_discriminates = false;
        match screen(ev, &JudgeThresholds::frozen(), N, &ready_null()) {
            ScreenResult::Failed { conjunct, detail } => {
                assert_eq!(conjunct, ScreenConjunct::S3CostBand);
                assert!(detail.contains("worse than no census"));
            }
            other => panic!("expected S3, got {other:?}"),
        }
    }

    #[test]
    fn a_candidate_profitable_only_at_the_cheap_edge_is_not_a_result() {
        let st = statistics(Some(0.99), 0.2, Some(0.2), N);
        let mut ev = evidence(&st);
        ev.cost_band.optimistic_edge_only = 2;
        assert!(matches!(
            screen(ev, &JudgeThresholds::frozen(), N, &ready_null()),
            ScreenResult::Failed {
                conjunct: ScreenConjunct::S3CostBand,
                ..
            }
        ));
    }

    #[test]
    fn the_measured_base_rate_fails_s4() {
        // -4.15 pips/trade is what this instrument does in every exit
        // configuration tested. It must never reach the OOS window.
        let st = statistics(Some(0.99), 0.2, Some(0.2), N);
        let mut ev = evidence(&st);
        ev.e_screen_pess = Some(-4.15);
        match screen(ev, &JudgeThresholds::frozen(), N, &ready_null()) {
            ScreenResult::Failed { conjunct, detail } => {
                assert_eq!(conjunct, ScreenConjunct::S4Expectancy);
                assert!(detail.contains("-4.15"));
            }
            other => panic!("expected S4, got {other:?}"),
        }
    }

    #[test]
    fn a_candidate_that_scores_like_the_shuffle_control_fails_s6() {
        let st = statistics(Some(0.99), 0.2, Some(0.2), N);
        let mut ev = evidence(&st);
        ev.e_screen_pess = Some(0.25);
        match screen(ev, &JudgeThresholds::frozen(), N, &ready_null()) {
            ScreenResult::Failed { conjunct, detail } => {
                assert_eq!(conjunct, ScreenConjunct::S6Shuffle);
                assert!(detail.contains("mining noise"));
            }
            other => panic!("expected S6, got {other:?}"),
        }
    }

    #[test]
    fn without_a_null_the_screen_is_unavailable_and_never_a_pass() {
        let st = statistics(Some(0.99), 0.2, Some(0.2), N);
        let out = screen(
            evidence(&st),
            &JudgeThresholds::frozen(),
            N,
            &ShuffleNull::new(),
        );
        assert!(out.null_unavailable());
        assert!(!out.passed());
        assert_eq!(out.reward(), 0);
        assert_eq!(
            out.census_index(),
            Some(ScreenConjunct::Unavailable.index())
        );
        assert!(out.e_screen_pess().is_none());
    }

    #[test]
    fn a_real_failure_is_named_even_while_the_null_is_unavailable() {
        // Reporting `Unavailable` for a candidate that also failed S1 would lose
        // the real cause from the journal.
        let st = statistics(Some(0.99), 0.2, Some(0.9), N);
        assert!(matches!(
            screen(
                evidence(&st),
                &JudgeThresholds::frozen(),
                N,
                &ShuffleNull::new()
            ),
            ScreenResult::Failed {
                conjunct: ScreenConjunct::S1Pbo,
                ..
            }
        ));
    }

    #[test]
    fn a_failed_search_is_judged_as_a_failure_not_skipped() {
        let st = statistics(Some(0.99), 0.2, Some(0.2), N);
        let mut ev = evidence(&st);
        ev.error = Some("CUDA out of memory");
        assert!(!screen(ev, &JudgeThresholds::frozen(), N, &ready_null()).passed());
    }

    #[test]
    fn every_conjunct_has_a_distinct_census_slot() {
        let all = [
            ScreenConjunct::S1Pbo,
            ScreenConjunct::S2Dsr,
            ScreenConjunct::S3CostBand,
            ScreenConjunct::S4Expectancy,
            ScreenConjunct::S5Trades,
            ScreenConjunct::S6Shuffle,
            ScreenConjunct::Unavailable,
        ];
        let mut seen = [false; 7];
        for c in all {
            assert!(!seen[c.index()], "duplicate census slot for {c:?}");
            seen[c.index()] = true;
            assert_eq!(c.label(), ScreenConjunct::ALL_LABELS[c.index()]);
        }
        assert!(seen.iter().all(|x| *x));
    }

    #[test]
    fn changing_a_threshold_changes_the_judge_hash() {
        let a = JudgeThresholds::frozen();
        let mut b = a.clone();
        b.pbo_max = 0.60;
        assert_ne!(a.judge_hash(), b.judge_hash());
        assert_eq!(a.judge_hash(), JudgeThresholds::frozen().judge_hash());
        // The derived floors are part of the identity too: a session cannot
        // resume under a different trade floor without saying so.
        assert_ne!(
            a.judge_hash(),
            a.clone().with_derived_floors(90, 40).judge_hash()
        );
    }

    #[test]
    fn a_derived_trade_floor_is_never_zero() {
        assert_eq!(n_min_trades(0.0, 36.0), 1);
        assert_eq!(n_min_trades(2.5, 36.0), 90);
        assert_eq!(n_min_trades(f64::NAN, 36.0), 1);
        // A partial trade rounds up...
        assert_eq!(n_min_trades(15.0, 2.5), 38, "37.5 trades is a floor of 38");
        // ...but floating-point dust does not manufacture one: 15 x (800/30)
        // evaluates to 400.000000000000057 and the floor is 400, not 401.
        assert_eq!(n_min_trades(15.0, 800.0 / 30.0), 400);
        assert_eq!(
            JudgeThresholds::frozen()
                .with_derived_floors(0, 0)
                .n_min_oos,
            1
        );
    }

    // ── the session champion matrix ─────────────────────────────────────────

    fn champion(sweep: u64, keys: &[i64], returns: &[f64]) -> ChampionRow {
        ChampionRow {
            sweep: SweepId(sweep),
            config_hash: format!("fnv64:{sweep:016x}"),
            strategy_id: format!("g{sweep}"),
            period_keys: keys.to_vec(),
            monthly_returns: returns.to_vec(),
            e_screen_pess: 0.5,
        }
    }

    #[test]
    fn the_session_champion_matrix_is_built_in_memory_from_public_fields() {
        let keys: Vec<i64> = (24_300..24_336).collect();
        let rows: Vec<ChampionRow> = (0..12)
            .map(|i| {
                let r: Vec<f64> = (0..36)
                    .map(|m| 0.01 * ((i + m) % 7) as f64 - 0.02)
                    .collect();
                champion(i as u64, &keys, &r)
            })
            .collect();
        let built = build_session_champion_matrix(&rows, 10_000.0).unwrap();
        assert_eq!(built.matrix.rows.len(), 12);
        assert_eq!(built.matrix.periods(), 36);
        assert_eq!(built.matrix.flags & 1, 1, "returns are balance fractions");
        assert!(!built.matrix.is_truncated());
        assert_eq!(built.rows_dropped, 0);
        // …and CSCV accepts it, which is what pbo_session depends on.
        assert!(pbo_cscv(&built.matrix).is_ok());
    }

    #[test]
    fn champions_are_aligned_on_the_months_they_share() {
        let a = champion(0, &[100, 101, 102], &[0.1, 0.2, 0.3]);
        let b = champion(1, &[101, 102, 103], &[0.4, 0.5, 0.6]);
        let built = build_session_champion_matrix(&[a, b], 1000.0).unwrap();
        assert_eq!(built.matrix.period_keys, vec![101, 102]);
        assert_eq!(built.matrix.rows[0].returns, vec![0.2, 0.3]);
        assert_eq!(built.matrix.rows[1].returns, vec![0.4, 0.5]);
        assert!(built.notes.iter().any(|n| n.contains("intersection")));
    }

    #[test]
    fn an_unusable_champion_is_dropped_and_named() {
        let good = champion(0, &[100, 101], &[0.1, 0.2]);
        let ragged = champion(1, &[100, 101], &[0.1]);
        let nan = champion(2, &[100, 101], &[f64::NAN, 0.2]);
        let built = build_session_champion_matrix(&[good, ragged, nan], 1000.0).unwrap();
        assert_eq!(built.rows_dropped, 2);
        assert_eq!(built.matrix.rows.len(), 1);
        assert!(built.notes.iter().any(|n| n.contains("non-finite")));
    }

    #[test]
    fn champions_with_no_shared_month_are_refused_not_padded() {
        let a = champion(0, &[100, 101], &[0.1, 0.2]);
        let b = champion(1, &[200, 201], &[0.1, 0.2]);
        let err = build_session_champion_matrix(&[a, b], 1000.0).unwrap_err();
        assert!(err.contains("share no calendar month"));
    }

    #[test]
    fn a_session_with_too_few_champions_has_no_measurable_pbo() {
        // Emergent and intended: no promotion before CSCV has enough rows. A
        // selection procedure with four observations has no measurable
        // overfitting probability, and a refusal is a FAIL.
        let keys: Vec<i64> = (24_300..24_336).collect();
        let rows: Vec<ChampionRow> = (0..4).map(|i| champion(i, &keys, &[0.01; 36])).collect();
        assert!(session_pbo(&rows, 1.0).is_err());
    }

    // ── path statistics ─────────────────────────────────────────────────────

    #[test]
    fn the_monthly_drawdown_is_measured_on_the_compounded_curve() {
        let ps = path_stats(&[0.2, -0.5, 0.1]);
        assert!((ps.worst_month_loss - 0.5).abs() < 1e-12);
        // 1.0 -> 1.2 -> 0.6 -> 0.66; peak 1.2, trough 0.6 => 50%.
        assert!((ps.max_drawdown - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_flat_series_has_no_drawdown_and_no_worst_month() {
        let ps = path_stats(&[0.0, 0.0]);
        assert_eq!(ps.worst_month_loss, 0.0);
        assert_eq!(ps.max_drawdown, 0.0);
    }

    #[test]
    fn month_keys_match_the_grid_the_rest_of_the_tree_buckets_by() {
        let jan = Utc
            .with_ymd_and_hms(2020, 1, 15, 0, 0, 0)
            .unwrap()
            .timestamp_millis();
        assert_eq!(month_key(jan), Some(2020 * 12 + 1));
    }
}
