//! The proposer — Thompson sampling on marginal Beta posteriors, with an
//! exploration floor that never anneals.
//!
//! # Where the state lives
//!
//! **The proposer holds no history.** The posterior and the coverage counters
//! are folds over the journal, owned by [`crate::session::Session`]; this module
//! reads them and adds its own `Beta(1, 1)` prior. That split is what makes the
//! block retraction exact and `N_session` un-resettable: there is no counter
//! here that a restart could zero.
//!
//! # Why marginals, and not something cleverer
//!
//! Axis A is eleven categorical factors with 3–5 levels each; the refusal vector
//! adds four more with three levels; axis B adds nine variants with up to five
//! parameters. The full product is of order 10⁶–10⁷ configurations against a
//! budget of order 10⁴ searches. That ratio decides the algorithm before any
//! taste enters: **only marginal (per-level) effects are estimable** — anything
//! that tried to estimate interactions would be fitting cells of size one.
//!
//! Rejected, with reasons, so nobody re-opens it:
//!
//! * **ε-greedy** — needs a tuned ε and an annealing schedule, which are exactly
//!   the knobs that let a design drift toward early exploitation.
//! * **UCB** — its confidence term needs a bounded reward scale and behaves
//!   badly when most arms have reward ≈ 0, which is the expected regime here.
//! * **Full-configuration bandits** — 10⁶ arms against a 10⁴ budget: every arm
//!   pulled at most once.
//! * **Bayesian optimisation with a GP** — the response surface is categorical
//!   and the observation noise dominates the signal; a GP would fit the noise,
//!   and would add a dependency for the privilege.
//!
//! # The reward is binary, and it is shuffle-corrected before it arrives
//!
//! `reward(p) = 1` if the proposal passed the screen, `0` otherwise. Not the
//! screen's continuous score: on this data the continuous in-sample score is
//! dominated by selection noise, so a proposer hill-climbing on it converges on
//! the first lucky draw. *"Did anything survive a deflated, shuffle-corrected
//! screen"* is far more robust, and Beta–Bernoulli's posterior width gives
//! exploration proportional to ignorance for free — no temperature to drift.
//!
//! Three anti-overfit mechanisms, all mechanical:
//!
//! 1. **The screen's conjunct S6 is inside the reward channel.** While the
//!    shuffle null has fewer than `MIN_NULL_OBS` observations the screen reports
//!    [`ScreenSignal::null_unavailable`], and an unavailable screen updates the
//!    posterior **not at all** — [`Proposer::credits_for`] returns an empty
//!    credit list. The first blocks are pure exploration by construction, not by
//!    a tuned schedule.
//! 2. **A hard exploration floor of [`EXPLORATION_FLOOR`] uniform draws per
//!    sweep, forever.** A proposer whose exploration decays converges on its own
//!    first mistake, and this loop runs for days on a noisy signal.
//! 3. **One success per level per sweep, and a block veto.** A level gets at
//!    most one success credited per sweep however many of the 100 proposals
//!    carried it; and when a block is later judged indistinguishable from its
//!    shuffle control, `PosteriorLedger::retract` flips every success it
//!    credited to a failure by replaying that block's own credits.

use std::collections::HashMap;

use anyhow::{Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

use neoethos_search::discovery::{DiscoveryConfig, DiscoveryMode};
use neoethos_search::run_identity::{PayoffCeilingInputs, payoff_inputs_for_config};

use crate::journal::PosteriorCredit;
use crate::objective::{
    ObjectiveChoice, ObjectiveError, ObjectiveVariant, RefusalDim, RefusalLevels, ScenarioKind,
};
use crate::proposal::{
    Proposal, ProposalKey, ProposalOrigin, ProposalRefused, ProposedRunSpec, ProposerCensus,
    SweepId, materialise,
};
use crate::session::Session;
use crate::space::{
    B_MIN_DRAWS, Capabilities, EXPLORATION_FLOOR, FACTOR_COUNT, FactorId, MAX_DELTA_FACTORS,
    PROPOSER_RETRIES, SWEEP_SEARCHES, SearchConfigDelta, SpaceError, SpaceReport, axis_a_report,
    identity_source, lane_horizon_conflict,
};

/// How many named examples the proposer's own census keeps before it stops
/// appending. `proposal::ProposerCensus::record_refusal` enforces the same bound
/// on the refusal path; this one exists because the *repair* paths
/// (`trust_region_reverted`, `constraint_resampled`) push their names directly,
/// and an unbounded example list on a days-long session would turn the census
/// into the artifact. `ProposerCensus::absorb` re-applies the bound on the way
/// up, so this is a floor on cost, never the thing that decides what is kept.
const CENSUS_EXAMPLES_MAX: usize = 32;

/// Push a named example, bounded. A count says how many were abandoned; the name
/// says which question to ask next, and requirement (b) is BOTH.
fn note_example(census: &mut ProposerCensus, what: String, which: String) {
    if census.examples.len() < CENSUS_EXAMPLES_MAX {
        census.examples.push((what, which));
    }
}

/// Journal an identity refusal under the record that is TRUE of it.
///
/// The two facts are separate and `ProposerCensus` already keeps separate
/// counters for them, but the JOURNAL used to conflate them — and the journal is
/// the only durable record, since the proposer's census lives in memory and dies
/// with the process.
///
/// A `Record::ProposalDuplicate` needs a real `first_seen_sweep`. A
/// **replicate-budget exhaustion has none**: the cell was never run under this
/// key. The old code wrote `first_seen.unwrap_or(sweep)` — naming the CURRENT
/// sweep as the first sighting of a configuration that had never been run — so
/// on the fold it became `census.duplicates_refused` and "we have enough
/// variance estimates for this cell" and "we ran this exact experiment before"
/// became one number under a fabricated attribution. Now the first is a
/// duplicate record and the second is a refusal record carrying its own reason
/// text.
fn record_identity_refusal(
    out: &mut DrawnSweep,
    slot: usize,
    config_hash: &str,
    describe: &str,
    reason: ProposalRefused,
    first_seen: Option<SweepId>,
) {
    out.census.record_refusal(&reason, describe);
    match first_seen {
        Some(first_seen_sweep) => out.duplicates.push(DuplicateDraw {
            slot,
            config_hash: config_hash.to_string(),
            first_seen_sweep,
        }),
        None => out.refused.push(RefusedDraw {
            slot,
            config_hash: config_hash.to_string(),
            reason,
        }),
    }
}

/// U5 fired at `from_slot`: count and name every slot the sweep will now never
/// draw.
///
/// `break 'slots` used to be the whole of it. The sweep came back with fewer
/// than [`SWEEP_SEARCHES`] proposals and the ONLY trace of the abandoned slots
/// was the difference between the declared size and the actual one — inside the
/// same function whose other early exit says, in as many words, *"Refusing to
/// shorten the sweep silently — a sweep of fewer than 100 searches is not the
/// experiment the judge is told it is."* One record per abandoned slot, so the
/// journal carries them and the session's own census folds them.
fn abandon_remaining_slots(
    out: &mut DrawnSweep,
    from_slot: usize,
    consecutive_collisions: usize,
) {
    out.exhausted = true;
    out.slots_abandoned = SWEEP_SEARCHES.saturating_sub(from_slot);
    let drawn = out.proposals.len();
    for abandoned in from_slot..SWEEP_SEARCHES {
        let reason = ProposalRefused::Space(SpaceError::SpaceEnumerated {
            slot: abandoned,
            drawn,
            of: SWEEP_SEARCHES,
            consecutive_collisions,
        });
        out.census.record_refusal(&reason, "<never drawn: the space is enumerated (U5)>");
        out.refused.push(RefusedDraw { slot: abandoned, config_hash: String::new(), reason });
    }
}

/// Replicates of one cell before it counts as exhausted.
///
/// **A proposer decision the design left open, made here and named.** A
/// replicate is a re-draw of the same `(axis_a, axis_b, refusals)` cell under a
/// different GA seed, and within-cell variance is exactly what the judge needs
/// to separate *"this configuration is better"* from *"this draw was luckier"*.
/// But an unbounded replicate dimension means no draw is ever a duplicate, the
/// dedupe never bites, and U5 (`ProposerExhausted` — the reachable space is
/// enumerated) can never fire. Three is enough for a variance estimate and few
/// enough that the space still exhausts.
pub const MAX_REPLICATES_PER_CELL: u8 = 3;

/// A cell: everything except the replicate seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellKey {
    pub axis_a: [u8; FACTOR_COUNT],
    pub variant: ObjectiveVariant,
    pub param: u8,
    pub refusals: [u8; 4],
}

impl CellKey {
    pub fn of(p: &Proposal) -> Self {
        let k = p.key();
        Self { axis_a: k.axis_a, variant: k.variant, param: k.param, refusals: k.refusals }
    }
}

/// The screen's verdict for one search, as the reward channel needs it.
///
/// A trait rather than a concrete type so the judge owns its own result shape:
/// the proposer needs exactly two bits from it and should not force a name on
/// the module that computes it.
pub trait ScreenSignal {
    /// Every conjunct S1–S6 held.
    fn passed(&self) -> bool;
    /// The shuffle null was not yet available, so the posterior must not be
    /// updated **at all** rather than credited a zero.
    fn null_unavailable(&self) -> bool;
}

// The judge's concrete result type is `ScreenResult` (`judge.rs`), a three-way
// enum — Passed / Failed / Unavailable. The trait above takes exactly the two
// bits the reward channel needs and nothing else.
impl ScreenSignal for crate::judge::ScreenResult {
    fn passed(&self) -> bool {
        crate::judge::ScreenResult::passed(self)
    }
    fn null_unavailable(&self) -> bool {
        crate::judge::ScreenResult::null_unavailable(self)
    }
}

/// A duplicate the sweep refused and redrew. Journalled as
/// `Record::ProposalDuplicate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplicateDraw {
    pub slot: usize,
    pub config_hash: String,
    pub first_seen_sweep: SweepId,
}

/// A proposal refused before it ran. Journalled as `Record::ProposalRefused`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefusedDraw {
    pub slot: usize,
    /// Empty when the refusal happened before the configuration could be
    /// stamped — an unreachable payoff floor is refused with a hash, a
    /// lane/horizon conflict without one.
    pub config_hash: String,
    pub reason: ProposalRefused,
}

/// One sweep's worth of proposals, and everything the draw threw away.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawnSweep {
    /// Exactly [`SWEEP_SEARCHES`] proposals, stamped, in draw order — unless
    /// `exhausted`.
    pub proposals: Vec<Proposal>,
    pub duplicates: Vec<DuplicateDraw>,
    pub refused: Vec<RefusedDraw>,
    /// Factors reverted toward the parent by the trust region during this
    /// sweep's draw. Surfaced beside the proposals because a sweep whose
    /// proposals were mostly reverted explored less than its slot count says.
    pub trust_region_reverted: usize,
    /// U5: `PROPOSER_RETRIES` consecutive uniform draws all collided. The
    /// reachable space is enumerated — a result about a searched space, not a
    /// failure.
    pub exhausted: bool,
    /// Slots the sweep never drew, because U5 fired part-way through.
    ///
    /// **Counted here AND emitted into `refused`**, one record per slot, so the
    /// journal carries them. Before this existed, `break 'slots` returned a
    /// short sweep and the only trace of the abandoned slots was the difference
    /// between [`SWEEP_SEARCHES`] and `proposals.len()` — a silent drop of up to
    /// 99 slots inside the one code path that the surrounding `bail!` says it
    /// refuses to take: *"a sweep of fewer than 100 searches is not the
    /// experiment the judge is told it is."*
    pub slots_abandoned: usize,
    pub census: ProposerCensus,
}

/// A refusal raised part way through a draw, carrying the partial configuration
/// it was raised on.
///
/// Requirement (b) is *counted **and named***. A refusal that never reached a
/// whole [`Proposal`] has no `describe()` to quote, and the census used to
/// record the literal string `"<draw>"` for it: eleven deaths, no names, and no
/// way to tell which levels killed them. `drawn_so_far` is every decision the
/// sampler had made when it hit the wall, in decision order.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawRefusal {
    pub reason: ProposalRefused,
    pub drawn_so_far: String,
}

impl DrawRefusal {
    fn new(reason: ProposalRefused, trace: &[String]) -> Self {
        Self { reason, drawn_so_far: trace.join(" ") }
    }
}

/// The baseline a posterior draw is a delta from.
#[derive(Debug, Clone, PartialEq)]
pub struct ParentReference {
    pub sweep: SweepId,
    pub axis_a: SearchConfigDelta,
}

/// Whether a refusal level can be searched at all.
///
/// The only dimension that can be structurally unreachable is the payoff floor:
/// `assert_payoff_floor_reachable` refuses a run whose floor sits above what its
/// own resolved exit geometry can produce, and under the shipped trail that
/// ceiling is the **measured 1.08** — so the 2.0 level would be refused at S3 on
/// every single draw. Pre-excluding it here (and naming it in the space report)
/// spends the sweep's 100 slots on searches that can run, instead of burning a
/// third of them on a refusal that was decided before the draw.
///
/// `None` = no ceiling has been measured yet, so nothing is pre-excluded and an
/// unreachable floor is refused and counted at S3 instead. Never silently
/// lowered: a clamped floor is a configuration nobody chose.
pub fn refusal_level_reachable(dim: RefusalDim, level: u8, payoff_ceiling: Option<f64>) -> bool {
    match (dim, payoff_ceiling) {
        (RefusalDim::PayoffFloor, Some(ceiling)) => {
            let v = dim.values()[level as usize];
            v <= 0.0 || v <= ceiling
        }
        _ => true,
    }
}

/// The proposer. Owns proposal and validation, and nothing that a crash could
/// lose.
#[derive(Debug)]
pub struct Proposer {
    caps: Capabilities,
    scenario: ScenarioKind,
    session_seed: u64,
    base_config: DiscoveryConfig,
    pip_value_per_lot: f64,
    normalize_features: bool,
    drawable_variants: Vec<ObjectiveVariant>,
    /// Replicates drawn per cell — rebuilt from the journal on resume.
    replicates: HashMap<CellKey, u8>,
    /// Semantic identities already drawn, and the sweep each was first drawn
    /// in — rebuilt from the journal on resume.
    keys: HashMap<ProposalKey, SweepId>,
    /// The B0 control's one draw, ever.
    b0_drawn: usize,
    /// The enforced payoff ceiling this configuration's own geometry produces,
    /// measured once the ATR scale is installed.
    payoff_ceiling: Option<f64>,
    census: ProposerCensus,
}

impl Proposer {
    /// Rebuild the proposer from the journal.
    ///
    /// The scenario is read from `DiscoveryConfig::mode`, which mirrors
    /// `system.trading_mode` — the goal constant. `DiscoveryMode::Strict` is
    /// refused rather than mapped: Strict is not one of the operator's two
    /// goals, and quietly treating it as one would point the loop at a target
    /// nobody set.
    pub fn restore(
        session: &Session,
        session_seed: u64,
        base_config: &DiscoveryConfig,
    ) -> Result<Self> {
        let scenario = match base_config.mode {
            DiscoveryMode::Risky => ScenarioKind::Risky,
            DiscoveryMode::PropFirm => ScenarioKind::PropFirm,
            DiscoveryMode::Strict => bail!(
                "DiscoveryConfig::mode is Strict, which is not one of the operator's goal \
                 scenarios (risky / prop_firm). The proposer will not guess which goal a Strict \
                 search is optimising toward — set `system.trading_mode` and re-run."
            ),
        };
        let caps = Capabilities::compiled();
        // THE AXIS-B LIVENESS GATE.
        //
        // `drawable.is_empty()` was the old test and it was too weak by exactly
        // the amount that mattered: a set containing only `ModeNative` variants
        // is not empty, and it is also not an axis — every sweep would run the
        // mode's own objective while the coverage table recorded an objective
        // variant per proposal. `axis_b_live_check` requires at least one
        // uncapped variant that OVERRIDES something, and names every missing
        // symbol when there is none. The operator asked for BOTH axes; this is
        // where "both" stops being a comment.
        let drawable = crate::objective::axis_b_live_check(&caps, scenario)
            .map_err(|inert| anyhow::anyhow!("{inert}"))?;

        let pip_value_per_lot = base_config.evaluation_config(None).pip_value_per_lot;
        let mut me = Self {
            caps,
            scenario,
            session_seed,
            base_config: base_config.clone(),
            pip_value_per_lot,
            normalize_features: neoethos_data::current_data_runtime_overrides().normalize_features,
            drawable_variants: drawable,
            replicates: HashMap::new(),
            keys: HashMap::new(),
            b0_drawn: 0,
            payoff_ceiling: None,
            census: ProposerCensus::default(),
        };

        // Replay the identity side. The posterior and the coverage counters are
        // the session's own folds and are NOT copied here — one fold, one
        // owner.
        let sweeps: Vec<SweepId> = session
            .sweeps
            .iter()
            .map(|s| s.sweep)
            .chain(session.open_sweep.iter().map(|s| s.sweep))
            .collect();
        for sweep in sweeps {
            if let Some(proposals) = session.proposals_of(sweep) {
                for p in proposals {
                    me.remember(p, sweep);
                }
            }
        }
        Ok(me)
    }

    fn remember(&mut self, p: &Proposal, sweep: SweepId) {
        *self.replicates.entry(CellKey::of(p)).or_insert(0) += 1;
        self.keys.entry(p.key()).or_insert(sweep);
        if p.variant() == ObjectiveVariant::B0ScoringTable {
            self.b0_drawn += 1;
        }
    }

    pub fn scenario(&self) -> ScenarioKind {
        self.scenario
    }

    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }

    pub fn census(&self) -> &ProposerCensus {
        &self.census
    }

    pub fn drawable_variants(&self) -> &[ObjectiveVariant] {
        &self.drawable_variants
    }

    pub fn payoff_ceiling(&self) -> Option<f64> {
        self.payoff_ceiling
    }

    /// How a proposal's identity is computed, for the session header.
    pub fn identity_source(&self) -> &'static str {
        identity_source(&self.caps)
    }

    /// The space, as drawn and as excluded, for the session header and every
    /// verdict. *"Unreachable"* is only ever a claim about a searched space, so
    /// the report names the parts that were never searchable at all.
    pub fn space_report(&self) -> SpaceReport {
        let (factors, mut excluded, mut cells) = axis_a_report(&self.caps);
        for v in ObjectiveVariant::ALL {
            match v.exclusion_reason(&self.caps, self.scenario) {
                Some(crate::objective::VariantExclusion::MissingKnob(knob)) => {
                    excluded.push(crate::space::ExcludedLevel::missing_knob(
                        "B",
                        "objective",
                        v.label().to_string(),
                        knob,
                    ));
                }
                Some(crate::objective::VariantExclusion::WrongScenario { requires, session }) => {
                    excluded.push(crate::space::ExcludedLevel::unreachable(
                        "B",
                        "objective",
                        v.label().to_string(),
                        format!(
                            "serves the {requires:?} scenario; this session's goal set is \
                             {session:?}"
                        ),
                    ));
                }
                None => {}
            }
        }
        cells = cells.saturating_mul(self.drawable_variants.len().max(1) as u128);
        for d in RefusalDim::ALL {
            let mut reachable = 0usize;
            for l in 0..d.arity() as u8 {
                if refusal_level_reachable(d, l, self.payoff_ceiling) {
                    reachable += 1;
                } else {
                    excluded.push(crate::space::ExcludedLevel::unreachable(
                        "B",
                        d.label(),
                        d.level_label(l),
                        format!(
                            "a payoff floor of {} sits above the enforced ceiling {:.2} that this \
                             configuration's own exit geometry can produce, so \
                             assert_payoff_floor_reachable would refuse every draw carrying it \
                             before a bar was read",
                            d.level_label(l),
                            self.payoff_ceiling.unwrap_or(f64::NAN)
                        ),
                    ));
                }
            }
            cells = cells.saturating_mul(reachable.max(1) as u128);
        }
        SpaceReport {
            factors,
            excluded,
            identity_source: identity_source(&self.caps).to_string(),
            drawable_cells: cells,
        }
    }

    /// U4 — *a space is not refuted if it was not searched* — evaluated over the
    /// DRAWABLE space only, against the session's own coverage fold.
    ///
    /// The axis-B half goes through [`crate::objective::axis_b_coverage`], the
    /// same function `verdict::check_unreachable` uses, so the proposer's view
    /// of "is the space searched" and the judge's cannot drift. Reading the
    /// coverage counters directly is what let an axis that produced NO runs pass
    /// this check: an axis with no credited keys has no under-drawn key.
    pub fn u4_satisfied(&self, session: &Session) -> bool {
        let axis_b = crate::objective::axis_b_coverage(&self.caps, B_MIN_DRAWS, &|label| {
            session.coverage.get("objective", label).sweeps
        });
        if !axis_b.satisfied() {
            return false;
        }
        for f in FactorId::ALL {
            for l in 0..f.arity() as u8 {
                if !f.level_available(l, &self.caps) {
                    continue;
                }
                if session.coverage.get(f.label(), &f.level_label(l)).sweeps < 1 {
                    return false;
                }
            }
        }
        for d in RefusalDim::ALL {
            for l in 0..d.arity() as u8 {
                if !refusal_level_reachable(d, l, self.payoff_ceiling) {
                    continue;
                }
                if session.coverage.get(d.label(), &d.level_label(l)).sweeps < 1 {
                    return false;
                }
            }
        }
        true
    }

    // ── the reward channel ─────────────────────────────────────────────────

    /// Turn one sweep's screen outcomes into posterior credits.
    ///
    /// * If **any** outcome is `Unavailable` the whole sweep credits nothing:
    ///   while the shuffle null is not yet established every reward is 0 and the
    ///   posterior is not touched at all.
    /// * A level that appears on 40 of the sweep's proposals and passes on 12 of
    ///   them gets **one** success, not twelve — otherwise a single lucky region
    ///   floods the posterior.
    /// * A level that was drawn and never passed gets exactly one failure, so an
    ///   arm's denominator grows at the same rate as its numerator.
    pub fn credits_for<S: ScreenSignal>(
        &self,
        proposals: &[Proposal],
        outcomes: &[S],
    ) -> Vec<PosteriorCredit> {
        if outcomes.iter().any(|o| o.null_unavailable()) {
            tracing::info!(
                target: "neoethos_autoresearch::proposer",
                proposals = proposals.len(),
                "screen conjunct S6 unavailable (the shuffle null has fewer than MIN_NULL_OBS \
                 observations): every reward is 0 and the posterior is NOT updated"
            );
            return Vec::new();
        }
        let mut drawn: HashMap<(&'static str, String), bool> = HashMap::new();
        for (p, o) in proposals.iter().zip(outcomes.iter()) {
            let passed = o.passed();
            for (factor, level) in p.coverage_levels() {
                let e = drawn.entry((factor, level)).or_insert(false);
                *e = *e || passed;
            }
        }
        let mut credits: Vec<PosteriorCredit> = drawn
            .into_iter()
            .map(|((factor, level), success)| PosteriorCredit {
                factor: factor.to_string(),
                level,
                success,
            })
            .collect();
        credits.sort_by(|a, b| (&a.factor, &a.level).cmp(&(&b.factor, &b.level)));
        credits
    }

    // ── the draw ───────────────────────────────────────────────────────────

    /// Draw one sweep: all [`SWEEP_SEARCHES`] proposals, stamped, **before any
    /// of them runs** — so a sweep is one experiment with one `sweep_hash` and
    /// is reproducible from `(session_seed, sweep_index)` plus the journal.
    ///
    /// S1 PROPOSE, S2 GUARD and S3 STAMP all happen here, in that order:
    /// sample → trust region → constraint repair → dedupe → materialise →
    /// money-path audit → payoff-ceiling gate → stamp.
    pub fn draw_sweep(&mut self, sweep: SweepId, session: &Session) -> Result<DrawnSweep> {
        // The payoff-ceiling inputs are the run's own resolved SL/TP band,
        // trailing geometry and round-trip cost. They read the ATR scale the
        // runner installed; without one the band is the absolute pip literal and
        // the gate is measuring a different search than the one about to run.
        let inputs = payoff_inputs_for_config(&self.base_config, self.pip_value_per_lot);
        if inputs.atr_pips.is_none() {
            tracing::warn!(
                target: "neoethos_autoresearch::proposer",
                sl_min_pips = inputs.sl_min_pips,
                tp_max_pips = inputs.tp_max_pips,
                "no ATR scale is installed, so the payoff ceiling is being computed against the \
                 ABSOLUTE pip band. Install the per-run scale before drawing, or the S3 gate \
                 judges a band this dataset will not search."
            );
        }
        self.payoff_ceiling = neoethos_search::run_identity::max_achievable_payoff(&inputs)
            .ok()
            .map(|c| c.enforced_ceiling);

        let parent = self.parent_of(session);
        let mut rng = ChaCha20Rng::seed_from_u64(mix(self.session_seed, sweep.0));
        let mut out = DrawnSweep {
            proposals: Vec::with_capacity(SWEEP_SEARCHES),
            duplicates: Vec::new(),
            refused: Vec::new(),
            trust_region_reverted: 0,
            exhausted: false,
            slots_abandoned: 0,
            census: ProposerCensus::default(),
        };
        let mut debt = self.coverage_debt(session);
        debt.reverse(); // pop() takes the most-owed first
        let mut consecutive_uniform_collisions = 0usize;

        'slots: for slot in 0..SWEEP_SEARCHES {
            // The exploration floor is exactly EXPLORATION_FLOOR uniform draws
            // and is never spent on anything else. Coverage debt is paid from
            // the posterior slots: a floor that can be borrowed against is not
            // a floor.
            let uniform = slot < EXPLORATION_FLOOR;
            let mut attempt = 0usize;

            loop {
                let force_uniform = uniform || attempt >= PROPOSER_RETRIES;
                let forced_variant = if force_uniform || attempt > 0 {
                    None
                } else {
                    debt.last().copied()
                };

                let candidate = match self.draw_one(
                    session,
                    &mut rng,
                    slot,
                    force_uniform,
                    forced_variant,
                    parent.as_ref(),
                    &mut out.census,
                ) {
                    Ok(c) => c,
                    Err(DrawRefusal { reason, drawn_so_far }) => {
                        // `drawn_so_far` and not `"<draw>"`. The census line for
                        // a refusal that never reached a whole proposal used to
                        // be the literal string `<draw>`, which counted the
                        // abandonment and named nothing: a reader could see that
                        // eleven draws died in the space and could not tell which
                        // levels killed them. The partial draw is the name.
                        out.census.record_refusal(&reason, &drawn_so_far);
                        out.refused.push(RefusedDraw {
                            slot,
                            config_hash: String::new(),
                            reason,
                        });
                        attempt += 1;
                        if attempt >= PROPOSER_RETRIES {
                            bail!(
                                "the proposer could not produce an expressible proposal for slot \
                                 {slot} in {PROPOSER_RETRIES} attempts. Refusing to shorten the \
                                 sweep silently — a sweep of fewer than {SWEEP_SEARCHES} searches \
                                 is not the experiment the judge is told it is."
                            );
                        }
                        continue;
                    }
                };

                // ── S3 STAMP ───────────────────────────────────────────────
                //
                // Stamped BEFORE the identity guard so that a duplicate is
                // journalled with the `config_hash` it would have run under. A
                // `ProposalDuplicate` record with an empty hash is a refusal
                // nobody can trace, which is the silent drop one level up.
                let cell = CellKey::of(&candidate);
                let replicate = self.replicates.get(&cell).copied().unwrap_or(0);
                let mut candidate = candidate;
                candidate.replicate_seed = replicate_seed(self.session_seed, &cell, replicate);
                match self.stamp(&candidate, &inputs) {
                    Ok(config_hash) => candidate.config_hash = config_hash,
                    Err(reason) => {
                        out.census.record_refusal(&reason, &candidate.describe());
                        out.refused.push(RefusedDraw {
                            slot,
                            config_hash: String::new(),
                            reason,
                        });
                        attempt += 1;
                        continue;
                    }
                }

                // ── S2 GUARD: identity ─────────────────────────────────────
                let first_seen = self.keys.get(&candidate.key()).copied();
                if first_seen.is_some() || replicate >= MAX_REPLICATES_PER_CELL {
                    let reason = match first_seen {
                        Some(first_seen) => ProposalRefused::Duplicate { first_seen },
                        None => ProposalRefused::CellReplicatesExhausted { replicates: replicate },
                    };
                    record_identity_refusal(
                        &mut out,
                        slot,
                        &candidate.config_hash,
                        &candidate.describe(),
                        reason,
                        first_seen,
                    );
                    if force_uniform {
                        out.census.uniform_collisions += 1;
                        consecutive_uniform_collisions += 1;
                        if consecutive_uniform_collisions >= PROPOSER_RETRIES {
                            abandon_remaining_slots(
                                &mut out,
                                slot,
                                consecutive_uniform_collisions,
                            );
                            tracing::warn!(
                                target: "neoethos_autoresearch::proposer",
                                %sweep,
                                drawn = out.proposals.len(),
                                of = SWEEP_SEARCHES,
                                slots_abandoned = out.slots_abandoned,
                                "U5 — the reachable space is enumerated. The sweep stops short; \
                                 every slot it never drew is journalled as a ProposalRefused so \
                                 nothing leaves without a record."
                            );
                            break 'slots;
                        }
                    }
                    attempt += 1;
                    continue;
                }

                // A stamp collision means two DIFFERENT proposals share a
                // config_hash: the stamp cannot tell them apart. The proposal
                // still runs — the semantic key is the authoritative identity —
                // but the gap is counted and named rather than absorbed.
                if let Some(first) = session.config_hashes.get(&candidate.config_hash) {
                    out.census.stamp_collisions += 1;
                    tracing::warn!(
                        target: "neoethos_autoresearch::proposer",
                        config_hash = %candidate.config_hash,
                        first_seen = %first,
                        proposal = %candidate.describe(),
                        "STAMP GAP: this proposal differs from an earlier one in a factor \
                         ResolvedConfigStamp does not carry, so the two share a config_hash. The \
                         run is a different experiment; its artifacts are not attributable by hash \
                         alone. docs/autoresearch-loop.md §14.7 lists the fields to add."
                    );
                }

                consecutive_uniform_collisions = 0;
                if matches!(candidate.origin, ProposalOrigin::CoverageDebt) {
                    debt.pop();
                    out.census.coverage_debt_forced += 1;
                }
                self.remember(&candidate, sweep);
                out.census.proposals_drawn += 1;
                out.proposals.push(candidate);
                break;
            }
        }

        out.trust_region_reverted = out.census.trust_region_reverted;
        self.census.absorb(&out.census);
        Ok(out)
    }

    /// Re-read the session after a `PosteriorRetracted`.
    ///
    /// It re-syncs the identity and replicate maps only. **The posterior is not
    /// copied here and never was**: [`Proposer::draw_sweep`] reads
    /// `session.posterior` at draw time, so a retraction is already visible to
    /// the next draw without any refresh. Stated rather than left as an empty
    /// body, because a method that silently does nothing is indistinguishable
    /// from one that forgot to.
    pub fn reload(&mut self, session: &Session) {
        self.replicates.clear();
        self.keys.clear();
        self.b0_drawn = 0;
        let sweeps: Vec<SweepId> = session
            .sweeps
            .iter()
            .map(|s| s.sweep)
            .chain(session.open_sweep.iter().map(|s| s.sweep))
            .collect();
        for sweep in sweeps {
            if let Some(proposals) = session.proposals_of(sweep) {
                for p in proposals {
                    self.remember(p, sweep);
                }
            }
        }
    }

    /// The session's best-ever screened sweep, as a delta baseline.
    fn parent_of(&self, session: &Session) -> Option<ParentReference> {
        let best = session.best_ever.as_ref()?;
        let proposals = session.proposals_of(best.sweep)?;
        let p = proposals.get(best.slot)?;
        Some(ParentReference { sweep: best.sweep, axis_a: p.axis_a })
    }

    /// Axis-B variants still owing coverage, most-owed first.
    fn coverage_debt(&self, session: &Session) -> Vec<ObjectiveVariant> {
        let mut owed: Vec<_> = self
            .drawable_variants
            .iter()
            .copied()
            .filter(|v| {
                v.spec().max_draws_per_session.is_none()
                    && session.coverage.get("objective", v.label()).sweeps < B_MIN_DRAWS
            })
            .collect();
        owed.sort_by_key(|v| (session.coverage.get("objective", v.label()).sweeps, v.index()));
        owed
    }

    /// S3: materialise, audit, gate on the payoff ceiling, and stamp.
    fn stamp(
        &self,
        proposal: &Proposal,
        inputs: &PayoffCeilingInputs,
    ) -> Result<String, ProposalRefused> {
        let spec: ProposedRunSpec = materialise(&self.base_config, proposal, inputs)?;
        let stamp = neoethos_search::run_identity::stamp_resolved_config(
            &spec.config,
            inputs,
            spec.payoff_ceiling,
            self.pip_value_per_lot,
            self.normalize_features,
        )
        .map_err(|e| ProposalRefused::AppliedValueMismatch {
            field: "config_hash".to_string(),
            expected: "a resolved stamp".to_string(),
            actual: e.to_string(),
        })?;
        Ok(stamp.config_hash)
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_one(
        &self,
        session: &Session,
        rng: &mut ChaCha20Rng,
        slot: usize,
        force_uniform: bool,
        forced_variant: Option<ObjectiveVariant>,
        parent: Option<&ParentReference>,
        census: &mut ProposerCensus,
    ) -> Result<Proposal, DrawRefusal> {
        let evidence = |factor: &str, level: &str| beta_params(session, factor, level);

        // Everything decided so far, in decision order. A refusal raised part
        // way through a draw is attached to it, so the census can name the
        // configuration that was abandoned instead of printing `<draw>`.
        let mut trace: Vec<String> = Vec::with_capacity(FACTOR_COUNT + 8);
        trace.push(format!("slot={slot}"));
        trace.push(format!("origin={}", if force_uniform { "uniform" } else { "posterior" }));
        macro_rules! named {
            ($trace:expr, $e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(reason) => return Err(DrawRefusal::new(reason, &$trace)),
                }
            };
        }

        // ── axis A ─────────────────────────────────────────────────────────
        let mut levels = [0u8; FACTOR_COUNT];
        for f in FactorId::ALL {
            let available: Vec<u8> =
                (0..f.arity() as u8).filter(|l| f.level_available(*l, &self.caps)).collect();
            levels[f.index()] = if force_uniform {
                available[rng.random_range(0..available.len())]
            } else {
                argmax_sampled(rng, &available, |l| evidence(f.label(), &f.level_label(l)))
            };
            trace.push(format!("{}={}", f.label(), f.level_label(levels[f.index()])));
        }
        let mut axis_a = named!(
            trace,
            SearchConfigDelta::new(levels, &self.caps).map_err(ProposalRefused::Space)
        );

        // ── trust region ───────────────────────────────────────────────────
        //
        // Axis A only. The objective variant, its parameter and the refusal
        // vector are deliberately OUTSIDE it: they are the half the operator
        // asked to be explored, their arity is small enough to enumerate, and
        // coverage debt already forces that enumeration. A trust region over
        // them would skip the skipped half all over again.
        if !force_uniform
            && let Some(parent) = parent
        {
            axis_a = named!(
                trace,
                apply_trust_region(axis_a, &parent.axis_a, &self.caps, &evidence, census)
            );
            for f in FactorId::ALL {
                trace.push(format!("{}'={}", f.label(), f.level_label(axis_a.level(f))));
            }
        }

        // ── axis B ─────────────────────────────────────────────────────────
        let variant = match forced_variant {
            Some(v) => v,
            None => {
                let pool: Vec<ObjectiveVariant> = self
                    .drawable_variants
                    .iter()
                    .copied()
                    .filter(|v| match v.spec().max_draws_per_session {
                        Some(cap) => self.b0_drawn < cap,
                        None => true,
                    })
                    .collect();
                // NO FALLBACK. This used to read
                //   `if pool.is_empty() { self.drawable_variants.clone() }`
                // which, when every drawable variant had spent its
                // `max_draws_per_session`, silently re-admitted the capped
                // control and drew it again — and again — past its declared
                // ceiling, with no counter and no census line. A cap that the
                // sampler quietly steps over is a configuration nobody chose
                // running under a name somebody did, which is the exact sin the
                // module header forbids.
                //
                // `axis_b_live_check` guarantees at least one UNCAPPED
                // overriding variant is drawable, so this branch is unreachable
                // in a session that started; if it is ever reached the space
                // declaration and the liveness gate have come apart, and the
                // honest answer is a named refusal that the slot loop counts.
                if pool.is_empty() {
                    return Err(DrawRefusal::new(
                        ProposalRefused::Objective(ObjectiveError::AllVariantsCapped {
                            drawable: self
                                .drawable_variants
                                .iter()
                                .map(|v| v.label().to_string())
                                .collect(),
                        }),
                        &trace,
                    ));
                }
                if force_uniform {
                    pool[rng.random_range(0..pool.len())]
                } else {
                    let idx: Vec<u8> = (0..pool.len() as u8).collect();
                    let picked = argmax_sampled(rng, &idx, |i| {
                        evidence("objective", pool[i as usize].label())
                    });
                    pool[picked as usize]
                }
            }
        };
        let mut param = rng.random_range(0..variant.param_count()) as u8;
        trace.push(format!("objective={}[{}]", variant.label(), variant.param_label(param)));

        // ── the lane / horizon pairing rule ────────────────────────────────
        //
        // An H4 lane is only ever PROPOSED with a horizon that can express it.
        // Repairing here — and counting it — rather than emitting a proposal
        // that S3 will refuse keeps the sweep at 100 real searches.
        let horizon_of = |p: u8| -> Result<usize, ProposalRefused> {
            Ok(ObjectiveChoice::new(
                variant,
                p,
                RefusalLevels::permissive(),
                &self.caps,
                self.scenario,
            )
            .map_err(ProposalRefused::Objective)?
            .label_horizon_bars(0))
        };
        let mut horizon = named!(trace, horizon_of(param));
        if lane_horizon_conflict(axis_a.base_timeframe(), axis_a.higher_lanes(), horizon).is_err() {
            // The pairing that was DRAWN and is about to be replaced. Counted
            // AND named: `constraint_resampled` alone said that N draws were
            // repaired and left no way to see which ones, so the configuration
            // the sampler actually chose vanished. (b) asks for both.
            let abandoned = format!(
                "{} + lanes[{}] with objective {}[{}] (horizon {horizon} base bars)",
                axis_a.base_timeframe(),
                axis_a.higher_lanes().join("+"),
                variant.label(),
                variant.param_label(param),
            );
            census.constraint_resampled += 1;
            let mut fixed = false;
            if variant == ObjectiveVariant::B4LabelHorizon {
                for p in (0..variant.param_count() as u8).rev() {
                    let h = named!(trace, horizon_of(p));
                    if lane_horizon_conflict(axis_a.base_timeframe(), axis_a.higher_lanes(), h)
                        .is_ok()
                    {
                        param = p;
                        horizon = h;
                        fixed = true;
                        break;
                    }
                }
            }
            let mut repair = if fixed {
                format!("horizon raised to {horizon} base bars")
            } else {
                "unrepaired".to_string()
            };
            if !fixed {
                for l in (0..FactorId::HigherLanes.arity() as u8).rev() {
                    let candidate = named!(
                        trace,
                        axis_a
                            .with_level(FactorId::HigherLanes, l, &self.caps)
                            .map_err(ProposalRefused::Space)
                    );
                    if lane_horizon_conflict(
                        candidate.base_timeframe(),
                        candidate.higher_lanes(),
                        horizon,
                    )
                    .is_ok()
                    {
                        axis_a = candidate;
                        repair = format!(
                            "higher_lanes reduced to [{}]",
                            axis_a.higher_lanes().join("+")
                        );
                        break;
                    }
                }
            }
            note_example(
                census,
                format!("constraint_resampled: {repair}"),
                abandoned,
            );
        }

        // ── the refusal vector ─────────────────────────────────────────────
        let mut refusal_levels = [0u8; 4];
        for d in RefusalDim::ALL {
            let available: Vec<u8> = (0..d.arity() as u8)
                .filter(|l| refusal_level_reachable(d, *l, self.payoff_ceiling))
                .collect();
            refusal_levels[d.index()] = if force_uniform {
                available[rng.random_range(0..available.len())]
            } else {
                argmax_sampled(rng, &available, |l| evidence(d.label(), &d.level_label(l)))
            };
        }
        for d in RefusalDim::ALL {
            trace.push(format!("{}={}", d.label(), d.level_label(refusal_levels[d.index()])));
        }
        let refusals = named!(
            trace,
            RefusalLevels::new(refusal_levels).map_err(ProposalRefused::Objective)
        );

        let origin = if force_uniform {
            ProposalOrigin::ExplorationFloor
        } else if forced_variant.is_some() {
            ProposalOrigin::CoverageDebt
        } else {
            ProposalOrigin::Posterior
        };

        Ok(named!(
            trace,
            crate::proposal::build(
                axis_a,
                variant,
                param,
                refusals,
                // Replaced in `draw_sweep` by the cell's own replicate seed.
                0,
                origin,
                slot,
                if force_uniform { None } else { parent.map(|p| p.sweep) },
                &self.caps,
                self.scenario,
            )
        ))
    }
}

/// `Beta(1 + successes, 1 + failures)` for one arm.
///
/// **The prior is the proposer's; the evidence is the session's.** The session
/// folds `PosteriorUpdated` and `PosteriorRetracted` and stores counts only, so
/// there is nowhere for a prior to be quietly baked into the history.
fn beta_params(session: &Session, factor: &str, level: &str) -> (f64, f64) {
    let (s, f) = session.posterior.evidence(factor, level);
    (1.0 + s as f64, 1.0 + f as f64)
}

/// Revert factors toward the parent until at most [`MAX_DELTA_FACTORS`] differ,
/// choosing what to revert by the **smallest posterior gap** — the factors the
/// evidence is least sure about go first, because reverting a factor the
/// posterior has an opinion about would throw away the opinion.
pub fn apply_trust_region<E>(
    mut axis_a: SearchConfigDelta,
    parent: &SearchConfigDelta,
    caps: &Capabilities,
    evidence: &E,
    census: &mut ProposerCensus,
) -> Result<SearchConfigDelta, ProposalRefused>
where
    E: Fn(&str, &str) -> (f64, f64),
{
    let mut differing = axis_a.differing_factors(parent);
    if differing.len() <= MAX_DELTA_FACTORS {
        return Ok(axis_a);
    }
    let mean = |f: FactorId, l: u8| {
        let (a, b) = evidence(f.label(), &f.level_label(l));
        a / (a + b)
    };
    differing.sort_by(|a, b| {
        let ga = (mean(*a, axis_a.level(*a)) - mean(*a, parent.level(*a))).abs();
        let gb = (mean(*b, axis_a.level(*b)) - mean(*b, parent.level(*b))).abs();
        ga.partial_cmp(&gb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let revert = differing.len() - MAX_DELTA_FACTORS;
    for f in differing.into_iter().take(revert) {
        // Counted AND named. The drawn level is a configuration the sampler
        // chose and the trust region then discarded; `trust_region_reverted`
        // alone recorded that N of them died and left no way to see which
        // factors kept losing to the parent — which is precisely the question a
        // reader of the census asks next.
        let drawn = f.level_label(axis_a.level(f));
        let reverted_to = f.level_label(parent.level(f));
        axis_a = axis_a.with_level(f, parent.level(f), caps).map_err(ProposalRefused::Space)?;
        census.trust_region_reverted += 1;
        note_example(
            census,
            format!("trust_region_reverted: {} -> {reverted_to}", f.label()),
            format!("{}={drawn} (drawn), outside the {MAX_DELTA_FACTORS}-factor trust region", f.label()),
        );
    }
    Ok(axis_a)
}

/// Thompson step: one Beta draw per candidate level, take the argmax. Ties go to
/// the lower index, deterministically — a random tiebreak would make the same
/// `(session_seed, sweep)` produce two different sweeps.
fn argmax_sampled<F>(rng: &mut ChaCha20Rng, levels: &[u8], params: F) -> u8
where
    F: Fn(u8) -> (f64, f64),
{
    let mut best = levels[0];
    let mut best_theta = f64::NEG_INFINITY;
    for l in levels {
        let (a, b) = params(*l);
        let theta = sample_beta(rng, a, b);
        if theta > best_theta {
            best_theta = theta;
            best = *l;
        }
    }
    best
}

// ── sampling ───────────────────────────────────────────────────────────────
//
// Beta from two Gammas (Marsaglia–Tsang) on the session's own ChaCha stream.
// Written here rather than pulled from `rand_distr` because the workspace's
// `rand_distr` is pinned to the rand-0.8 line while this crate is on rand 0.9,
// and because a sampler the session's determinism depends on belongs in the
// crate whose tests assert that determinism.

fn open_unit(rng: &mut ChaCha20Rng) -> f64 {
    loop {
        let u: f64 = rng.random();
        if u > 0.0 && u < 1.0 {
            return u;
        }
    }
}

fn standard_normal(rng: &mut ChaCha20Rng) -> f64 {
    let u1 = open_unit(rng);
    let u2 = open_unit(rng);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Marsaglia–Tsang, for `shape >= 1`. Every posterior here is
/// `Beta(1 + successes, 1 + failures)`, so both shapes are at least 1 by
/// construction; the clamp is a guard, not a behaviour.
fn sample_gamma(rng: &mut ChaCha20Rng, shape: f64) -> f64 {
    let shape = shape.max(1.0);
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = open_unit(rng);
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v3;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

fn sample_beta(rng: &mut ChaCha20Rng, a: f64, b: f64) -> f64 {
    let x = sample_gamma(rng, a);
    let y = sample_gamma(rng, b);
    if x + y <= 0.0 { 0.5 } else { x / (x + y) }
}

/// splitmix64 — one stream per `(session_seed, sweep)`, so a sweep's 100 draws
/// are reproducible from those two numbers and the journal.
fn mix(seed: u64, sweep: u64) -> u64 {
    let mut z = seed ^ sweep.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The GA seed for the `n`-th replicate of a cell. Deterministic, so a resumed
/// session redraws the same replicate rather than starting a fresh one.
pub fn replicate_seed(session_seed: u64, cell: &CellKey, replicate: u8) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u64| {
        h ^= b;
        h = h.wrapping_mul(0x1000_0000_01b3);
    };
    eat(session_seed);
    for l in cell.axis_a {
        eat(l as u64);
    }
    eat(cell.variant.index() as u64);
    eat(cell.param as u64);
    for l in cell.refusals {
        eat(l as u64);
    }
    eat(replicate as u64);
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::RefusalDim as RD;

    fn evidence_uniform() -> impl Fn(&str, &str) -> (f64, f64) {
        |_: &str, _: &str| (1.0, 1.0)
    }

    #[test]
    fn beta_sampling_is_in_range_and_tracks_its_mean() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let mut sum = 0.0;
        for _ in 0..2000 {
            let x = sample_beta(&mut rng, 8.0, 2.0);
            assert!((0.0..=1.0).contains(&x), "beta sample out of range: {x}");
            sum += x;
        }
        let mean = sum / 2000.0;
        assert!((mean - 0.8).abs() < 0.03, "Beta(8,2) mean was {mean}, expected ~0.8");
    }

    #[test]
    fn thompson_prefers_the_arm_with_the_evidence() {
        // Not a property of one draw — a property of many. An arm with 30
        // successes and 1 failure should win the argmax far more often than one
        // with 1 and 30.
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let mut wins = 0;
        for _ in 0..500 {
            let picked = argmax_sampled(&mut rng, &[0, 1], |l| {
                if l == 0 { (31.0, 2.0) } else { (2.0, 31.0) }
            });
            if picked == 0 {
                wins += 1;
            }
        }
        assert!(wins > 450, "the evidenced arm won only {wins}/500");
    }

    #[test]
    fn the_prior_is_uniform_so_the_loop_starts_with_no_opinion() {
        // Beta(1,1) on every arm: with no evidence the two arms must be within
        // sampling noise of a coin.
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        let mut zeros = 0;
        for _ in 0..1000 {
            if argmax_sampled(&mut rng, &[0, 1], |_| (1.0, 1.0)) == 0 {
                zeros += 1;
            }
        }
        assert!((400..600).contains(&zeros), "uniform prior favoured a level: {zeros}/1000");
    }

    #[test]
    fn the_trust_region_bounds_the_delta_and_counts_every_revert() {
        let caps = Capabilities::all_for_tests();
        let parent = SearchConfigDelta::new([0; FACTOR_COUNT], &caps).unwrap();
        let mut far = parent;
        for f in FactorId::ALL {
            far = far.with_level(f, 1, &caps).unwrap();
        }
        assert_eq!(far.differing_factors(&parent).len(), FACTOR_COUNT);

        let mut census = ProposerCensus::default();
        let bounded =
            apply_trust_region(far, &parent, &caps, &evidence_uniform(), &mut census).unwrap();
        assert_eq!(bounded.differing_factors(&parent).len(), MAX_DELTA_FACTORS);
        assert_eq!(census.trust_region_reverted, FACTOR_COUNT - MAX_DELTA_FACTORS);
    }

    #[test]
    fn the_trust_region_reverts_the_factors_the_posterior_is_least_sure_about() {
        let caps = Capabilities::all_for_tests();
        let parent = SearchConfigDelta::new([0; FACTOR_COUNT], &caps).unwrap();
        let far = parent
            .with_level(FactorId::Population, 1, &caps)
            .unwrap()
            .with_level(FactorId::Generations, 1, &caps)
            .unwrap()
            .with_level(FactorId::GeneWidth, 1, &caps)
            .unwrap()
            .with_level(FactorId::CandidateCount, 1, &caps)
            .unwrap();
        // `population` level 1 has a strong record; everything else is untried,
        // so the gap there is 0 and it is reverted first.
        let evidence = |factor: &str, level: &str| {
            if factor == "population" && level == "2048" { (30.0, 1.0) } else { (1.0, 1.0) }
        };
        let mut census = ProposerCensus::default();
        let bounded = apply_trust_region(far, &parent, &caps, &evidence, &mut census).unwrap();
        assert_eq!(census.trust_region_reverted, 1);
        assert_eq!(
            bounded.level(FactorId::Population),
            1,
            "the factor the posterior has an opinion about must survive the revert"
        );
    }

    #[test]
    fn an_unavailable_screen_credits_nothing_at_all() {
        struct Sig(bool, bool);
        impl ScreenSignal for Sig {
            fn passed(&self) -> bool {
                self.0
            }
            fn null_unavailable(&self) -> bool {
                self.1
            }
        }
        let caps = Capabilities::all_for_tests();
        let p = crate::proposal::build(
            SearchConfigDelta::new([0; FACTOR_COUNT], &caps).unwrap(),
            ObjectiveVariant::B5TerminalWealth,
            0,
            RefusalLevels::permissive(),
            1,
            ProposalOrigin::ExplorationFloor,
            0,
            None,
            &caps,
            ScenarioKind::Risky,
        )
        .unwrap();
        let proposer = test_proposer();
        assert!(
            proposer.credits_for(&[p.clone()], &[Sig(true, true)]).is_empty(),
            "an unavailable screen must not touch the posterior"
        );
        let credits = proposer.credits_for(&[p], &[Sig(true, false)]);
        assert!(!credits.is_empty());
        assert!(credits.iter().all(|c| c.success));
    }

    #[test]
    fn a_level_takes_one_success_per_sweep_however_many_carried_it() {
        struct Pass(bool);
        impl ScreenSignal for Pass {
            fn passed(&self) -> bool {
                self.0
            }
            fn null_unavailable(&self) -> bool {
                false
            }
        }
        let caps = Capabilities::all_for_tests();
        let delta = SearchConfigDelta::new([0; FACTOR_COUNT], &caps).unwrap();
        let mut proposals = Vec::new();
        for slot in 0..10 {
            proposals.push(
                crate::proposal::build(
                    delta,
                    ObjectiveVariant::B5TerminalWealth,
                    0,
                    RefusalLevels::permissive(),
                    slot as u64,
                    ProposalOrigin::ExplorationFloor,
                    slot,
                    None,
                    &caps,
                    ScenarioKind::Risky,
                )
                .unwrap(),
            );
        }
        let outcomes: Vec<Pass> = (0..10).map(|i| Pass(i % 2 == 0)).collect();
        let credits = test_proposer().credits_for(&proposals, &outcomes);
        // Ten proposals, all carrying the same levels, five of them passing:
        // exactly ONE credit per (factor, level), and it is a success.
        let population: Vec<_> = credits.iter().filter(|c| c.factor == "population").collect();
        assert_eq!(population.len(), 1);
        assert!(population[0].success);
    }

    #[test]
    fn a_drawn_level_that_never_passed_takes_exactly_one_failure() {
        struct Fail;
        impl ScreenSignal for Fail {
            fn passed(&self) -> bool {
                false
            }
            fn null_unavailable(&self) -> bool {
                false
            }
        }
        let caps = Capabilities::all_for_tests();
        let p = crate::proposal::build(
            SearchConfigDelta::new([0; FACTOR_COUNT], &caps).unwrap(),
            ObjectiveVariant::B5TerminalWealth,
            0,
            RefusalLevels::permissive(),
            1,
            ProposalOrigin::ExplorationFloor,
            0,
            None,
            &caps,
            ScenarioKind::Risky,
        )
        .unwrap();
        let credits = test_proposer().credits_for(&[p.clone(), p], &[Fail, Fail]);
        assert!(credits.iter().all(|c| !c.success));
        assert_eq!(credits.iter().filter(|c| c.factor == "population").count(), 1);
    }

    #[test]
    fn the_credit_keys_are_the_coverage_keys() {
        // If these two ever drifted, an arm's evidence and its coverage would
        // describe different things and U4 would be counting a space the
        // posterior never saw.
        struct Pass;
        impl ScreenSignal for Pass {
            fn passed(&self) -> bool {
                true
            }
            fn null_unavailable(&self) -> bool {
                false
            }
        }
        let caps = Capabilities::all_for_tests();
        let p = crate::proposal::build(
            SearchConfigDelta::new([0; FACTOR_COUNT], &caps).unwrap(),
            ObjectiveVariant::B5TerminalWealth,
            0,
            RefusalLevels::permissive(),
            1,
            ProposalOrigin::ExplorationFloor,
            0,
            None,
            &caps,
            ScenarioKind::Risky,
        )
        .unwrap();
        let credits = test_proposer().credits_for(&[p.clone()], &[Pass]);
        let coverage: Vec<(String, String)> = p
            .coverage_levels()
            .into_iter()
            .map(|(f, l)| (f.to_string(), l))
            .collect();
        for c in &credits {
            assert!(
                coverage.contains(&(c.factor.clone(), c.level.clone())),
                "credit `{}={}` has no matching coverage key",
                c.factor,
                c.level
            );
        }
        assert_eq!(credits.len(), coverage.len());
    }

    #[test]
    fn an_unreachable_payoff_floor_is_excluded_and_named() {
        assert!(refusal_level_reachable(RD::PayoffFloor, 0, Some(1.08)));
        assert!(refusal_level_reachable(RD::PayoffFloor, 1, Some(1.08)));
        // level 2 is the 2.0 floor; the shipped trail caps the enforced ceiling
        // at the MEASURED 1.08.
        assert!(!refusal_level_reachable(RD::PayoffFloor, 2, Some(1.08)));
        // With no measurement, nothing is pre-excluded — S3 refuses and counts.
        assert!(refusal_level_reachable(RD::PayoffFloor, 2, None));
        // No other dimension is ever pre-excluded by a payoff ceiling.
        assert!(refusal_level_reachable(RD::WinRateFloor, 2, Some(0.0)));
    }

    #[test]
    fn replicates_of_one_cell_get_different_seeds_and_replay_identically() {
        let cell = CellKey {
            axis_a: [0; FACTOR_COUNT],
            variant: ObjectiveVariant::B5TerminalWealth,
            param: 0,
            refusals: [0; 4],
        };
        let a = replicate_seed(7, &cell, 0);
        assert_ne!(a, replicate_seed(7, &cell, 1));
        assert_eq!(a, replicate_seed(7, &cell, 0), "a resumed session must redraw the same seed");
        assert_ne!(a, replicate_seed(8, &cell, 0));
    }

    #[test]
    fn the_sweep_stream_is_a_function_of_seed_and_sweep_only() {
        assert_eq!(mix(42, 3), mix(42, 3));
        assert_ne!(mix(42, 3), mix(42, 4));
        assert_ne!(mix(42, 3), mix(43, 3));
    }

    /// A proposer with no session behind it, for the pure-function tests.
    fn test_proposer() -> Proposer {
        Proposer {
            caps: Capabilities::all_for_tests(),
            scenario: ScenarioKind::Risky,
            session_seed: 1,
            base_config: DiscoveryConfig::default(),
            pip_value_per_lot: 10.0,
            normalize_features: false,
            drawable_variants: vec![ObjectiveVariant::B5TerminalWealth],
            replicates: HashMap::new(),
            keys: HashMap::new(),
            b0_drawn: 0,
            payoff_ceiling: None,
            census: ProposerCensus::default(),
        }
    }

    fn empty_drawn_sweep() -> DrawnSweep {
        DrawnSweep {
            proposals: Vec::new(),
            duplicates: Vec::new(),
            refused: Vec::new(),
            trust_region_reverted: 0,
            exhausted: false,
            slots_abandoned: 0,
            census: ProposerCensus::default(),
        }
    }

    // ── DEFECT 3: the axis-B liveness gate lives in `restore` ───────────────

    #[test]
    fn the_proposer_refuses_a_session_whose_objective_axis_cannot_vary() {
        // `restore` used to bail only when the drawable set was EMPTY. A set of
        // ModeNative variants is not empty and is also not an axis: every sweep
        // would run the mode's own objective while the coverage table recorded
        // an objective variant per proposal, and U4 would then certify a space
        // that was never searched on axis B.
        //
        // The gate is `objective::axis_b_live_check`; this asserts the proposer
        // routes through it and that the shipped build passes it.
        let caps = Capabilities::compiled();
        for scenario in crate::objective::SCENARIOS {
            let drawable = crate::objective::axis_b_live_check(&caps, scenario)
                .expect("the shipped build must present a varying objective axis");
            assert!(crate::objective::carries_the_axis(&drawable));
        }
        // …and the rule the gate applies rejects a ModeNative-only set.
        assert!(!crate::objective::carries_the_axis(&[ObjectiveVariant::B5TerminalWealth]));
    }

    #[test]
    fn u4_is_false_while_the_objective_axis_has_no_runs() {
        // `Proposer::u4_satisfied` and `verdict::check_unreachable` must agree,
        // and neither may certify an axis that produced nothing. A default
        // Session has an empty coverage fold: every objective variant reads 0.
        let mut p = test_proposer();
        p.caps = Capabilities::compiled();
        p.drawable_variants =
            crate::objective::axis_b_live_check(&p.caps, ScenarioKind::Risky).unwrap();
        assert!(
            !p.u4_satisfied(&Session::default()),
            "U4 certified a space whose objective axis produced no runs"
        );
    }

    // ── DEFECT 4: no abandoned configuration may vanish ─────────────────────

    #[test]
    fn a_partial_draw_that_is_refused_names_what_it_had_drawn() {
        // HOLE: the census line for a refusal raised inside `draw_one` was the
        // literal string `"<draw>"`. The abandonment was counted and the
        // configuration was never named, so a reader could see that N draws died
        // in the objective and could not tell which levels killed them.
        //
        // Forced through the one refusal `draw_one` can raise on its own: a
        // drawable set consisting of a single capped control whose cap is spent.
        let mut p = test_proposer();
        p.drawable_variants = vec![ObjectiveVariant::B0ScoringTable];
        p.b0_drawn = 1; // B0's max_draws_per_session is 1, and it is spent.
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let mut census = ProposerCensus::default();
        let err = p
            .draw_one(&Session::default(), &mut rng, 7, true, None, None, &mut census)
            .expect_err("every drawable variant is capped and spent");

        // Counted AND named: the partial draw, not a placeholder.
        assert_ne!(err.drawn_so_far, "<draw>");
        assert!(err.drawn_so_far.contains("slot=7"), "{}", err.drawn_so_far);
        assert!(err.drawn_so_far.contains("population="), "{}", err.drawn_so_far);
        assert!(err.drawn_so_far.contains("base_timeframe="), "{}", err.drawn_so_far);
        // …and the reason names the cap rather than the sampler silently
        // re-admitting the control past its own ceiling.
        assert!(
            matches!(
                err.reason,
                ProposalRefused::Objective(ObjectiveError::AllVariantsCapped { .. })
            ),
            "{:?}",
            err.reason
        );
        assert!(err.reason.to_string().contains("max_draws_per_session"));
    }

    #[test]
    fn a_capped_control_is_never_re_admitted_past_its_ceiling() {
        // The other half of the same hole: the old code read
        //   `if pool.is_empty() { self.drawable_variants.clone() }`
        // so a spent cap produced a draw of the capped variant anyway, with no
        // counter and no census line. It must refuse instead.
        let mut p = test_proposer();
        p.drawable_variants = vec![ObjectiveVariant::B0ScoringTable];
        p.b0_drawn = ObjectiveVariant::B0ScoringTable.spec().max_draws_per_session.unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(5);
        let mut census = ProposerCensus::default();
        for slot in 0..8 {
            assert!(
                p.draw_one(&Session::default(), &mut rng, slot, true, None, None, &mut census)
                    .is_err(),
                "slot {slot} drew a variant whose per-session cap was already spent"
            );
        }
    }

    #[test]
    fn a_spent_replicate_budget_is_not_journalled_as_a_duplicate() {
        // HOLE: both were pushed to `duplicates`, and a replicate exhaustion has
        // no first sighting — the code wrote `first_seen.unwrap_or(sweep)`, i.e.
        // it named the CURRENT sweep as the sweep that first ran a configuration
        // that had never been run. The fold then counted it as
        // `duplicates_refused`. Two different facts, one bucket, one fabricated
        // attribution — and the journal is the only durable record.
        let mut out = empty_drawn_sweep();
        record_identity_refusal(
            &mut out,
            3,
            "fnv64:aaaa",
            "population=512 objective=B1_selectivity[max_in_market=0.10]",
            ProposalRefused::CellReplicatesExhausted { replicates: MAX_REPLICATES_PER_CELL },
            None,
        );
        assert!(out.duplicates.is_empty(), "a spent replicate budget is not a duplicate");
        assert_eq!(out.refused.len(), 1);
        assert_eq!(out.refused[0].slot, 3);
        assert_eq!(out.refused[0].config_hash, "fnv64:aaaa");
        assert!(out.refused[0].reason.to_string().contains("replicate budget"));
        assert_eq!(out.census.cell_replicates_exhausted, 1);
        assert_eq!(out.census.duplicates_refused, 0);
        // …and the example names the configuration, not a placeholder.
        assert!(out.census.examples[0].1.contains("B1_selectivity"));

        // A real duplicate still journals as one, with the REAL first sighting.
        let mut out = empty_drawn_sweep();
        record_identity_refusal(
            &mut out,
            4,
            "fnv64:bbbb",
            "population=512",
            ProposalRefused::Duplicate { first_seen: SweepId(2) },
            Some(SweepId(2)),
        );
        assert_eq!(out.duplicates.len(), 1);
        assert_eq!(out.duplicates[0].first_seen_sweep, SweepId(2));
        assert!(out.refused.is_empty());
        assert_eq!(out.census.duplicates_refused, 1);
    }

    #[test]
    fn u5_counts_and_names_every_slot_the_sweep_never_drew() {
        // HOLE: `break 'slots` returned a short sweep. Slots `slot..100` were
        // never drawn and left NO record — the only trace was the difference
        // between SWEEP_SEARCHES and `proposals.len()`, in the same function
        // whose other early exit refuses "to shorten the sweep silently".
        let mut out = empty_drawn_sweep();
        abandon_remaining_slots(&mut out, 12, PROPOSER_RETRIES);

        assert!(out.exhausted);
        assert_eq!(out.slots_abandoned, SWEEP_SEARCHES - 12);
        // Every abandoned slot has exactly one journal record, in order, and no
        // drawn slot has one.
        assert_eq!(out.refused.len(), SWEEP_SEARCHES - 12);
        let slots: Vec<usize> = out.refused.iter().map(|r| r.slot).collect();
        assert_eq!(slots, (12..SWEEP_SEARCHES).collect::<Vec<_>>());
        // Counted in the census, and named with the reason.
        assert_eq!(out.census.space_refused, SWEEP_SEARCHES - 12);
        let text = out.refused[0].reason.to_string();
        assert!(text.contains("ABANDONED"), "{text}");
        assert!(text.contains("enumerated"), "{text}");
        assert!(text.contains(&format!("{PROPOSER_RETRIES} consecutive")), "{text}");
        // The accounting identity the whole fix exists for: every slot of the
        // sweep is either drawn or named. In `draw_sweep` the slots before
        // `from_slot` each pushed exactly one proposal (the inner loop leaves no
        // other way out), so `from_slot + slots_abandoned == SWEEP_SEARCHES` is
        // the same statement as `proposals + abandoned == SWEEP_SEARCHES`.
        assert_eq!(12 + out.slots_abandoned, SWEEP_SEARCHES);
    }

    #[test]
    fn a_trust_region_revert_names_the_level_it_discarded() {
        // HOLE: `trust_region_reverted` was a bare count. The level the sampler
        // actually chose — a configuration the loop abandoned — was never named,
        // so "which factors keep losing to the parent" was unanswerable from the
        // report.
        let caps = Capabilities::all_for_tests();
        let parent = SearchConfigDelta::new([0; FACTOR_COUNT], &caps).unwrap();
        let mut far = parent;
        for f in FactorId::ALL {
            far = far.with_level(f, 1, &caps).unwrap();
        }
        let mut census = ProposerCensus::default();
        apply_trust_region(far, &parent, &caps, &evidence_uniform(), &mut census).unwrap();
        assert_eq!(census.trust_region_reverted, FACTOR_COUNT - MAX_DELTA_FACTORS);
        assert_eq!(
            census.examples.len(),
            FACTOR_COUNT - MAX_DELTA_FACTORS,
            "every revert must be named, not only counted"
        );
        assert!(census.examples.iter().all(|(what, which)| {
            what.starts_with("trust_region_reverted:") && which.contains("(drawn)")
        }));
    }

    #[test]
    fn the_named_examples_are_bounded_so_a_long_session_cannot_become_its_own_census() {
        let mut census = ProposerCensus::default();
        for i in 0..(CENSUS_EXAMPLES_MAX * 4) {
            note_example(&mut census, format!("what{i}"), format!("which{i}"));
        }
        assert_eq!(census.examples.len(), CENSUS_EXAMPLES_MAX);
    }
}
