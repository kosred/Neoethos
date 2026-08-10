//! A proposal, its identity, and the validation every proposal passes before it
//! is allowed to run.
//!
//! Three properties this module is responsible for, all mechanical:
//!
//! 1. **The run is the proposal.** [`materialise`] applies the two axes in the
//!    one order that survives `DiscoveryConfig::apply_mode_overrides`, then
//!    *verifies* — field by field — that every value the proposal declared is
//!    the value the config carries. A mode override that silently overwrote an
//!    objective floor would be caught here rather than producing a sweep whose
//!    label and content disagree.
//! 2. **Nothing on the money path moves.** [`crate::space::money_path_audit`]
//!    diffs the produced config against the reference over every frozen field.
//!    A violation is **refused and counted, never clamped**: a clamped proposal
//!    is a configuration nobody chose, recorded under a name somebody did.
//! 3. **No silent drops.** Every refusal has a variant, a reason string and a
//!    census line ([`ProposerCensus`]).
//!
//! # Identity: the proposer's key, and the run's hash
//!
//! Dedupe is on [`ProposalKey`] — the full semantic identity of a proposal
//! (every axis-A level, the objective, its parameter, the refusal vector and
//! the replicate seed). It is **not** on `config_hash`, and that is deliberate.
//! `ResolvedConfigStamp` does not carry `max_indicators`, `higher_timeframes`,
//! `corr_threshold`, `min_trades_per_day`, the streaming plan, the working-set
//! cursor or the GA seed, so six axis-A factors and the replicate dimension are
//! invisible to it: deduping on the hash alone would delete most of the space
//! and every replicate, silently. The hash is recorded as provenance, and when
//! two different keys produce the same hash that is a **stamp gap** —
//! [`IdentityOutcome::StampCollision`], counted and named, with the fields to
//! add written into `docs/autoresearch-loop.md` §14.7.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use neoethos_search::discovery::DiscoveryConfig;
use neoethos_search::orchestration::StreamingPlan;
use neoethos_search::run_identity::{
    PayoffCeiling, PayoffCeilingInputs, assert_payoff_floor_reachable,
};

use crate::objective::{
    ObjectiveChoice, ObjectiveError, ObjectiveVariant, RefusalDim, RefusalLevels, ScenarioKind,
};
use crate::space::{
    Capabilities, CursorPolicy, FACTOR_COUNT, FactorId, MoneyPathViolation, SearchConfigDelta,
    SpaceError, money_path_audit,
};

/// A sweep's ordinal in the session.
///
/// **Owned by [`crate::session`]** — re-exported here because a proposal names
/// its parent. Two `SweepId` types would be two histories.
pub use crate::session::SweepId;

/// Where a proposal came from. Carried into the journal and the census, because
/// "the loop improved" means nothing if the improvement came from the uniform
/// floor and the posterior never learned anything.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalOrigin {
    /// Thompson-sampled from the marginal posteriors, inside the trust region.
    Posterior,
    /// One of the `EXPLORATION_FLOOR` uniform draws. Never annealed.
    ExplorationFloor,
    /// Forced to an under-drawn axis-B variant so U4 can be claimed honestly.
    /// A space is not refuted if it was not searched.
    CoverageDebt,
    /// Human- or model-authored, admitted out of band. Must be expressible in
    /// the typed space, is stamped and deduplicated like any other, is counted
    /// separately, and has **no** privileged path through the judge.
    External { note: String },
}

impl ProposalOrigin {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Posterior => "posterior",
            Self::ExplorationFloor => "exploration_floor",
            Self::CoverageDebt => "coverage_debt",
            Self::External { .. } => "external",
        }
    }
}

/// One proposed `(search configuration, objective)` pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    /// The baseline this is a delta from — the session's best-ever screened
    /// sweep, or `None` for a uniform draw.
    pub parent: Option<SweepId>,
    pub axis_a: SearchConfigDelta,
    pub axis_b: ObjectiveChoice,
    /// GA seed. A **replicate dimension, not a factor**: repeated draws of the
    /// same cell with different seeds are how within-cell variance is estimated,
    /// and the judge needs that. Two proposals differing only here are NOT
    /// duplicates.
    pub replicate_seed: u64,
    pub origin: ProposalOrigin,
    /// Slot within the sweep, in draw order. Part of the replay coordinates.
    pub slot: usize,
    /// `ResolvedConfigStamp::config_hash` for the configuration this proposal
    /// resolves to, filled at S3 by the proposer before the proposal is
    /// journalled. Empty until stamped — and a proposal that reaches the runner
    /// with an empty hash is a proposal whose artifacts cannot be attributed,
    /// which [`crate::proposer`] refuses rather than emits.
    #[serde(default)]
    pub config_hash: String,
}

impl Proposal {
    pub fn refusals(&self) -> RefusalLevels {
        self.axis_b.refusals()
    }

    /// `(factor, level)` pairs for the coverage counters and the posterior
    /// ledger — the SAME strings on both sides, so an arm's evidence and its
    /// coverage can never come apart.
    ///
    /// The session folds these on `ProposalDrawn`; the proposer credits the
    /// posterior against them at S7. One function, two readers.
    pub fn coverage_levels(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::with_capacity(FACTOR_COUNT + 6);
        for f in FactorId::ALL {
            out.push((f.label(), f.level_label(self.axis_a.level(f))));
        }
        for d in RefusalDim::ALL {
            out.push((d.label(), d.level_label(self.axis_b.refusals().level(d))));
        }
        out.push(("objective", self.axis_b.variant().label().to_string()));
        out.push((
            "objective_param",
            format!(
                "{}:{}",
                self.axis_b.variant().label(),
                self.axis_b.variant().param_label(self.axis_b.param())
            ),
        ));
        out
    }

    pub fn variant(&self) -> ObjectiveVariant {
        self.axis_b.variant()
    }

    pub fn key(&self) -> ProposalKey {
        ProposalKey {
            axis_a: self.axis_a.levels(),
            variant: self.axis_b.variant(),
            param: self.axis_b.param(),
            refusals: self.axis_b.refusals().levels(),
            replicate_seed: self.replicate_seed,
        }
    }

    /// A one-line description for the census and the journal.
    pub fn describe(&self) -> String {
        let mut parts: Vec<String> = FactorId::ALL
            .into_iter()
            .map(|f| format!("{}={}", f.label(), f.level_label(self.axis_a.level(f))))
            .collect();
        parts.push(format!(
            "objective={}[{}]",
            self.axis_b.variant().label(),
            self.axis_b.variant().param_label(self.axis_b.param())
        ));
        for d in RefusalDim::ALL {
            parts.push(format!("{}={}", d.label(), d.level_label(self.axis_b.refusals().level(d))));
        }
        parts.push(format!("replicate_seed={}", self.replicate_seed));
        parts.join(" ")
    }
}

/// The proposer's semantic identity for a proposal. Complete by construction:
/// it is every level the proposer chose, so nothing the loop varies can be
/// invisible to the dedupe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProposalKey {
    pub axis_a: [u8; FACTOR_COUNT],
    pub variant: ObjectiveVariant,
    pub param: u8,
    pub refusals: [u8; 4],
    pub replicate_seed: u64,
}

/// Why a proposal was not run. Every variant is counted in [`ProposerCensus`]
/// and journalled as `ProposalRefused`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalRefused {
    /// The axis-A delta could not be expressed.
    Space(SpaceError),
    /// The objective could not be expressed.
    Objective(ObjectiveError),
    /// A higher lane whose candle outlives the label horizon.
    LaneHorizon { detail: String },
    /// The configured payoff floor sits above what this configuration's own
    /// resolved geometry can produce. Refused **before** a bar is read: the
    /// answer was fixed before the run started.
    PayoffFloorUnreachable {
        configured_floor: f64,
        enforced_ceiling: f64,
        binding: String,
        detail: String,
    },
    /// The produced config differs from the reference on the money path or on
    /// the validation geometry. Refused, never clamped.
    MoneyPath { violations: Vec<MoneyPathViolation> },
    /// The produced config does not carry a value the proposal declared —
    /// something downstream overwrote it. The run would not be the proposal.
    AppliedValueMismatch { field: String, expected: String, actual: String },
    /// The same semantic key has already been run in this session.
    Duplicate { first_seen: SweepId },
    /// The cell has already been drawn `MAX_REPLICATES_PER_CELL` times. Not a
    /// duplicate — a replicate budget spent. Named separately because "we have
    /// enough variance estimates for this cell" and "we ran this exact
    /// experiment before" are different facts and a census that conflated them
    /// would hide which one the space is running out of.
    CellReplicatesExhausted { replicates: u8 },
}

impl std::fmt::Display for ProposalRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Space(e) => write!(f, "{e}"),
            Self::Objective(e) => write!(f, "{e}"),
            Self::LaneHorizon { detail } => write!(f, "{detail}"),
            Self::PayoffFloorUnreachable {
                configured_floor,
                enforced_ceiling,
                binding,
                detail,
            } => write!(
                f,
                "payoff floor {configured_floor:.2} exceeds this configuration's enforced ceiling \
                 {enforced_ceiling:.2} (binding: {binding}); the run's answer would be fixed \
                 before a bar was read. {detail}"
            ),
            Self::MoneyPath { violations } => write!(
                f,
                "the proposal moved {} frozen field(s): {}",
                violations.len(),
                violations
                    .iter()
                    .map(|v| format!("{} {} -> {}", v.field, v.reference, v.proposed))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::AppliedValueMismatch { field, expected, actual } => write!(
                f,
                "`{field}` was proposed as {expected} but the resolved config carries {actual}: \
                 the run would not be the proposal"
            ),
            Self::Duplicate { first_seen } => {
                write!(f, "identical configuration already run in {first_seen}")
            }
            Self::CellReplicatesExhausted { replicates } => write!(
                f,
                "this cell already has {replicates} replicate(s); the replicate budget for it is \
                 spent"
            ),
        }
    }
}

impl ProposalRefused {
    pub fn reason_label(&self) -> &'static str {
        match self {
            Self::Space(_) => "space",
            Self::Objective(_) => "objective",
            Self::LaneHorizon { .. } => "lane_horizon",
            Self::PayoffFloorUnreachable { .. } => "payoff_floor_unreachable",
            Self::MoneyPath { .. } => "money_path",
            Self::AppliedValueMismatch { .. } => "applied_value_mismatch",
            Self::Duplicate { .. } => "duplicate",
            Self::CellReplicatesExhausted { .. } => "cell_replicates_exhausted",
        }
    }
}

/// A proposal turned into something the runner can execute.
///
/// It carries the resolved `DiscoveryConfig` **and** the `PayoffCeiling` that
/// proves the configuration's own floor is reachable. There is no constructor
/// that omits the ceiling, so an unchecked configuration cannot reach the
/// runner: the type is the gate.
#[derive(Debug, Clone)]
pub struct ProposedRunSpec {
    pub proposal: Proposal,
    pub config: DiscoveryConfig,
    pub streaming_plan: StreamingPlan,
    /// Where the working set starts. `Some` only when the compiled search can
    /// honour it; the runner must fail loudly rather than ignore a `Some` it
    /// cannot apply.
    pub start_cursor: Option<CursorPolicy>,
    pub payoff_ceiling: PayoffCeiling,
    /// Label horizon, in base bars, that the lane/horizon rule was checked
    /// against.
    pub label_horizon_bars: usize,
    /// Values the mode overrides derived rather than the proposal choosing —
    /// named so a reader never mistakes a consequence for a decision.
    pub derived: Vec<String>,
    /// How many hypotheses this single search represents for the honest N.
    /// `1` for every variant except B2, whose declared bucket family multiplies
    /// into the trial count the DSR deflates against (§7.2).
    pub hypothesis_multiplier: usize,
}

impl Proposal {
    /// Turn this proposal into the configuration it names — the one call the
    /// runner makes at S3/S4.
    ///
    /// **It refuses, rather than quietly dropping, anything a bare
    /// `DiscoveryConfig` cannot carry.** The streaming plan and the working-set
    /// cursor are not `DiscoveryConfig` fields, so a caller that took only the
    /// config would run a single whole-vocabulary pass while the journal
    /// recorded `streaming_batches=12` — the silent no-op this crate exists to
    /// prevent, wearing the label of a real experiment. Callers that can honour
    /// a streaming plan use [`Proposal::resolve_spec`] and get both.
    pub fn resolve(&self, base: &DiscoveryConfig) -> anyhow::Result<DiscoveryConfig> {
        let spec = self.resolve_spec(base)?;
        if spec.streaming_plan.enabled || spec.start_cursor.is_some() {
            anyhow::bail!(
                "this proposal carries a streaming plan (max_batches={}) and cursor policy {:?}, \
                 neither of which is a DiscoveryConfig field. `resolve` returns the config alone, \
                 so honouring it here would run a single whole-vocabulary pass under a streaming \
                 label. Call `Proposal::resolve_spec` and pass `streaming_plan` / `start_cursor` \
                 to `run_streaming_working_set`.",
                spec.streaming_plan.max_batches,
                self.axis_a.cursor_policy()
            );
        }
        Ok(spec.config)
    }

    /// The full run spec: the config, the streaming plan, the cursor policy and
    /// the payoff ceiling that proves the proposal's own floor is reachable.
    pub fn resolve_spec(&self, base: &DiscoveryConfig) -> anyhow::Result<ProposedRunSpec> {
        let pip_value_per_lot = base.evaluation_config(None).pip_value_per_lot;
        let inputs = neoethos_search::run_identity::payoff_inputs_for_config(base, pip_value_per_lot);
        materialise(base, self, &inputs).map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Turn a proposal into a run spec, or refuse it by name.
///
/// # Ordering — not negotiable
///
/// ```text
///   baseline
///     -> axis_a.apply_structural       (timeframe, population, prefilter, …)
///     -> DiscoveryConfig::apply_mode_overrides
///          (it rewrites min_trades_per_day in PropFirm and scales the
///           trade-frequency floors by timeframe_label, so it must see the
///           PROPOSED timeframe and must run BEFORE the objective)
///     -> axis_b.apply                  (objective + the refusal vector)
///     -> money_path_audit              (nothing frozen moved)
///     -> verify_applied                (the run IS the proposal)
/// ```
///
/// `payoff_inputs` are the run's own resolved SL/TP band, trailing geometry and
/// round-trip cost — from `neoethos_search::run_identity::payoff_inputs_for_config`,
/// which must be called by the runner **after** the per-run ATR scale is
/// installed. The proposer cannot obtain them itself, and it does not pretend
/// to: no ceiling, no run spec.
pub fn materialise(
    baseline: &DiscoveryConfig,
    proposal: &Proposal,
    payoff_inputs: &PayoffCeilingInputs,
) -> Result<ProposedRunSpec, ProposalRefused> {
    let reference = baseline.clone().apply_mode_overrides();

    let mut cfg = baseline.clone();
    proposal.axis_a.apply_structural(&mut cfg).map_err(ProposalRefused::Space)?;

    // The lane/horizon pairing rule, checked against the horizon this proposal
    // actually implies. An H4 lane under a 35-bar label is refused here, by
    // name, and never run.
    let horizon = proposal.axis_b.label_horizon_bars(0);
    crate::space::lane_horizon_conflict(
        proposal.axis_a.base_timeframe(),
        proposal.axis_a.higher_lanes(),
        horizon,
    )
    .map_err(|e| ProposalRefused::LaneHorizon { detail: e.to_string() })?;

    let before_mode = (cfg.min_trades_per_day, cfg.filtering.min_trades_per_month);
    let mut cfg = cfg.apply_mode_overrides();
    let mut derived = Vec::new();
    if cfg.min_trades_per_day != before_mode.0 {
        derived.push(format!(
            "min_trades_per_day {} -> {} (set by apply_mode_overrides for mode {:?})",
            before_mode.0, cfg.min_trades_per_day, cfg.mode
        ));
    }
    if cfg.filtering.min_trades_per_month != before_mode.1 {
        derived.push(format!(
            "filtering.min_trades_per_month {} -> {} (scaled by apply_mode_overrides for \
             timeframe {})",
            before_mode.1, cfg.filtering.min_trades_per_month, cfg.timeframe_label
        ));
    }

    proposal.axis_b.apply(&mut cfg).map_err(ProposalRefused::Objective)?;

    money_path_audit(&reference, &cfg)
        .map_err(|violations| ProposalRefused::MoneyPath { violations })?;
    verify_applied(proposal, &cfg)?;

    // THE CONFIG-IDENTITY GATE. A floor above what this configuration's own
    // geometry can produce is a run whose answer is fixed before a bar is read.
    let configured_floor = proposal.refusals().value(RefusalDim::PayoffFloor);
    let ceiling = assert_payoff_floor_reachable(configured_floor, payoff_inputs).map_err(|e| {
        // The ceiling itself is still computable; report the numbers, not just
        // the refusal, so the census can say WHICH term bound it.
        let (enforced, binding) =
            match neoethos_search::run_identity::max_achievable_payoff(payoff_inputs) {
                Ok(c) => (c.enforced_ceiling, format!("{:?}", c.binding)),
                Err(_) => (f64::NAN, "uncomputable".to_string()),
            };
        ProposalRefused::PayoffFloorUnreachable {
            configured_floor,
            enforced_ceiling: enforced,
            binding,
            detail: e.to_string(),
        }
    })?;

    let start_cursor = match proposal.axis_a.cursor_policy() {
        CursorPolicy::Start => None,
        other => Some(other),
    };

    Ok(ProposedRunSpec {
        proposal: proposal.clone(),
        streaming_plan: proposal.axis_a.streaming_plan(),
        start_cursor,
        payoff_ceiling: ceiling,
        label_horizon_bars: horizon,
        derived,
        hypothesis_multiplier: proposal
            .axis_b
            .variant()
            .hypothesis_multiplier(proposal.axis_b.param()),
        config: cfg,
    })
}

/// Prove, field by field, that the resolved config carries what the proposal
/// declared. Anything downstream that overwrote a proposed value shows up here
/// as a refusal instead of as a sweep whose label and content disagree.
fn verify_applied(proposal: &Proposal, cfg: &DiscoveryConfig) -> Result<(), ProposalRefused> {
    macro_rules! want {
        ($field:literal, $expected:expr, $actual:expr) => {
            if $expected != $actual {
                return Err(ProposalRefused::AppliedValueMismatch {
                    field: $field.to_string(),
                    expected: format!("{:?}", $expected),
                    actual: format!("{:?}", $actual),
                });
            }
        };
    }

    let a = &proposal.axis_a;
    want!("population", a.population(), cfg.population);
    want!("population_auto", false, cfg.population_auto);
    want!("generations", a.generations(), cfg.generations);
    want!("max_indicators", a.gene_width(), cfg.max_indicators);
    want!(
        "runtime_overrides.prefilter_top_k",
        a.prefilter_width(),
        cfg.runtime_overrides.prefilter_top_k
    );
    want!(
        "runtime_overrides.prefilter_insample_frac",
        a.prefilter_insample_frac(),
        cfg.runtime_overrides.prefilter_insample_frac
    );
    want!("timeframe_label", a.base_timeframe(), cfg.timeframe_label.as_str());
    let lanes: Vec<String> = a.higher_lanes().iter().map(|s| (*s).to_string()).collect();
    want!("higher_timeframes", lanes, cfg.higher_timeframes);
    want!("candidate_count", a.candidate_count(), cfg.candidate_count);
    let (psize, corr) = a.portfolio_shape();
    want!("portfolio_size", psize, cfg.portfolio_size);
    want!("corr_threshold", corr, cfg.corr_threshold);

    let r = proposal.refusals();
    want!(
        "target_profile.min_expectancy_t_stat",
        r.value(RefusalDim::ExpectancySignificance),
        cfg.target_profile.min_expectancy_t_stat
    );
    want!(
        "target_profile.min_win_rate",
        r.value(RefusalDim::WinRateFloor),
        cfg.target_profile.min_win_rate
    );
    want!(
        "target_profile.min_payoff_ratio",
        r.value(RefusalDim::PayoffFloor),
        cfg.target_profile.min_payoff_ratio
    );
    want!("max_pbo", r.value(RefusalDim::CandidatePboCap), cfg.max_pbo);

    if proposal.variant() == ObjectiveVariant::B1Selectivity {
        want!("min_trades_per_day", 0.0, cfg.min_trades_per_day);
        want!(
            "target_profile.max_in_market",
            crate::objective::B1_MAX_IN_MARKET_LEVELS[proposal.axis_b.param() as usize],
            cfg.target_profile.max_in_market
        );
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────
// Identity ledger.
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityOutcome {
    New,
    /// The same semantic key was already run. Refuse and redraw.
    DuplicateKey { first_seen: SweepId },
    /// Two DIFFERENT proposals produced the same `config_hash`: the stamp
    /// cannot tell them apart. The proposal still runs — the proposer's key is
    /// the authoritative identity — but the artifact is not attributable by
    /// hash alone and the gap is named here so it gets fixed rather than
    /// absorbed.
    StampCollision { first_seen: SweepId, differing: Vec<String> },
}

/// Session-wide identity bookkeeping, rebuildable by replaying the journal.
#[derive(Debug, Default, Clone)]
pub struct IdentityLedger {
    by_key: HashMap<ProposalKey, SweepId>,
    by_hash: HashMap<String, (ProposalKey, SweepId)>,
}

impl IdentityLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contains(&self, key: &ProposalKey) -> bool {
        self.by_key.contains_key(key)
    }

    /// The sweep a key was first drawn in — what a duplicate refusal names.
    pub fn first_seen(&self, key: &ProposalKey) -> Option<SweepId> {
        self.by_key.get(key).copied()
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    /// Record a drawn proposal. `config_hash` is `None` before S3 stamping.
    pub fn observe(
        &mut self,
        key: ProposalKey,
        config_hash: Option<&str>,
        sweep: SweepId,
    ) -> IdentityOutcome {
        if let Some(first) = self.by_key.get(&key) {
            return IdentityOutcome::DuplicateKey { first_seen: *first };
        }
        let mut outcome = IdentityOutcome::New;
        if let Some(hash) = config_hash
            && let Some((other, first)) = self.by_hash.get(hash)
            && *other != key
        {
            outcome =
                IdentityOutcome::StampCollision { first_seen: *first, differing: differing(other, &key) };
        }
        self.by_key.insert(key, sweep);
        if let Some(hash) = config_hash {
            self.by_hash.entry(hash.to_string()).or_insert((key, sweep));
        }
        outcome
    }
}

/// Which parts of two keys differ — the census entry for a stamp collision.
fn differing(a: &ProposalKey, b: &ProposalKey) -> Vec<String> {
    let mut out = Vec::new();
    for f in FactorId::ALL {
        if a.axis_a[f.index()] != b.axis_a[f.index()] {
            out.push(format!(
                "{}: {} vs {}",
                f.label(),
                f.level_label(a.axis_a[f.index()]),
                f.level_label(b.axis_a[f.index()])
            ));
        }
    }
    if a.variant != b.variant {
        out.push(format!("objective: {} vs {}", a.variant.label(), b.variant.label()));
    } else if a.param != b.param {
        out.push(format!(
            "objective_param: {} vs {}",
            a.variant.param_label(a.param),
            b.variant.param_label(b.param)
        ));
    }
    for d in RefusalDim::ALL {
        if a.refusals[d.index()] != b.refusals[d.index()] {
            out.push(format!(
                "{}: {} vs {}",
                d.label(),
                d.level_label(a.refusals[d.index()]),
                d.level_label(b.refusals[d.index()])
            ));
        }
    }
    if a.replicate_seed != b.replicate_seed {
        out.push(format!("replicate_seed: {} vs {}", a.replicate_seed, b.replicate_seed));
    }
    out
}

// ───────────────────────────────────────────────────────────────────────────
// The census — no silent drops.
// ───────────────────────────────────────────────────────────────────────────

/// The proposer's half of the session's `AbandonCensus`. Every abandoned
/// configuration is counted **and named**; `examples` is bounded so a long
/// session cannot turn the census into the artifact.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposerCensus {
    pub proposals_drawn: usize,
    pub duplicates_refused: usize,
    /// Cells whose replicate budget was already spent.
    pub cell_replicates_exhausted: usize,
    pub payoff_floor_unreachable: usize,
    pub lane_horizon_refused: usize,
    pub money_path_refused: usize,
    pub applied_value_mismatch: usize,
    pub space_refused: usize,
    pub objective_refused: usize,
    /// Factors reverted toward the parent by the trust region.
    pub trust_region_reverted: usize,
    /// Draws resampled because the drawn level pair was not co-expressible
    /// (an H4 lane against a horizon that cannot span it, for instance).
    pub constraint_resampled: usize,
    /// Different proposals that share a `config_hash`. Not a drop — a stamp gap.
    pub stamp_collisions: usize,
    /// Slots forced to an under-drawn axis-B variant to pay coverage debt.
    pub coverage_debt_forced: usize,
    /// Uniform draws that collided consecutively; `PROPOSER_RETRIES` of these
    /// in a row is U5, `ProposerExhausted`.
    pub uniform_collisions: usize,
    pub examples: Vec<(String, String)>,
}

const CENSUS_EXAMPLES_MAX: usize = 32;

impl ProposerCensus {
    pub fn record_refusal(&mut self, refused: &ProposalRefused, describe: &str) {
        match refused {
            ProposalRefused::Space(_) => self.space_refused += 1,
            ProposalRefused::Objective(_) => self.objective_refused += 1,
            ProposalRefused::LaneHorizon { .. } => self.lane_horizon_refused += 1,
            ProposalRefused::PayoffFloorUnreachable { .. } => self.payoff_floor_unreachable += 1,
            ProposalRefused::MoneyPath { .. } => self.money_path_refused += 1,
            ProposalRefused::AppliedValueMismatch { .. } => self.applied_value_mismatch += 1,
            ProposalRefused::Duplicate { .. } => self.duplicates_refused += 1,
            ProposalRefused::CellReplicatesExhausted { .. } => {
                self.cell_replicates_exhausted += 1
            }
        }
        if self.examples.len() < CENSUS_EXAMPLES_MAX {
            self.examples.push((
                format!("{}: {}", refused.reason_label(), refused),
                describe.to_string(),
            ));
        }
    }

    /// Fold one sweep's census into the session-wide one. Counts add; the
    /// examples are appended until the bound, so a long session cannot turn the
    /// census into the artifact.
    pub fn absorb(&mut self, other: &ProposerCensus) {
        self.proposals_drawn += other.proposals_drawn;
        self.duplicates_refused += other.duplicates_refused;
        self.cell_replicates_exhausted += other.cell_replicates_exhausted;
        self.payoff_floor_unreachable += other.payoff_floor_unreachable;
        self.lane_horizon_refused += other.lane_horizon_refused;
        self.money_path_refused += other.money_path_refused;
        self.applied_value_mismatch += other.applied_value_mismatch;
        self.space_refused += other.space_refused;
        self.objective_refused += other.objective_refused;
        self.trust_region_reverted += other.trust_region_reverted;
        self.constraint_resampled += other.constraint_resampled;
        self.stamp_collisions += other.stamp_collisions;
        self.coverage_debt_forced += other.coverage_debt_forced;
        self.uniform_collisions += other.uniform_collisions;
        for e in &other.examples {
            if self.examples.len() >= CENSUS_EXAMPLES_MAX {
                break;
            }
            self.examples.push(e.clone());
        }
    }

    pub fn total_refused(&self) -> usize {
        self.duplicates_refused
            + self.cell_replicates_exhausted
            + self.payoff_floor_unreachable
            + self.lane_horizon_refused
            + self.money_path_refused
            + self.applied_value_mismatch
            + self.space_refused
            + self.objective_refused
    }
}

/// Convenience: build a proposal without going through the sampler. Every
/// argument is validated by the types it is built from, so this cannot widen
/// the space — it is how the loop admits an [`ProposalOrigin::External`]
/// suggestion, and how tests build a known point.
pub fn build(
    axis_a: SearchConfigDelta,
    variant: ObjectiveVariant,
    param: u8,
    refusals: RefusalLevels,
    replicate_seed: u64,
    origin: ProposalOrigin,
    slot: usize,
    parent: Option<SweepId>,
    caps: &Capabilities,
    scenario: ScenarioKind,
) -> Result<Proposal, ProposalRefused> {
    let axis_b = ObjectiveChoice::new(variant, param, refusals, caps, scenario)
        .map_err(ProposalRefused::Objective)?;
    Ok(Proposal {
        parent,
        axis_a,
        axis_b,
        replicate_seed,
        origin,
        slot,
        // Filled at S3 by `proposer::Proposer::draw_sweep`, which stamps every
        // proposal before it is journalled.
        config_hash: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> Capabilities {
        Capabilities::all_for_tests()
    }

    fn delta(levels: [u8; FACTOR_COUNT]) -> SearchConfigDelta {
        SearchConfigDelta::new(levels, &caps()).unwrap()
    }

    fn proposal(levels: [u8; FACTOR_COUNT], refusals: [u8; 4]) -> Proposal {
        build(
            delta(levels),
            ObjectiveVariant::B5TerminalWealth,
            0,
            RefusalLevels::new(refusals).unwrap(),
            7,
            ProposalOrigin::ExplorationFloor,
            0,
            None,
            &caps(),
            ScenarioKind::Risky,
        )
        .unwrap()
    }

    /// The shipped exit geometry: the trail arms at 1R and gives back 1R, so it
    /// is in the regime the 1.08 measurement covers.
    fn shipped_inputs() -> PayoffCeilingInputs {
        PayoffCeilingInputs {
            sl_min_pips: 6.0,
            sl_max_pips: 40.0,
            tp_min_pips: 6.0,
            tp_max_pips: 45.0,
            initializer_rr_max: 4.0,
            initializer_rr_min: 1.5,
            atr_pips: Some(8.0),
            cost_pips_round_trip: 3.4,
            trailing_enabled: true,
            trailing_be_trigger_r: 1.0,
            trailing_give_back_r: 1.0,
            trailing_min_lock_pips: 2.0,
        }
    }

    #[test]
    fn a_payoff_floor_above_the_measured_trailing_ceiling_is_refused_before_a_bar_is_read() {
        // level 2 of the payoff-floor dimension is 2.0; the trail caps the
        // enforced ceiling at MEASURED_TRAILING_PAYOFF_CEILING = 1.08.
        let p = proposal([0; FACTOR_COUNT], [0, 0, 2, 0]);
        let inputs = shipped_inputs();
        let ceiling = neoethos_search::run_identity::max_achievable_payoff(&inputs).unwrap();
        assert!(ceiling.enforced_ceiling < 2.0, "test fixture no longer exercises the gate");
        let err = assert_payoff_floor_reachable(
            p.refusals().value(RefusalDim::PayoffFloor),
            &inputs,
        )
        .unwrap_err();
        assert!(err.to_string().to_lowercase().contains("payoff"));
        // …and a floor at or below the ceiling is accepted.
        assert!(assert_payoff_floor_reachable(0.0, &inputs).is_ok());
    }

    #[test]
    fn identical_keys_are_duplicates_and_different_seeds_are_not() {
        let mut ledger = IdentityLedger::new();
        let a = proposal([0; FACTOR_COUNT], [0; 4]);
        let mut b = a.clone();
        b.replicate_seed = 8;

        assert_eq!(ledger.observe(a.key(), None, SweepId(1)), IdentityOutcome::New);
        assert_eq!(
            ledger.observe(a.key(), None, SweepId(2)),
            IdentityOutcome::DuplicateKey { first_seen: SweepId(1) }
        );
        // The replicate is a DIFFERENT experiment — deduping it away would
        // delete the within-cell variance the judge needs.
        assert_eq!(ledger.observe(b.key(), None, SweepId(2)), IdentityOutcome::New);
    }

    #[test]
    fn two_proposals_sharing_a_config_hash_are_a_named_stamp_gap_not_a_drop() {
        let mut ledger = IdentityLedger::new();
        let a = proposal([0; FACTOR_COUNT], [0; 4]);
        let mut levels = [0u8; FACTOR_COUNT];
        // gene_width is NOT in ResolvedConfigStamp, so these two really can
        // hash identically today.
        levels[FactorId::GeneWidth.index()] = 2;
        let b = proposal(levels, [0; 4]);

        assert_eq!(ledger.observe(a.key(), Some("fnv64:same"), SweepId(1)), IdentityOutcome::New);
        match ledger.observe(b.key(), Some("fnv64:same"), SweepId(2)) {
            IdentityOutcome::StampCollision { first_seen, differing } => {
                assert_eq!(first_seen, SweepId(1));
                assert!(differing.iter().any(|d| d.starts_with("gene_width")), "{differing:?}");
            }
            other => panic!("expected a named stamp collision, got {other:?}"),
        }
    }

    #[test]
    fn the_census_names_every_refusal_and_never_just_counts_it() {
        let mut census = ProposerCensus::default();
        let p = proposal([0; FACTOR_COUNT], [0; 4]);
        census.record_refusal(&ProposalRefused::Duplicate { first_seen: SweepId(3) }, &p.describe());
        census.record_refusal(
            &ProposalRefused::PayoffFloorUnreachable {
                configured_floor: 2.0,
                enforced_ceiling: 1.08,
                binding: "TrailingGiveBack".to_string(),
                detail: "measured".to_string(),
            },
            &p.describe(),
        );
        assert_eq!(census.total_refused(), 2);
        assert_eq!(census.examples.len(), 2);
        assert!(census.examples[1].0.contains("payoff_floor_unreachable"));
    }

    #[test]
    fn the_key_covers_every_factor_the_proposer_varies() {
        // If a factor were missing from the key, two different configurations
        // would dedupe into one and the space would silently shrink.
        let base = proposal([0; FACTOR_COUNT], [0; 4]);
        for f in FactorId::ALL {
            if f.arity() < 2 {
                continue;
            }
            let mut levels = [0u8; FACTOR_COUNT];
            levels[f.index()] = 1;
            let other = proposal(levels, [0; 4]);
            assert_ne!(base.key(), other.key(), "{} is invisible to ProposalKey", f.label());
        }
        for d in RefusalDim::ALL {
            let mut refusals = [0u8; 4];
            refusals[d.index()] = 1;
            let other = proposal([0; FACTOR_COUNT], refusals);
            assert_ne!(base.key(), other.key(), "{} is invisible to ProposalKey", d.label());
        }
    }
}
