//! Axis B — the objective variants, and the refusal vector.
//!
//! This is the half that is usually skipped, and it is the half where the
//! measured base rate says the answer must live: expectancy was **−4.15 pips
//! per trade in every exit configuration tested** while the payoff ratio moved
//! 0.91 → 2.53. Exit geometry redistributes between win rate and payoff; the
//! product stays at minus the cost.
//!
//! # The variants are READ, never synthesised
//!
//! [`ObjectiveVariant`] is a closed enum and [`VARIANTS`] is a `const` table.
//! There is no constructor that takes a formula, a weight vector, a field name
//! or a string. A proposer that could invent an objective could reach any goal
//! you like and mean nothing — so it cannot: the only thing the sampler chooses
//! is an index into this table.
//!
//! # What does NOT count as a variant (§7.0)
//!
//! Named here so no future builder adds them and no reader mistakes them for
//! content. Each only reshapes the (win-rate, payoff) split of a fixed
//! population of trades, and the product of that split is pinned at minus the
//! cost:
//!
//! * **exit geometry** — trailing on/off, `be_trigger_r`, give-back, min-lock,
//!   SL/TP clamps. Measured: expectancy −4.15 pips in every one of them.
//! * **the RR ladder / `min_payoff_ratio` as an objective** — the same thing
//!   under another name. Payoff 2.53 at expectancy −4.18 pips is a
//!   gate-passing money-loser. (It survives as a *refusal* level, which is a
//!   different thing: a refusal cannot buy a promotion, only cost candidates.)
//! * **re-weighting the four scoring tables** — different monotone summaries of
//!   the *same* in-sample per-trade distribution. Worth exactly **one** sweep as
//!   a control ([`ObjectiveVariant::B0ScoringTable`]), never a family.
//!
//! A genuinely different objective must change at least one of: **(i)** which
//! trades are eligible at all, **(ii)** what quantity is estimated, **(iii)**
//! what the unit of evaluation is, or **(iv)** what event is predicted. Every
//! entry in [`VARIANTS`] declares which, in [`ObjectiveDimension`], and a test
//! asserts none of them is `None`.

use serde::{Deserialize, Serialize};

use neoethos_search::discovery::DiscoveryConfig;

use crate::space::{Capabilities, RequiredKnob};

/// Which scenario a session is optimising toward.
///
/// **Owned by [`crate::goals`]**, which loads it from `system.trading_mode` —
/// re-exported here rather than redeclared, because two `ScenarioKind`s would
/// be two answers to "which goal is this session optimising toward", and the
/// whole point of the goal constants is that there is exactly one.
pub use crate::goals::ScenarioKind;

/// The four ways an objective can genuinely differ (§7.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDimension {
    /// (i) which trades are eligible at all
    EligiblePopulation,
    /// (ii) what quantity is being estimated
    EstimatedQuantity,
    /// (iii) what the unit of evaluation is — trade vs path vs month vs portfolio
    EvaluationUnit,
    /// (iv) what event is being predicted
    PredictedEvent,
}

/// The declared objective set. Closed, ordered, and part of the wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveVariant {
    /// The scoring-table control. One sweep, ever — a family of these would be
    /// a family of re-parameterisations.
    B0ScoringTable,
    /// Selectivity / abstention — fewer, different trades.
    B1Selectivity,
    /// Conditional / regime-restricted expectancy.
    B2Conditional,
    /// Cost-elastic: score at the pessimistic edge of the cost band.
    B3CostElastic,
    /// Label / holding horizon — the event being predicted.
    B4LabelHorizon,
    /// Path-level / terminal-wealth.
    B5TerminalWealth,
    /// Monthly consistency under path constraints (the prop-firm half).
    B6MonthlyConsistency,
    /// Significance-first.
    B7Significance,
    /// Portfolio-level objective.
    B8Portfolio,
}

pub const VARIANT_COUNT: usize = 9;

/// How a variant reaches the search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Expression {
    /// Fully expressible through declared overrides on `DiscoveryConfig`.
    Overrides,
    /// The variant *is* the mode's own objective for its scenario: nothing is
    /// overridden. Recorded as a named variant anyway, so the coverage table
    /// does not silently treat "the default ran" as "the variant ran", and so
    /// `exact_form_requires` can state what the exact functional would need.
    ModeNative,
}

/// One row of the declared objective set.
#[derive(Debug, Clone, Copy)]
pub struct VariantSpec {
    pub id: ObjectiveVariant,
    pub label: &'static str,
    /// Which of the four dimensions this variant actually moves.
    pub changes: ObjectiveDimension,
    /// `None` = either scenario; `Some(k)` = only under scenario `k`.
    pub scenario: Option<ScenarioKind>,
    pub expression: Expression,
    /// Knobs required to draw it at all.
    pub requires: &'static [RequiredKnob],
    /// Knob required for the *exact* functional §7 describes, when the
    /// expressible form is a mode-native approximation of it. Always rendered
    /// beside the variant in the report — an approximation that is not stated
    /// is a silent substitution.
    pub exact_form_requires: Option<RequiredKnob>,
    /// `DiscoveryConfig` fields the variant writes. Asserted disjoint from
    /// `FROZEN_FIELDS`.
    pub writes: &'static [&'static str],
    /// Number of variant-scoped parameter levels (1 = the variant has none).
    pub params: usize,
    /// Draw ceiling per session, when the variant is a control.
    pub max_draws_per_session: Option<usize>,
    /// The one-line justification that this is not a re-parameterisation.
    pub why_not_a_reparameterisation: &'static str,
}

/// B1's exposure ceilings — the concrete "fewer, different trades" lever.
pub const B1_MAX_IN_MARKET_LEVELS: &[f64] = &[0.10, 0.05, 0.02];
/// B2's conditioning families, declared BEFORE the run. Choosing the bucket
/// after seeing the result is the classic subgroup overfit; the family is fixed
/// here, the complement's expectancy is reported alongside always, and the
/// bucket count multiplies into the trial count the DSR deflates against.
pub const B2_CONDITIONING_LEVELS: &[(&str, usize)] = &[
    ("session:asia|london|ny", 3),
    ("atr_percentile_bucket", 4),
    ("day_of_week", 5),
    ("regime_label", 3),
    ("post_high_impact_news_window", 2),
];
/// B4's label horizons, in base bars. `35` is the shipped default and it
/// resolves inside a single H4 candle — which is why the H4 lane is reachable
/// only from `120` upward on an M5 base.
pub const B4_HORIZON_LEVELS: &[usize] = &[35, 120, 480, 1440];
/// B0's four scoring tables (`neoethos_search::scoring::named`).
pub const B0_TABLE_LEVELS: &[&str] = &["ga_fitness", "archive_score", "window_score", "quality_score"];

/// The declared set. **Read-only.**
pub const VARIANTS: [VariantSpec; VARIANT_COUNT] = [
    VariantSpec {
        id: ObjectiveVariant::B0ScoringTable,
        label: "B0_scoring_table",
        changes: ObjectiveDimension::EstimatedQuantity,
        scenario: None,
        expression: Expression::Overrides,
        requires: &[RequiredKnob::FitnessTable],
        exact_form_requires: None,
        writes: &["fitness_table"],
        params: 4,
        // One sweep, as a control. A family of monotone re-summaries of one
        // per-trade distribution is not a family of objectives.
        max_draws_per_session: Some(1),
        why_not_a_reparameterisation:
            "IT IS ONE — carried only as the named control the other eight are compared against.",
    },
    VariantSpec {
        id: ObjectiveVariant::B1Selectivity,
        label: "B1_selectivity",
        changes: ObjectiveDimension::EligiblePopulation,
        scenario: None,
        expression: Expression::Overrides,
        requires: &[RequiredKnob::InMarketFitness],
        exact_form_requires: None,
        writes: &["min_trades_per_day", "target_profile.max_in_market", "fitness_table"],
        params: 3,
        max_draws_per_session: None,
        why_not_a_reparameterisation:
            "the average trade loses 4.15 pips, so the only exit is FEWER, DIFFERENT trades: the \
             volume floor is removed, a hard exposure budget is imposed, and net is scored per BAR \
             IN MARKET so being flat is not penalised.",
    },
    VariantSpec {
        id: ObjectiveVariant::B2Conditional,
        label: "B2_conditional",
        changes: ObjectiveDimension::EligiblePopulation,
        scenario: None,
        expression: Expression::Overrides,
        requires: &[RequiredKnob::ConditioningSet],
        exact_form_requires: None,
        writes: &["conditioning_set"],
        params: 5,
        max_draws_per_session: None,
        why_not_a_reparameterisation:
            "expectancy is estimated WITHIN a conditioning set declared before the run; the \
             complement is reported alongside always and the family's bucket count multiplies into \
             the trial count N the DSR deflates against.",
    },
    VariantSpec {
        id: ObjectiveVariant::B3CostElastic,
        label: "B3_cost_elastic",
        changes: ObjectiveDimension::EstimatedQuantity,
        scenario: None,
        expression: Expression::Overrides,
        requires: &[RequiredKnob::CostEdgeScoring],
        exact_form_requires: None,
        writes: &["score_at_cost_band_edge"],
        params: 1,
        max_draws_per_session: None,
        why_not_a_reparameterisation:
            "cost scales with trade COUNT, so scoring at the pessimistic band edge systematically \
             re-ranks a high-frequency candidate below a low-frequency one with the same \
             point-estimate net. It is not a monotone transform of the point-estimate ranking.",
    },
    VariantSpec {
        id: ObjectiveVariant::B4LabelHorizon,
        label: "B4_label_horizon",
        changes: ObjectiveDimension::PredictedEvent,
        scenario: None,
        expression: Expression::Overrides,
        requires: &[RequiredKnob::LabelHorizon],
        exact_form_requires: None,
        writes: &["label_max_hold_bars"],
        params: 4,
        max_draws_per_session: None,
        why_not_a_reparameterisation:
            "the triple-barrier horizon is the TARGET VARIABLE the prefilter ranks against and the \
             GA is scored on — not an exit rule. It is the only honest route to the H4 lane: a \
             35-bar label resolves inside a single H4 candle.",
    },
    VariantSpec {
        id: ObjectiveVariant::B5TerminalWealth,
        label: "B5_terminal_wealth",
        changes: ObjectiveDimension::EvaluationUnit,
        scenario: Some(ScenarioKind::Risky),
        // Risky mode already evolves under `scoring::ga_fitness_growth` (Kelly
        // log-growth, scoring_version 5) with the growth-tilted post-GA
        // ranking. That is the mode's own objective, so the variant overrides
        // nothing — and `exact_form_requires` says what the literal
        // `p_reach_target` functional of §7.5 would still need.
        expression: Expression::ModeNative,
        requires: &[],
        exact_form_requires: Some(RequiredKnob::PortfolioObjective),
        writes: &[],
        params: 1,
        max_draws_per_session: None,
        why_not_a_reparameterisation:
            "the unit is a PATH: two candidates with identical per-trade expectancy have different \
             P(reach target) through variance, compounding and serial structure. It cannot \
             manufacture edge — goal_report's own test shows a negative-edge system reaches a 500x \
             target with probability < 0.05 at EVERY risk level.",
    },
    VariantSpec {
        id: ObjectiveVariant::B6MonthlyConsistency,
        label: "B6_monthly_consistency",
        changes: ObjectiveDimension::EvaluationUnit,
        scenario: Some(ScenarioKind::PropFirm),
        // PropFirm mode installs the window-pass gate (`derive_prop_firm_gate`)
        // and judges on windows clearing the monthly bar without violating the
        // daily-loss or overall-drawdown rules. That is this objective.
        expression: Expression::ModeNative,
        requires: &[],
        exact_form_requires: None,
        writes: &[],
        params: 1,
        max_draws_per_session: None,
        why_not_a_reparameterisation:
            "the unit is a MONTH under path constraints. A drawdown-path constraint is not a \
             function of the per-trade mean: two candidates with identical expectancy can be \
             either inside or outside the rule.",
    },
    VariantSpec {
        id: ObjectiveVariant::B7Significance,
        label: "B7_significance",
        changes: ObjectiveDimension::EstimatedQuantity,
        scenario: None,
        expression: Expression::Overrides,
        requires: &[RequiredKnob::TStatObjective],
        exact_form_requires: None,
        writes: &["fitness_table"],
        params: 1,
        max_draws_per_session: None,
        why_not_a_reparameterisation:
            "maximising the expectancy t-STATISTIC rather than the expectancy penalises the \
             high-variance low-count candidates that dominate an expectancy ranking, and attacks \
             selection bias at the candidate level where it is cheapest to fight.",
    },
    VariantSpec {
        id: ObjectiveVariant::B8Portfolio,
        label: "B8_portfolio",
        changes: ObjectiveDimension::EvaluationUnit,
        scenario: None,
        expression: Expression::Overrides,
        requires: &[RequiredKnob::PortfolioObjective],
        exact_form_requires: None,
        writes: &["score_portfolio_after_pruning"],
        params: 1,
        max_draws_per_session: None,
        why_not_a_reparameterisation:
            "the selection unit is the PORTFOLIO after correlation pruning, not the best single \
             gene. Diversification is a real, non-redistributive effect on path statistics, and the \
             live artifact is a portfolio anyway.",
    },
];

impl ObjectiveVariant {
    pub const ALL: [ObjectiveVariant; VARIANT_COUNT] = [
        ObjectiveVariant::B0ScoringTable,
        ObjectiveVariant::B1Selectivity,
        ObjectiveVariant::B2Conditional,
        ObjectiveVariant::B3CostElastic,
        ObjectiveVariant::B4LabelHorizon,
        ObjectiveVariant::B5TerminalWealth,
        ObjectiveVariant::B6MonthlyConsistency,
        ObjectiveVariant::B7Significance,
        ObjectiveVariant::B8Portfolio,
    ];

    /// The declared set, as a slice. Same order as [`ObjectiveVariant::ALL`].
    pub fn all() -> &'static [ObjectiveVariant] {
        &Self::ALL
    }

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|v| *v == self).expect("ALL covers the enum")
    }

    pub fn spec(self) -> &'static VariantSpec {
        &VARIANTS[self.index()]
    }

    pub fn label(self) -> &'static str {
        self.spec().label
    }

    /// The variants U4 requires coverage of: everything except the B0 control,
    /// which is capped at one draw and so could never reach `B_MIN_DRAWS`.
    pub fn coverage_set() -> impl Iterator<Item = ObjectiveVariant> {
        Self::ALL.into_iter().filter(|v| v.spec().max_draws_per_session.is_none())
    }

    /// Drawable iff every required knob is compiled in and the scenario matches.
    pub fn drawable(self, caps: &Capabilities, scenario: ScenarioKind) -> bool {
        let spec = self.spec();
        if let Some(s) = spec.scenario
            && s != scenario
        {
            return false;
        }
        spec.requires.iter().all(|k| k.available(caps))
    }

    /// Why it is not drawable, named. Never `None` when `drawable` is false.
    pub fn exclusion_reason(
        self,
        caps: &Capabilities,
        scenario: ScenarioKind,
    ) -> Option<VariantExclusion> {
        let spec = self.spec();
        if let Some(s) = spec.scenario
            && s != scenario
        {
            return Some(VariantExclusion::WrongScenario { requires: s, session: scenario });
        }
        for k in spec.requires {
            if !k.available(caps) {
                return Some(VariantExclusion::MissingKnob(*k));
            }
        }
        None
    }

    pub fn param_count(self) -> usize {
        self.spec().params
    }

    pub fn param_label(self, param: u8) -> String {
        let p = param as usize;
        match self {
            Self::B0ScoringTable => B0_TABLE_LEVELS.get(p).map(|s| (*s).to_string()),
            Self::B1Selectivity => {
                B1_MAX_IN_MARKET_LEVELS.get(p).map(|v| format!("max_in_market={v:.2}"))
            }
            Self::B2Conditional => {
                B2_CONDITIONING_LEVELS.get(p).map(|(n, b)| format!("{n}({b} buckets)"))
            }
            Self::B4LabelHorizon => B4_HORIZON_LEVELS.get(p).map(|v| format!("{v} bars")),
            _ => (p == 0).then(|| "-".to_string()),
        }
        .unwrap_or_else(|| format!("<out-of-range:{param}>"))
    }

    /// How many hypotheses a variant's parameter implies, for the honest N.
    ///
    /// B2 declares a FAMILY of buckets, and the whole point of §7.2 is that the
    /// bucket count multiplies into the trial count the DSR deflates against.
    /// Every other variant contributes 1.
    pub fn hypothesis_multiplier(self, param: u8) -> usize {
        match self {
            Self::B2Conditional => {
                B2_CONDITIONING_LEVELS.get(param as usize).map(|(_, b)| *b).unwrap_or(1)
            }
            _ => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariantExclusion {
    MissingKnob(RequiredKnob),
    WrongScenario { requires: ScenarioKind, session: ScenarioKind },
}

// ───────────────────────────────────────────────────────────────────────────
// The refusal vector — what the SEARCH refuses (§7.9).
// ───────────────────────────────────────────────────────────────────────────

pub const EXPECTANCY_SIGNIFICANCE_LEVELS: &[f64] = &[0.0, 2.0, 3.0];
pub const WIN_RATE_FLOOR_LEVELS: &[f64] = &[0.0, 0.35, 0.45];
pub const PAYOFF_FLOOR_LEVELS: &[f64] = &[0.0, 1.0, 2.0];
/// The per-candidate CPCV ceiling (`DiscoveryConfig::max_pbo`). A **different
/// object** from the judge's session-level CSCV PBO, which is not a
/// `DiscoveryConfig` field and is not reachable from here at all.
pub const CANDIDATE_PBO_CAP_LEVELS: &[f64] = &[1.0, 0.7, 0.5];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalDim {
    ExpectancySignificance,
    WinRateFloor,
    PayoffFloor,
    CandidatePboCap,
}

pub const REFUSAL_DIM_COUNT: usize = 4;

impl RefusalDim {
    pub const ALL: [RefusalDim; REFUSAL_DIM_COUNT] = [
        RefusalDim::ExpectancySignificance,
        RefusalDim::WinRateFloor,
        RefusalDim::PayoffFloor,
        RefusalDim::CandidatePboCap,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::ExpectancySignificance => 0,
            Self::WinRateFloor => 1,
            Self::PayoffFloor => 2,
            Self::CandidatePboCap => 3,
        }
    }

    pub fn arity(self) -> usize {
        self.values().len()
    }

    pub fn values(self) -> &'static [f64] {
        match self {
            Self::ExpectancySignificance => EXPECTANCY_SIGNIFICANCE_LEVELS,
            Self::WinRateFloor => WIN_RATE_FLOOR_LEVELS,
            Self::PayoffFloor => PAYOFF_FLOOR_LEVELS,
            Self::CandidatePboCap => CANDIDATE_PBO_CAP_LEVELS,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ExpectancySignificance => "expectancy_significance",
            Self::WinRateFloor => "win_rate_floor",
            Self::PayoffFloor => "payoff_floor",
            Self::CandidatePboCap => "candidate_pbo_cap",
        }
    }

    pub fn field(self) -> &'static str {
        match self {
            Self::ExpectancySignificance => "target_profile.min_expectancy_t_stat",
            Self::WinRateFloor => "target_profile.min_win_rate",
            Self::PayoffFloor => "target_profile.min_payoff_ratio",
            Self::CandidatePboCap => "max_pbo",
        }
    }

    pub fn level_label(self, level: u8) -> String {
        self.values()
            .get(level as usize)
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| format!("<out-of-range:{level}>"))
    }
}

/// The four search-side floors, as a vector. Independent of the variant, so
/// "what is maximised" and "what is refused" move separately — which is exactly
/// what the operator asked for.
///
/// `TargetProfile::min_net_expectancy_per_trade` is **not** a dimension here. It
/// is unconditional in `TargetProfile::evaluate` and it is the floor under
/// everything else; a loop that could raise or lower it would be varying the one
/// gate that says "the average trade must make money".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RefusalLevels {
    levels: [u8; REFUSAL_DIM_COUNT],
}

impl RefusalLevels {
    pub fn new(levels: [u8; REFUSAL_DIM_COUNT]) -> Result<Self, ObjectiveError> {
        for d in RefusalDim::ALL {
            let l = levels[d.index()];
            if (l as usize) >= d.arity() {
                return Err(ObjectiveError::RefusalLevelOutOfRange {
                    dim: d,
                    level: l,
                    arity: d.arity(),
                });
            }
        }
        Ok(Self { levels })
    }

    /// The permissive corner: every optional floor off. The baseline of the
    /// refusal vector, not a preference.
    pub fn permissive() -> Self {
        Self { levels: [0; REFUSAL_DIM_COUNT] }
    }

    pub fn level(&self, dim: RefusalDim) -> u8 {
        self.levels[dim.index()]
    }

    pub fn levels(&self) -> [u8; REFUSAL_DIM_COUNT] {
        self.levels
    }

    pub fn value(&self, dim: RefusalDim) -> f64 {
        dim.values()[self.level(dim) as usize]
    }

    pub fn with_level(mut self, dim: RefusalDim, level: u8) -> Result<Self, ObjectiveError> {
        self.levels[dim.index()] = level;
        Self::new(self.levels)
    }

    pub fn differing_dims(&self, other: &Self) -> Vec<RefusalDim> {
        RefusalDim::ALL
            .into_iter()
            .filter(|d| self.levels[d.index()] != other.levels[d.index()])
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveError {
    RefusalLevelOutOfRange { dim: RefusalDim, level: u8, arity: usize },
    ParamOutOfRange { variant: ObjectiveVariant, param: u8, arity: usize },
    VariantNotDrawable { variant: ObjectiveVariant, reason: VariantExclusion },
}

impl std::fmt::Display for ObjectiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RefusalLevelOutOfRange { dim, level, arity } => write!(
                f,
                "refusal dimension `{}` has {arity} levels; level {level} does not exist",
                dim.label()
            ),
            Self::ParamOutOfRange { variant, param, arity } => write!(
                f,
                "objective `{}` has {arity} parameter levels; level {param} does not exist",
                variant.label()
            ),
            Self::VariantNotDrawable { variant, reason } => match reason {
                VariantExclusion::MissingKnob(k) => write!(
                    f,
                    "objective `{}` needs `{}` (cargo feature `{}`, edit in {}), which the \
                     compiled neoethos-search does not provide — the variant is EXCLUDED from the \
                     drawable space rather than silently reduced to the default objective",
                    variant.label(),
                    k.symbol(),
                    k.feature(),
                    k.required_elsewhere_section()
                ),
                VariantExclusion::WrongScenario { requires, session } => write!(
                    f,
                    "objective `{}` serves the {requires:?} scenario; this session's goal set is \
                     {session:?}",
                    variant.label()
                ),
            },
        }
    }
}

impl std::error::Error for ObjectiveError {}

/// The axis-B half of a proposal: a variant, its parameter, and the refusal
/// vector. Constructed only through [`ObjectiveChoice::new`], which validates
/// against the declared table and the compiled capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectiveChoice {
    variant: ObjectiveVariant,
    param: u8,
    refusals: RefusalLevels,
}

impl ObjectiveChoice {
    pub fn new(
        variant: ObjectiveVariant,
        param: u8,
        refusals: RefusalLevels,
        caps: &Capabilities,
        scenario: ScenarioKind,
    ) -> Result<Self, ObjectiveError> {
        if let Some(reason) = variant.exclusion_reason(caps, scenario) {
            return Err(ObjectiveError::VariantNotDrawable { variant, reason });
        }
        if (param as usize) >= variant.param_count() {
            return Err(ObjectiveError::ParamOutOfRange {
                variant,
                param,
                arity: variant.param_count(),
            });
        }
        Ok(Self { variant, param, refusals })
    }

    pub fn variant(&self) -> ObjectiveVariant {
        self.variant
    }
    pub fn param(&self) -> u8 {
        self.param
    }
    pub fn refusals(&self) -> RefusalLevels {
        self.refusals
    }

    /// The label horizon this proposal implies, in base bars.
    ///
    /// `baseline_hold` is the run's own configured horizon (`0` in the shipped
    /// `EvaluationConfig`, which `discovery.rs` then reads as the documented 35).
    /// Used by the lane/horizon pairing rule: an H4 lane is only ever proposed
    /// with a horizon that can express it.
    pub fn label_horizon_bars(&self, baseline_hold: usize) -> usize {
        match self.variant {
            ObjectiveVariant::B4LabelHorizon => B4_HORIZON_LEVELS[self.param as usize],
            _ => {
                if baseline_hold > 0 {
                    baseline_hold
                } else {
                    35
                }
            }
        }
    }

    /// Apply the objective and the refusal vector.
    ///
    /// **Runs AFTER `DiscoveryConfig::apply_mode_overrides`**, because that
    /// function rewrites `min_trades_per_day` in PropFirm mode: an objective
    /// applied before it would be silently overwritten and the run would not be
    /// the proposal. [`crate::proposal::materialise`] owns the ordering and
    /// verifies it afterwards.
    pub fn apply(&self, cfg: &mut DiscoveryConfig) -> Result<(), ObjectiveError> {
        // ── the refusal vector: what the SEARCH refuses ─────────────────────
        cfg.target_profile.min_expectancy_t_stat =
            self.refusals.value(RefusalDim::ExpectancySignificance);
        cfg.target_profile.min_win_rate = self.refusals.value(RefusalDim::WinRateFloor);
        cfg.target_profile.min_payoff_ratio = self.refusals.value(RefusalDim::PayoffFloor);
        cfg.max_pbo = self.refusals.value(RefusalDim::CandidatePboCap);

        // ── the objective ──────────────────────────────────────────────────
        match self.variant {
            ObjectiveVariant::B1Selectivity => {
                // The volume floor is REMOVED, not lowered: the average trade
                // loses 4.15 pips, so a floor that demands more trades demands
                // more of the loss.
                cfg.min_trades_per_day = 0.0;
                cfg.target_profile.max_in_market = B1_MAX_IN_MARKET_LEVELS[self.param as usize];
                #[cfg(feature = "search-in-market-fitness")]
                {
                    cfg.fitness_table = neoethos_search::scoring::FitnessTable::NetPerBarInMarket;
                }
            }
            ObjectiveVariant::B0ScoringTable => {
                #[cfg(feature = "search-fitness-table")]
                {
                    cfg.fitness_table =
                        neoethos_search::scoring::FitnessTable::from_name(
                            B0_TABLE_LEVELS[self.param as usize],
                        );
                }
            }
            ObjectiveVariant::B2Conditional => {
                #[cfg(feature = "search-conditioning-set")]
                {
                    cfg.conditioning_set =
                        Some(B2_CONDITIONING_LEVELS[self.param as usize].0.to_string());
                }
            }
            ObjectiveVariant::B3CostElastic => {
                #[cfg(feature = "search-cost-edge-scoring")]
                {
                    cfg.score_at_cost_band_edge = true;
                }
            }
            ObjectiveVariant::B4LabelHorizon => {
                #[cfg(feature = "search-label-horizon")]
                {
                    cfg.label_max_hold_bars = B4_HORIZON_LEVELS[self.param as usize];
                }
            }
            ObjectiveVariant::B7Significance => {
                #[cfg(feature = "search-t-stat-objective")]
                {
                    cfg.fitness_table = neoethos_search::scoring::FitnessTable::ExpectancyTStat;
                }
            }
            ObjectiveVariant::B8Portfolio => {
                #[cfg(feature = "search-portfolio-objective")]
                {
                    cfg.score_portfolio_after_pruning = true;
                }
            }
            // ModeNative: the mode's own objective already IS this variant.
            // Nothing is written, and the report says so beside the coverage
            // count rather than leaving "the default ran" to look like a
            // variation.
            ObjectiveVariant::B5TerminalWealth | ObjectiveVariant::B6MonthlyConsistency => {}
        }
        Ok(())
    }
}

/// The axis-B half of the space report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantReport {
    pub variant: String,
    pub changes: ObjectiveDimension,
    pub expression: Expression,
    pub params: Vec<String>,
    pub drawable: bool,
    pub excluded_because: Option<String>,
    pub exact_form_requires: Option<String>,
    pub max_draws_per_session: Option<usize>,
    pub why_not_a_reparameterisation: String,
}

pub fn axis_b_report(caps: &Capabilities, scenario: ScenarioKind) -> Vec<VariantReport> {
    ObjectiveVariant::ALL
        .into_iter()
        .map(|v| {
            let spec = v.spec();
            let reason = v.exclusion_reason(caps, scenario);
            VariantReport {
                variant: v.label().to_string(),
                changes: spec.changes,
                expression: spec.expression,
                params: (0..v.param_count() as u8).map(|p| v.param_label(p)).collect(),
                drawable: reason.is_none(),
                excluded_because: reason.map(|r| {
                    ObjectiveError::VariantNotDrawable { variant: v, reason: r }.to_string()
                }),
                exact_form_requires: spec.exact_form_requires.map(|k| {
                    format!(
                        "the exact functional needs `{}` ({}); what runs today is the mode's own \
                         objective",
                        k.symbol(),
                        k.required_elsewhere_section()
                    )
                }),
                max_draws_per_session: spec.max_draws_per_session,
                why_not_a_reparameterisation: spec.why_not_a_reparameterisation.to_string(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::FROZEN_FIELDS;

    #[test]
    fn the_declared_set_is_ordered_and_complete() {
        for (i, v) in ObjectiveVariant::ALL.into_iter().enumerate() {
            assert_eq!(v.index(), i);
            assert_eq!(VARIANTS[i].id, v, "VARIANTS row {i} does not match the enum order");
            assert!(VARIANTS[i].params >= 1, "{} declares zero parameter levels", v.label());
        }
    }

    #[test]
    fn no_objective_writes_a_frozen_field() {
        for spec in VARIANTS {
            for field in spec.writes {
                assert!(
                    !FROZEN_FIELDS.contains(field),
                    "objective `{}` writes frozen field `{field}`",
                    spec.label
                );
            }
        }
        for d in RefusalDim::ALL {
            assert!(
                !FROZEN_FIELDS.contains(&d.field()),
                "refusal dimension `{}` writes frozen field `{}`",
                d.label(),
                d.field()
            );
        }
    }

    #[test]
    fn the_unconditional_expectancy_floor_is_not_a_refusal_dimension() {
        // §7.9. A loop that could move this would be varying the one gate that
        // says the average trade must make money.
        assert!(
            RefusalDim::ALL
                .into_iter()
                .all(|d| d.field() != "target_profile.min_net_expectancy_per_trade")
        );
        assert!(FROZEN_FIELDS.contains(&"target_profile.min_net_expectancy_per_trade"));
    }

    #[test]
    fn b0_is_the_only_capped_control() {
        let capped: Vec<_> = ObjectiveVariant::ALL
            .into_iter()
            .filter(|v| v.spec().max_draws_per_session.is_some())
            .collect();
        assert_eq!(capped, vec![ObjectiveVariant::B0ScoringTable]);
        assert_eq!(ObjectiveVariant::coverage_set().count(), VARIANT_COUNT - 1);
    }

    #[test]
    fn a_variant_whose_knob_is_absent_is_excluded_and_names_the_symbol() {
        let caps = Capabilities::compiled(); // every pending knob off
        let err = ObjectiveChoice::new(
            ObjectiveVariant::B4LabelHorizon,
            1,
            RefusalLevels::permissive(),
            &caps,
            ScenarioKind::Risky,
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("label_max_hold_bars"), "{text}");
        assert!(text.contains("EXCLUDED"), "{text}");
    }

    #[test]
    fn the_prop_firm_objective_is_not_drawable_in_a_risky_session() {
        let caps = Capabilities::all_for_tests();
        assert!(!ObjectiveVariant::B6MonthlyConsistency.drawable(&caps, ScenarioKind::Risky));
        assert!(ObjectiveVariant::B6MonthlyConsistency.drawable(&caps, ScenarioKind::PropFirm));
        assert!(ObjectiveVariant::B5TerminalWealth.drawable(&caps, ScenarioKind::Risky));
        assert!(!ObjectiveVariant::B5TerminalWealth.drawable(&caps, ScenarioKind::PropFirm));
    }

    #[test]
    fn b2s_bucket_count_multiplies_into_the_honest_n() {
        // §7.2: without this the conditional variant is the classic subgroup
        // overfit.
        assert_eq!(ObjectiveVariant::B2Conditional.hypothesis_multiplier(2), 5);
        assert_eq!(ObjectiveVariant::B1Selectivity.hypothesis_multiplier(0), 1);
    }

    #[test]
    fn refusal_levels_are_validated_not_clamped() {
        assert!(RefusalLevels::new([0, 0, 5, 0]).is_err());
        let r = RefusalLevels::new([1, 2, 2, 2]).unwrap();
        assert_eq!(r.value(RefusalDim::ExpectancySignificance), 2.0);
        assert_eq!(r.value(RefusalDim::WinRateFloor), 0.45);
        assert_eq!(r.value(RefusalDim::PayoffFloor), 2.0);
        assert_eq!(r.value(RefusalDim::CandidatePboCap), 0.5);
    }

    #[test]
    fn b4_sets_the_label_horizon_and_the_others_inherit_the_baseline() {
        let caps = Capabilities::all_for_tests();
        let b4 = ObjectiveChoice::new(
            ObjectiveVariant::B4LabelHorizon,
            2,
            RefusalLevels::permissive(),
            &caps,
            ScenarioKind::Risky,
        )
        .unwrap();
        assert_eq!(b4.label_horizon_bars(0), 480);
        let b1 = ObjectiveChoice::new(
            ObjectiveVariant::B1Selectivity,
            0,
            RefusalLevels::permissive(),
            &caps,
            ScenarioKind::Risky,
        )
        .unwrap();
        // The shipped EvaluationConfig ships 0, which discovery.rs reads as 35.
        assert_eq!(b1.label_horizon_bars(0), 35);
        assert_eq!(b1.label_horizon_bars(120), 120);
    }

    #[test]
    fn every_variant_names_a_dimension_it_actually_moves() {
        for spec in VARIANTS {
            if spec.id == ObjectiveVariant::B0ScoringTable {
                continue; // the declared control, and it says so
            }
            assert!(
                !spec.why_not_a_reparameterisation.is_empty(),
                "{} has no justification",
                spec.label
            );
        }
    }
}
