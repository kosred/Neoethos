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

impl Expression {
    /// Whether drawing this variant makes the run differ from the mode's own
    /// default objective.
    ///
    /// A [`Expression::ModeNative`] variant overrides nothing: a session in
    /// which it is the ONLY drawable variant sweeps axis A under one fixed,
    /// implicit objective — the default — while the coverage table records that
    /// an objective variant was drawn. That is the shape of "axis B ran" with
    /// none of the content, and [`axis_b_live_check`] refuses it.
    pub fn varies_the_objective(self) -> bool {
        matches!(self, Self::Overrides)
    }
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

/// The fields B1 writes, which depend on what the compiled search has.
///
/// Two of the three are unconditional: `min_trades_per_day` and
/// `TargetProfile::max_in_market` both exist on today's `DiscoveryConfig` and
/// `TargetProfile::evaluate` already enforces the second. The third — scoring
/// net per BAR IN MARKET — needs a knob that does not exist yet, so it is
/// declared in `exact_form_requires` and appears here only when the feature that
/// proves the symbol exists is on. A `writes` list that named a field the build
/// cannot write would be the same silent substitution this module exists to
/// prevent, one level up.
pub const B1_WRITES: &[&str] = if cfg!(feature = "search-in-market-fitness") {
    &[
        "min_trades_per_day",
        "target_profile.max_in_market",
        "fitness_table",
    ]
} else {
    &["min_trades_per_day", "target_profile.max_in_market"]
};
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
pub const B0_TABLE_LEVELS: &[&str] = &[
    "ga_fitness",
    "archive_score",
    "window_score",
    "quality_score",
];

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
        why_not_a_reparameterisation: "IT IS ONE — carried only as the named control the other eight are compared against.",
    },
    VariantSpec {
        id: ObjectiveVariant::B1Selectivity,
        label: "B1_selectivity",
        changes: ObjectiveDimension::EligiblePopulation,
        scenario: None,
        expression: Expression::Overrides,
        // ── THE AXIS-B LIVENESS ANCHOR. ────────────────────────────────────
        //
        // Empty, and that is the fix for the axis being inert in the shipped
        // build. The two levers that DEFINE this variant both exist on today's
        // `DiscoveryConfig`: `min_trades_per_day` (removing the volume floor)
        // and `TargetProfile::max_in_market`, which `TargetProfile::evaluate`
        // already enforces with a named rejection (`TooMuchTimeInMarket`).
        //
        // It used to require `FitnessTable::NetPerBarInMarket`, a symbol
        // `neoethos-search` does not have. Every other `Overrides` variant
        // needs a knob that does not exist yet either, so gating this one too
        // left the DEFAULT BUILD with `ModeNative` variants only — a variant
        // that writes nothing, i.e. an axis that runs the mode's own default and
        // reports that it explored the objective space. `axis_b_live_check`
        // now refuses to start such a session; this row is what lets it start.
        //
        // The clause that is NOT expressible is declared, not dropped: see
        // `exact_form_requires` immediately below, which is rendered beside the
        // variant in every space report.
        requires: &[],
        exact_form_requires: if cfg!(feature = "search-in-market-fitness") {
            None
        } else {
            Some(RequiredKnob::InMarketFitness)
        },
        writes: B1_WRITES,
        params: 3,
        max_draws_per_session: None,
        why_not_a_reparameterisation: "the average trade loses 4.15 pips, so the only exit is FEWER, DIFFERENT trades: the \
             volume floor is removed and a hard exposure budget is imposed, both of which change \
             WHICH trades are eligible rather than how the same population is summarised. (The \
             third clause of the exact form — scoring net per BAR IN MARKET so that being flat is \
             not penalised — needs a knob the compiled search does not have, and is declared in \
             `exact_form_requires` rather than silently omitted.)",
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
        why_not_a_reparameterisation: "expectancy is estimated WITHIN a conditioning set declared before the run; the \
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
        why_not_a_reparameterisation: "cost scales with trade COUNT, so scoring at the pessimistic band edge systematically \
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
        why_not_a_reparameterisation: "the triple-barrier horizon is the TARGET VARIABLE the prefilter ranks against and the \
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
        why_not_a_reparameterisation: "the unit is a PATH: two candidates with identical per-trade expectancy have different \
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
        why_not_a_reparameterisation: "the unit is a MONTH under path constraints. A drawdown-path constraint is not a \
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
        why_not_a_reparameterisation: "maximising the expectancy t-STATISTIC rather than the expectancy penalises the \
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
        why_not_a_reparameterisation: "the selection unit is the PORTFOLIO after correlation pruning, not the best single \
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
        Self::ALL
            .iter()
            .position(|v| *v == self)
            .expect("ALL covers the enum")
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
        Self::ALL
            .into_iter()
            .filter(|v| v.spec().max_draws_per_session.is_none())
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
            return Some(VariantExclusion::WrongScenario {
                requires: s,
                session: scenario,
            });
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
            Self::B1Selectivity => B1_MAX_IN_MARKET_LEVELS
                .get(p)
                .map(|v| format!("max_in_market={v:.2}")),
            Self::B2Conditional => B2_CONDITIONING_LEVELS
                .get(p)
                .map(|(n, b)| format!("{n}({b} buckets)")),
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
            Self::B2Conditional => B2_CONDITIONING_LEVELS
                .get(param as usize)
                .map(|(_, b)| *b)
                .unwrap_or(1),
            _ => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariantExclusion {
    MissingKnob(RequiredKnob),
    WrongScenario {
        requires: ScenarioKind,
        session: ScenarioKind,
    },
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
        Self {
            levels: [0; REFUSAL_DIM_COUNT],
        }
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
    RefusalLevelOutOfRange {
        dim: RefusalDim,
        level: u8,
        arity: usize,
    },
    ParamOutOfRange {
        variant: ObjectiveVariant,
        param: u8,
        arity: usize,
    },
    VariantNotDrawable {
        variant: ObjectiveVariant,
        reason: VariantExclusion,
    },
    /// Every drawable variant has spent its `max_draws_per_session`.
    ///
    /// The proposer used to handle this by falling back to the full drawable
    /// list, which re-admitted the capped control past its own ceiling with no
    /// counter and no census line. Named here so the slot is refused, counted
    /// and named instead.
    AllVariantsCapped { drawable: Vec<String> },
}

impl std::fmt::Display for ObjectiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RefusalLevelOutOfRange { dim, level, arity } => write!(
                f,
                "refusal dimension `{}` has {arity} levels; level {level} does not exist",
                dim.label()
            ),
            Self::ParamOutOfRange {
                variant,
                param,
                arity,
            } => write!(
                f,
                "objective `{}` has {arity} parameter levels; level {param} does not exist",
                variant.label()
            ),
            Self::AllVariantsCapped { drawable } => write!(
                f,
                "every drawable objective variant [{}] has spent its max_draws_per_session, so \
                 there is no variant left to draw. The slot is REFUSED and counted rather than \
                 re-admitting a capped control past its own ceiling, which would run a \
                 configuration nobody chose under a name somebody did",
                drawable.join(", ")
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
        Ok(Self {
            variant,
            param,
            refusals,
        })
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
                    cfg.fitness_table = neoethos_search::scoring::FitnessTable::from_name(
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
                params: (0..v.param_count() as u8)
                    .map(|p| v.param_label(p))
                    .collect(),
                drawable: reason.is_none(),
                excluded_because: reason.map(|r| {
                    ObjectiveError::VariantNotDrawable {
                        variant: v,
                        reason: r,
                    }
                    .to_string()
                }),
                exact_form_requires: spec.exact_form_requires.map(|k| {
                    // The sentence must name what ACTUALLY runs, and that
                    // differs by expression. Saying "the mode's own objective"
                    // for an `Overrides` variant would describe a run nobody
                    // made — the approximation would be stated, but stated
                    // wrongly, which is worse than not stating it.
                    match spec.expression {
                        Expression::ModeNative => format!(
                            "the exact functional needs `{}` ({}); what runs today is the mode's \
                             own objective",
                            k.symbol(),
                            k.required_elsewhere_section()
                        ),
                        Expression::Overrides => format!(
                            "the exact functional needs `{}` ({}); what runs today is the \
                             expressible half of it — the overrides [{}] — and NOT the missing \
                             term",
                            k.symbol(),
                            k.required_elsewhere_section(),
                            spec.writes.join(", ")
                        ),
                    }
                }),
                max_draws_per_session: spec.max_draws_per_session,
                why_not_a_reparameterisation: spec.why_not_a_reparameterisation.to_string(),
            }
        })
        .collect()
}

// ───────────────────────────────────────────────────────────────────────────
// Is axis B actually running? (§7.0, and the operator's "BOTH TOGETHER".)
// ───────────────────────────────────────────────────────────────────────────

/// The two scenarios, as a table. `ScenarioKind` is owned by [`crate::goals`];
/// this is the local enumeration the coverage arithmetic needs and it is
/// asserted exhaustive by [`tests::the_scenario_table_is_exhaustive`].
pub const SCENARIOS: [ScenarioKind; 2] = [ScenarioKind::Risky, ScenarioKind::PropFirm];

/// Why the objective axis would not actually vary in this build.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisBInert {
    pub scenario: ScenarioKind,
    /// Variants that CAN be drawn — possibly non-empty, and still inert.
    pub drawable: Vec<&'static str>,
    /// Why each variant that would have varied the objective is unavailable.
    pub excluded: Vec<String>,
}

impl std::fmt::Display for AxisBInert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AXIS B IS INERT in this build: of the objective variants drawable for the {:?} \
             scenario [{}], not one overrides anything — every drawable variant is the mode's own \
             default objective under a name. The loop would sweep the SEARCH CONFIGURATION under \
             one fixed objective while its coverage table recorded that it explored the objective \
             space, and U4 would then certify a space it never searched on the axis the measured \
             base rate says the answer must live on (expectancy was -4.15 pips per trade in EVERY \
             exit configuration tested). Refusing to start rather than reporting a refutation \
             nobody could act on. What is missing: {}",
            self.scenario,
            self.drawable.join(", "),
            if self.excluded.is_empty() {
                "nothing is excluded, which means the declared table itself carries no \
                 overriding variant — that is a defect in VARIANTS, not in the build"
                    .to_string()
            } else {
                self.excluded.join("; ")
            }
        )
    }
}

impl std::error::Error for AxisBInert {}

/// The liveness rule, on its own so it can be exercised over a table: a set of
/// variants carries the objective axis iff at least one of them is **uncapped**
/// and **overrides** something.
///
/// Uncapped, because [`ObjectiveVariant::B0ScoringTable`] is the declared
/// control with `max_draws_per_session = 1`: one sweep out of a session cannot
/// carry an axis. Overriding, because a [`Expression::ModeNative`] variant runs
/// the mode's own objective and changes nothing.
pub fn carries_the_axis(variants: &[ObjectiveVariant]) -> bool {
    variants.iter().any(|v| {
        v.spec().expression.varies_the_objective() && v.spec().max_draws_per_session.is_none()
    })
}

/// The drawable axis-B set, or a refusal naming why the axis cannot vary.
///
/// **This is the gate that keeps the operator's second axis from being skipped.**
/// It is not enough that *some* variant is drawable — see [`carries_the_axis`]
/// for the rule and why it is that rule.
pub fn axis_b_live_check(
    caps: &Capabilities,
    scenario: ScenarioKind,
) -> Result<Vec<ObjectiveVariant>, AxisBInert> {
    let drawable: Vec<ObjectiveVariant> = ObjectiveVariant::ALL
        .into_iter()
        .filter(|v| v.drawable(caps, scenario))
        .collect();

    if carries_the_axis(&drawable) {
        return Ok(drawable);
    }

    let excluded = ObjectiveVariant::ALL
        .into_iter()
        .filter(|v| carries_the_axis(std::slice::from_ref(v)))
        .filter_map(|v| {
            v.exclusion_reason(caps, scenario).map(|r| {
                ObjectiveError::VariantNotDrawable {
                    variant: v,
                    reason: r,
                }
                .to_string()
            })
        })
        .collect();

    Err(AxisBInert {
        scenario,
        drawable: drawable.iter().map(|v| v.label()).collect(),
        excluded,
    })
}

/// The axis-B half of U4, computed against the **declared** drawable set rather
/// than against whatever happens to be in the coverage counters. Produced by
/// [`axis_b_coverage`].
#[derive(Debug, Clone, PartialEq)]
pub struct AxisBCoverage {
    /// Any variant at all was credited at least once. **False is enough on its
    /// own to fail U4.**
    pub axis_ran: bool,
    /// The scenario whose native objective was actually covered, when one was.
    pub scenario: Option<ScenarioKind>,
    /// `(variant, sweeps drawn)` for every declared-drawable variant below the
    /// bar. Non-empty whenever `axis_ran` is false.
    pub under_drawn: Vec<(&'static str, usize)>,
}

impl AxisBCoverage {
    /// U4's axis-B conjunct.
    pub fn satisfied(&self) -> bool {
        self.axis_ran && self.scenario.is_some() && self.under_drawn.is_empty()
    }

    /// The line U4 prints, whichever way it went.
    pub fn detail(&self, min_draws: usize) -> String {
        if !self.axis_ran {
            return format!(
                "NO objective variant was ever credited. Axis B — the half the measured base rate \
                 says the answer must live on — produced no runs, so this space was not searched \
                 on it and cannot be refuted on it. (An empty axis-B coverage list satisfies \
                 'every variant was drawn {min_draws} times' only vacuously; that is the trap this \
                 conjunct exists to fail.)"
            );
        }
        if self.scenario.is_none() {
            return format!(
                "no scenario's own objective variant reached {min_draws} sweep(s)' worth of \
                 draws, so the objective the goal set is actually defined by was never exercised: \
                 [{}]",
                self.render_under()
            );
        }
        if self.under_drawn.is_empty() {
            format!(
                "every declared-drawable objective variant reached {min_draws} sweep(s)' worth of \
                 draws (scenario-native objective: {:?})",
                self.scenario
            )
        } else {
            format!(
                "under-drawn objective variants (< {min_draws} sweeps): [{}]",
                self.render_under()
            )
        }
    }

    fn render_under(&self) -> String {
        self.under_drawn
            .iter()
            .map(|(l, n)| format!("{l} ({n})"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Evaluate [`AxisBCoverage`] for a session.
///
/// # The hole this closes
///
/// U4 says *a space is not refuted if it was not searched*. Reading the coverage
/// counters alone cannot say that, because the counters only contain keys that
/// were **credited**: an axis that produced no runs at all contributes no keys,
/// "no key is under-drawn" is then vacuously true, and the loop certifies a
/// space it never explored on the axis that matters most. `session.coverage
/// .is_empty()` does not catch it either — axis A is credited on every proposal,
/// so the counters are non-empty while the objective axis is untouched. The
/// DECLARED set is the only thing that knows what should have been drawn, so it
/// is the thing U4 is evaluated against.
///
/// `sweeps_of` is the session's own fold — `coverage.get("objective", label)
/// .sweeps` — passed as a closure so this function has no session dependency and
/// can be exercised from a table in a test.
///
/// The scenario is **inferred from the journal**, not passed in: exactly one of
/// the scenario-specific variants is drawable in any session (they are
/// `ModeNative` and always drawable in their own scenario), so the scenario
/// whose native variants are all covered is the scenario the session ran. If
/// none is, the session never exercised the objective its goal set is defined
/// by, and U4 fails saying so — which is the honest answer either way and needs
/// no extra argument that a caller could get wrong.
pub fn axis_b_coverage(
    caps: &Capabilities,
    min_draws: usize,
    sweeps_of: &dyn Fn(&str) -> usize,
) -> AxisBCoverage {
    let axis_ran = ObjectiveVariant::ALL
        .into_iter()
        .any(|v| sweeps_of(v.label()) > 0);

    let expressible = |v: ObjectiveVariant| v.spec().requires.iter().all(|k| k.available(caps));

    // The scenario-agnostic half: every one of them must clear the bar.
    let mut under_drawn: Vec<(&'static str, usize)> = ObjectiveVariant::coverage_set()
        .filter(|v| v.spec().scenario.is_none() && expressible(*v))
        .filter_map(|v| {
            let n = sweeps_of(v.label());
            (n < min_draws).then_some((v.label(), n))
        })
        .collect();

    // The scenario-native half: exactly one scenario's group must clear it.
    let mut scenario = None;
    let mut best_gap: Option<Vec<(&'static str, usize)>> = None;
    for k in SCENARIOS {
        let gap: Vec<(&'static str, usize)> = ObjectiveVariant::coverage_set()
            .filter(|v| v.spec().scenario == Some(k) && expressible(*v))
            .filter_map(|v| {
                let n = sweeps_of(v.label());
                (n < min_draws).then_some((v.label(), n))
            })
            .collect();
        let group_exists = ObjectiveVariant::coverage_set()
            .any(|v| v.spec().scenario == Some(k) && expressible(v));
        if !group_exists {
            continue;
        }
        if gap.is_empty() {
            scenario = Some(k);
            best_gap = Some(Vec::new());
            break;
        }
        if best_gap.as_ref().is_none_or(|g| gap.len() < g.len()) {
            best_gap = Some(gap);
        }
    }
    under_drawn.extend(best_gap.unwrap_or_default());
    under_drawn.sort_unstable();
    under_drawn.dedup();

    AxisBCoverage {
        axis_ran,
        scenario,
        under_drawn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::FROZEN_FIELDS;

    #[test]
    fn the_declared_set_is_ordered_and_complete() {
        for (i, v) in ObjectiveVariant::ALL.into_iter().enumerate() {
            assert_eq!(v.index(), i);
            assert_eq!(
                VARIANTS[i].id, v,
                "VARIANTS row {i} does not match the enum order"
            );
            assert!(
                VARIANTS[i].params >= 1,
                "{} declares zero parameter levels",
                v.label()
            );
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

    // ── DEFECT 3, first half: axis B must RUN in the shipped default build ──

    #[test]
    fn the_scenario_table_is_exhaustive() {
        // `axis_b_coverage` infers the session's scenario by trying each entry
        // of SCENARIOS. A scenario missing from the table would be a scenario
        // whose native objective U4 could never see, so it would certify a
        // space it never searched — the exact failure this wave closes.
        // Exhaustive BY COMPILER: adding a `ScenarioKind` variant makes this
        // match fail to build, which is the only way a hand-written table like
        // SCENARIOS stays honest.
        for k in SCENARIOS {
            let _: &str = match k {
                ScenarioKind::Risky => "risky",
                ScenarioKind::PropFirm => "prop_firm",
            };
        }
        assert_eq!(SCENARIOS.len(), 2);
        assert_ne!(SCENARIOS[0], SCENARIOS[1]);
        // …and every scenario-specific variant names one of them.
        for spec in VARIANTS {
            if let Some(k) = spec.scenario {
                assert!(
                    SCENARIOS.contains(&k),
                    "{} names a scenario not in SCENARIOS",
                    spec.label
                );
            }
        }
    }

    #[test]
    fn axis_b_is_live_in_the_default_build() {
        // THE REGRESSION TEST FOR DEFECT 3. `Capabilities::compiled()` with the
        // shipped feature set (all pending knobs OFF) must still yield an
        // objective axis that actually varies — under BOTH goal scenarios.
        //
        // Before this wave every `Overrides` variant required a symbol
        // `neoethos-search` does not have, so the default build's drawable set
        // was `{B5}` (Risky) or `{B6}` (PropFirm): one ModeNative variant that
        // writes nothing. The loop swept axis A under the mode's own objective
        // and its coverage table said it had explored the objective space.
        let caps = Capabilities::compiled();
        for scenario in SCENARIOS {
            let drawable = axis_b_live_check(&caps, scenario)
                .unwrap_or_else(|e| panic!("axis B is inert in the shipped build: {e}"));
            assert!(
                carries_the_axis(&drawable),
                "{scenario:?}: drawable set {:?} carries no uncapped overriding variant",
                drawable.iter().map(|v| v.label()).collect::<Vec<_>>()
            );
            // …and it is not merely the capped control doing the work.
            assert!(
                drawable
                    .iter()
                    .any(|v| *v == ObjectiveVariant::B1Selectivity),
                "{scenario:?}: B1 is the variant whose levers the compiled search actually has"
            );
        }
    }

    #[test]
    fn b1_is_expressible_with_symbols_the_compiled_search_has() {
        // The claim the liveness fix rests on: B1's two levers are real fields
        // and `ObjectiveChoice::apply` writes them under the DEFAULT feature
        // set. If `min_trades_per_day` or `max_in_market` ever moved behind a
        // feature, this fails rather than the axis quietly going inert again.
        let caps = Capabilities::compiled();
        let choice = ObjectiveChoice::new(
            ObjectiveVariant::B1Selectivity,
            1,
            RefusalLevels::permissive(),
            &caps,
            ScenarioKind::Risky,
        )
        .expect("B1 must be drawable with no pending knob compiled in");
        let mut cfg = DiscoveryConfig {
            min_trades_per_day: 7.5,
            ..Default::default()
        };
        choice.apply(&mut cfg).unwrap();
        assert_eq!(
            cfg.min_trades_per_day, 0.0,
            "the volume floor must be REMOVED, not lowered"
        );
        assert_eq!(cfg.target_profile.max_in_market, B1_MAX_IN_MARKET_LEVELS[1]);
        // And the term that is NOT expressible is declared rather than dropped.
        assert_eq!(
            ObjectiveVariant::B1Selectivity.spec().exact_form_requires,
            Some(RequiredKnob::InMarketFitness)
        );
        assert!(
            !B1_WRITES.contains(&"fitness_table"),
            "writes must not name a field it cannot write"
        );
    }

    #[test]
    fn a_mode_native_only_axis_does_not_carry_the_axis() {
        // The rule itself, over a table. ModeNative overrides nothing, and the
        // capped control is one sweep out of a session.
        assert!(!carries_the_axis(&[ObjectiveVariant::B5TerminalWealth]));
        assert!(!carries_the_axis(&[ObjectiveVariant::B6MonthlyConsistency]));
        assert!(!carries_the_axis(&[ObjectiveVariant::B0ScoringTable]));
        assert!(!carries_the_axis(&[]));
        assert!(carries_the_axis(&[ObjectiveVariant::B1Selectivity]));
        assert!(carries_the_axis(&[
            ObjectiveVariant::B5TerminalWealth,
            ObjectiveVariant::B1Selectivity
        ]));
    }

    #[test]
    fn the_inert_refusal_names_the_missing_symbols() {
        // The message an operator would actually get. Built directly, because
        // the shipped build can no longer produce the condition — which is the
        // point, and is why the renderer still has to be exercised.
        let caps = Capabilities::compiled();
        let inert = AxisBInert {
            scenario: ScenarioKind::Risky,
            drawable: vec![ObjectiveVariant::B5TerminalWealth.label()],
            excluded: vec![
                ObjectiveError::VariantNotDrawable {
                    variant: ObjectiveVariant::B4LabelHorizon,
                    reason: ObjectiveVariant::B4LabelHorizon
                        .exclusion_reason(&caps, ScenarioKind::Risky)
                        .unwrap(),
                }
                .to_string(),
            ],
        };
        let text = inert.to_string();
        assert!(text.contains("AXIS B IS INERT"), "{text}");
        assert!(text.contains("label_max_hold_bars"), "{text}");
        assert!(text.contains("B5_terminal_wealth"), "{text}");
    }

    // ── DEFECT 3, second half: U4 may not certify an unsearched axis ────────

    #[test]
    fn u4_cannot_certify_an_axis_that_produced_no_runs() {
        // THE REGRESSION TEST FOR THE U4 HALF OF DEFECT 3.
        //
        // Reading the coverage counters alone, "no credited objective key is
        // under-drawn" is TRUE when nothing was credited at all — the classic
        // "for all x in {}" trap. Evaluated against the DECLARED drawable set it
        // is false, and says so in words an operator can act on.
        let caps = Capabilities::compiled();
        let nothing_ran = axis_b_coverage(&caps, crate::space::B_MIN_DRAWS, &|_: &str| 0);
        assert!(!nothing_ran.axis_ran);
        assert!(
            !nothing_ran.satisfied(),
            "U4 certified an axis that produced no runs"
        );
        assert!(!nothing_ran.under_drawn.is_empty());
        let detail = nothing_ran.detail(crate::space::B_MIN_DRAWS);
        assert!(detail.contains("produced no runs"), "{detail}");
        assert!(detail.contains("vacuously"), "{detail}");
    }

    #[test]
    fn u4_needs_every_declared_drawable_variant_and_the_scenario_native_one() {
        let caps = Capabilities::compiled();
        let min = crate::space::B_MIN_DRAWS;

        // Everything drawn enough: satisfied, and the scenario is inferred.
        let all = axis_b_coverage(&caps, min, &|_: &str| min);
        assert!(all.satisfied(), "{:?}", all);
        assert!(all.scenario.is_some());

        // The scenario-agnostic half covered but the goal set's own objective
        // never drawn: NOT satisfied. A session that never ran the objective
        // its goals are defined by has not searched the space.
        let no_native = axis_b_coverage(&caps, min, &|label: &str| {
            if label == ObjectiveVariant::B5TerminalWealth.label()
                || label == ObjectiveVariant::B6MonthlyConsistency.label()
            {
                0
            } else {
                min
            }
        });
        assert!(no_native.axis_ran);
        assert!(!no_native.satisfied());
        assert!(no_native.scenario.is_none());

        // One variant a single sweep short: NOT satisfied, and named with its
        // count so the report says which question to ask next.
        let short = axis_b_coverage(&caps, min, &|label: &str| {
            if label == ObjectiveVariant::B1Selectivity.label() {
                min - 1
            } else {
                min
            }
        });
        assert!(!short.satisfied());
        assert!(
            short
                .under_drawn
                .contains(&(ObjectiveVariant::B1Selectivity.label(), min - 1)),
            "{:?}",
            short.under_drawn
        );
        assert!(
            short.detail(min).contains("B1_selectivity (2)"),
            "{}",
            short.detail(min)
        );
    }

    #[test]
    fn u4_never_demands_a_variant_the_build_cannot_draw() {
        // The other side of the same coin: a space is not refuted if it was not
        // searched, but a variant that is EXCLUDED and named in the space report
        // was never searchable, so demanding it would make R2 unreachable
        // forever — the loop could then never say "unreachable", which is the
        // one thing this project has never been able to say.
        let caps = Capabilities::compiled();
        let cov = axis_b_coverage(&caps, crate::space::B_MIN_DRAWS, &|_: &str| {
            crate::space::B_MIN_DRAWS
        });
        assert!(cov.satisfied());
        for v in ObjectiveVariant::ALL {
            if v.exclusion_reason(&caps, ScenarioKind::Risky).is_some()
                && v.exclusion_reason(&caps, ScenarioKind::PropFirm).is_some()
            {
                assert!(
                    !cov.under_drawn.iter().any(|(l, _)| *l == v.label()),
                    "U4 demands `{}`, which no build can draw",
                    v.label()
                );
            }
        }
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
